#include "ninfer-brain.h"
#include "brain-sampling.h"
#include "voice-output.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <thread>
#include <chrono>

namespace {
constexpr size_t hidden_width = 5120;
constexpr uint32_t hidden_layer = 16;
std::atomic<uint64_t> next_session{0};

ninfer::CancellationView cancellation(std::atomic<bool> & cancelled) {
    return ninfer::CancellationView([&cancelled] { return cancelled.load(); });
}

ninfer::OwnedMedia lossless_bitmap(const mtmd_bitmap * bitmap) {
    const uint32_t width = mtmd_bitmap_get_nx(bitmap), height = mtmd_bitmap_get_ny(bitmap);
    const size_t stride = (size_t(width) * 3 + 3) & ~size_t(3);
    ninfer::OwnedMedia image;
    image.kind = ninfer::MediaKind::Image;
    image.media_type = "image/bmp"; image.source_name = "realtime-image.bmp";
    image.bytes.resize(54 + stride * height);
    auto put = [&](size_t offset, uint32_t value, int bytes = 4) {
        for (int i = 0; i < bytes; ++i) { image.bytes[offset + i] = uint8_t(value >> (8 * i)); }
    };
    image.bytes[0] = 'B'; image.bytes[1] = 'M';
    put(2, image.bytes.size()); put(10, 54); put(14, 40); put(18, width); put(22, height);
    put(26, 1, 2); put(28, 24, 2); put(34, stride * height);
    const auto * rgb = mtmd_bitmap_get_data(bitmap);
    for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
        const size_t source = (y * width + x) * 3, dest = 54 + (height - 1 - y) * stride + x * 3;
        image.bytes[dest] = rgb[source + 2]; image.bytes[dest + 1] = rgb[source + 1]; image.bytes[dest + 2] = rgb[source];
    }
    return image;
}

} // namespace

struct ninfer_brain_session::encoded_prompt {
    std::vector<ninfer::TokenId> tokens;
    std::vector<ninfer::InputEmbeddingSpan> embeddings;
    std::vector<ninfer::RawImageSpan> images;
};

struct ninfer_brain_session::logical_state final : external_state {
    std::string key;
    std::shared_ptr<const encoded_prompt> prefix;
    std::string partial_prefix, partial_marker;
    std::vector<float> partial_rows;
    std::vector<uint32_t> execution_frontiers;
};

ninfer_brain_session::ninfer_brain_session(ninfer::Engine & shared_engine, const std::string & package,
                                         const frankie_options & config) :
    brain_session(package, config, brain_backend::external), engine(shared_engine), state(std::make_shared<logical_state>()) {
    if (engine.options().hidden_layer != hidden_layer || engine.options().max_context < context_tokens()) {
        throw std::runtime_error("NInfer voice requires layer-16 features and the configured voice context capacity");
    }
    if (!engine.options().context_cache.enabled) {
        throw std::runtime_error("Frankie requires native context caching for audio precommit and interruption rollback");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    for (const std::string text : {"<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n", "<think>\n</think>\n", "Hmm", "Yeah", "Wow", "Aw", "Nothing"}) {
        if (common_tokenize(vocab, text, false, true) != engine.tokenize_text(text)) {
            throw std::runtime_error("NInfer brain tokenizer does not match the Frankie speech bundle");
        }
    }
    state->key = "frankie-voice-" + std::to_string(++next_session);
}

ninfer_brain_session::~ninfer_brain_session() {
    try { discard_session(); }
    catch (const std::exception & error) { std::cerr << "voice cache cleanup: " << error.what() << "\n"; }
}

void ninfer_brain_session::discard_session() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!engine.discard_session(state->key)) {
        if (!engine.is_available() || std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("NInfer voice cache remains borrowed after cancellation");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void ninfer_brain_session::record_frontier(uint32_t frontier) {
    const auto position = std::lower_bound(state->execution_frontiers.begin(), state->execution_frontiers.end(), frontier);
    if (position == state->execution_frontiers.end() || *position != frontier) { state->execution_frontiers.insert(position, frontier); }
}

ninfer_brain_session::encoded_prompt ninfer_brain_session::encode_prompt(
        const request & input, const std::string & prompt, const std::map<std::string, size_t> & pending_audio) const {
    if (prompt.size() > 4 * 1024 * 1024 || input.chat.messages.size() > max_messages ||
        input.chat.tools.size() > 32 || input.audio_rows.size() > max_audio_segments || input.images.size() > 4) {
        throw std::runtime_error("voice prompt bounds");
    }
    encoded_prompt result;
    auto text = [&](const std::string & value) {
        auto ids = engine.tokenize_text(value);
        result.tokens.insert(result.tokens.end(), ids.begin(), ids.end());
    };
    size_t offset = 0;
    for (;;) {
        const auto marker = prompt.find("[FRANKIE_", offset);
        if (marker == std::string::npos) { text(prompt.substr(offset)); break; }
        text(prompt.substr(offset, marker - offset));
        const auto end = prompt.find(']', marker);
        if (end == std::string::npos) { throw std::runtime_error("broken media marker"); }
        const auto key = prompt.substr(marker, end - marker + 1);
        const auto audio = input.audio_rows.find(key);
        const auto future = pending_audio.find(key);
        if (audio != input.audio_rows.end()) {
            const auto & values = audio->second;
            if (values.empty() || values.size() % hidden_width || values.size() / hidden_width > audio_row_limit() ||
                !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("audio embedding shape or finite-value bounds");
            }
            ninfer::InputEmbeddingSpan span;
            span.begin = result.tokens.size(); span.width = hidden_width; span.values = values;
            result.embeddings.push_back(std::move(span));
            result.tokens.resize(result.tokens.size() + values.size() / hidden_width, 0);
        } else if (future != pending_audio.end()) {
            if (!future->second || future->second > audio_row_limit()) { throw std::runtime_error("pending audio row bounds"); }
            result.tokens.resize(result.tokens.size() + future->second, 0);
        } else if (input.images.count(key)) {
            ninfer::PreparationControl control;
            control.cancellation = ninfer::CancellationView([this] { return cancelled.load(); });
            auto image = engine.prepare_image(lossless_bitmap(input.images.at(key).get()), control);
            ninfer::RawImageSpan span;
            span.begin = result.tokens.size(); span.image = std::move(image);
            const auto image_tokens = span.image.token_ids();
            result.tokens.insert(result.tokens.end(), image_tokens.begin(), image_tokens.end());
            result.images.push_back(std::move(span));
        } else { throw std::runtime_error("unknown media marker"); }
        if (result.tokens.size() > context_tokens()) { throw std::runtime_error("voice prompt context exceeded"); }
        offset = end + 1;
    }
    return result;
}

ninfer::PreparedPrompt ninfer_brain_session::prepare(const encoded_prompt & prompt, bool thinking, std::optional<uint32_t> checkpoint) const {
    ninfer::RawPromptOptions raw;
    raw.session_key = state->key;
    raw.enable_thinking = thinking;
    raw.rewrite_checkpoint = checkpoint.value_or(uint32_t(prompt.tokens.size()));
    for (const auto frontier : state->execution_frontiers) {
        if (frontier <= prompt.tokens.size()) { raw.rewrite_execution_frontiers.push_back(frontier); }
    }
    return engine.prepare_embeddings(prompt.tokens, prompt.embeddings, std::move(raw), prompt.images);
}

void ninfer_brain_session::cache_prefix(encoded_prompt prompt) {
    auto prepared = prepare(prompt, false);
    (void) engine.evaluate_prompt(std::move(prepared), {}, cancellation(cancelled));
    record_frontier(prompt.tokens.size());
    state->prefix = std::make_shared<const encoded_prompt>(std::move(prompt));
}

size_t ninfer_brain_session::prompt_tokens(const request & input, const std::map<std::string, size_t> & pending) const {
    const auto formatted = common_chat_templates_apply(templates.get(), input.chat);
    return encode_prompt(input, formatted.prompt, pending).tokens.size();
}

void ninfer_brain_session::reset(bool preserve_checkpoint) {
    auto lock = lock_compute();
    if (!preserve_checkpoint) { discard_session(); }
    auto fresh = std::make_shared<logical_state>();
    fresh->key = preserve_checkpoint ? state->key : "frankie-voice-" + std::to_string(++next_session);
    if (preserve_checkpoint) { fresh->prefix = state->prefix; fresh->execution_frontiers = state->execution_frontiers; }
    state = std::move(fresh);
}

brain_session::saved_state ninfer_brain_session::suspend_state() {
    auto lock = lock_compute();
    saved_state saved{};
    saved.external = std::make_shared<logical_state>(*state);
    return saved;
}

void ninfer_brain_session::restore_state(saved_state saved) {
    auto lock = lock_compute();
    auto restored = std::dynamic_pointer_cast<logical_state>(saved.external);
    if (!restored) { throw std::runtime_error("invalid NInfer voice checkpoint"); }
    state = std::make_shared<logical_state>(*restored);
    // Exact prompt identity selects the native KV/GDN/MTP checkpoint at the next
    // submission. Nothing is inferred from the cancelled branch's visible text.
}

void ninfer_brain_session::precommit(const request & input, const std::string & marker,
                                     const std::vector<float> & rows, size_t count) {
    auto lock = lock_compute();
    if (!count || rows.size() % hidden_width || count >= rows.size() / hidden_width || count >= audio_row_limit()) {
        throw std::runtime_error("precommit row bounds");
    }
    auto snapshot = input;
    const auto formatted = common_chat_templates_apply(templates.get(), snapshot.chat);
    const auto cut = formatted.prompt.find(marker);
    if (cut == std::string::npos || formatted.prompt.find(marker, cut + 1) != std::string::npos) {
        throw std::runtime_error("precommit marker missing or ambiguous");
    }
    const auto prefix = formatted.prompt.substr(0, cut);
    if (state->partial_marker == marker && state->partial_prefix == prefix && count <= state->partial_rows.size() / hidden_width) { return; }
    std::vector<float> committed(rows.begin(), rows.begin() + count * hidden_width);
    if (state->partial_marker == marker && state->partial_prefix == prefix) {
        std::copy(state->partial_rows.begin(), state->partial_rows.end(), committed.begin());
    }
    snapshot.audio_rows[marker] = committed;
    cache_prefix(encode_prompt(snapshot, prefix + marker));
    state->partial_marker = marker; state->partial_prefix = prefix; state->partial_rows = std::move(committed);
}

void ninfer_brain_session::finish_audio(const request & input, const std::string & marker, std::vector<float> & rows) {
    auto lock = lock_compute();
    const auto formatted = common_chat_templates_apply(templates.get(), input.chat);
    if (marker == state->partial_marker && formatted.prompt.compare(0, state->partial_prefix.size(), state->partial_prefix) == 0 &&
        formatted.prompt.compare(state->partial_prefix.size(), marker.size(), marker) == 0) {
        if (rows.size() < state->partial_rows.size() + hidden_width) { throw std::runtime_error("final audio shorter than precommit"); }
        std::copy(state->partial_rows.begin(), state->partial_rows.end(), rows.begin());
    }
}

brain_session::listener_reaction ninfer_brain_session::probe_listener(const request & input) {
    auto lock = lock_compute();
    if (state->partial_marker.empty() || state->partial_rows.size() < 8 * hidden_width) { return {}; }
    auto snapshot = input;
    snapshot.chat.enable_thinking = false; snapshot.chat.add_generation_prompt = true;
    snapshot.audio_rows[state->partial_marker] = state->partial_rows;
    const auto formatted = common_chat_templates_apply(templates.get(), snapshot.chat);
    const auto marker = formatted.prompt.find(state->partial_marker);
    if (marker == std::string::npos) { throw std::runtime_error("listener probe has no active audio marker"); }
    const std::string ask = "\n(You are listening while the user is still talking. Which one-word listener reaction fits what they have said so far? Answer with exactly one of: Hmm, Yeah, Wow, Aw, Nothing.)";
    auto prompt = encode_prompt(snapshot, formatted.prompt.substr(0, marker + state->partial_marker.size()) + ask +
                                        formatted.prompt.substr(marker + state->partial_marker.size()));
    std::vector<ninfer::TokenId> ids;
    for (const char * word : {"Hmm", "Yeah", "Wow", "Aw", "Nothing"}) {
        const auto token = engine.tokenize_text(word);
        if (token.size() != 1) { throw std::runtime_error("listener reaction must be one token"); }
        ids.push_back(token.front());
    }
    const auto evaluated = engine.evaluate_prompt(prepare(prompt, false, state->prefix ? std::optional<uint32_t>(state->prefix->tokens.size()) : std::nullopt), ids, cancellation(cancelled));
    if (evaluated.logits.size() != ids.size()) { throw std::runtime_error("listener logits missing"); }
    listener_reaction result;
    const auto maximum = *std::max_element(evaluated.logits.begin(), evaluated.logits.end());
    float sum = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!std::isfinite(evaluated.logits[i])) { throw std::runtime_error("invalid listener logits"); }
        result.probabilities[i] = std::exp(evaluated.logits[i] - maximum); sum += result.probabilities[i];
    }
    for (auto & value : result.probabilities) { value /= sum; }
    result.index = std::max_element(result.probabilities.begin(), result.probabilities.end()) - result.probabilities.begin();
    return result;
}

void ninfer_brain_session::warm_prefix(common_chat_templates_inputs input) {
    auto lock = lock_compute();
    if (input.messages.size() != 1 || input.messages.front().role != "system") { throw std::runtime_error("warm prefix requires only system context"); }
    input.add_generation_prompt = false;
    common_chat_msg boundary; boundary.role = "user"; boundary.content = "[FRANKIE_WARM_BOUNDARY]";
    input.messages.push_back(boundary);
    auto formatted = common_chat_templates_apply(templates.get(), input);
    const auto cut = formatted.prompt.find(boundary.content);
    if (cut == std::string::npos || formatted.prompt.find(boundary.content, cut + 1) != std::string::npos) {
        throw std::runtime_error("ambiguous warm prefix boundary");
    }
    formatted.prompt.resize(cut);
    cache_prefix(encode_prompt({}, formatted.prompt));
}

brain_session::response ninfer_brain_session::generate(const request & input, const stream_callback & on_text,
                                                       const stage_callback & on_stage) {
    auto lock = lock_compute();
    if (input.reasoning_budget < 0 || input.reasoning_budget > 32768) { throw std::runtime_error("reasoning budget bounds"); }
    const auto formatted = common_chat_templates_apply(templates.get(), input.chat);
    auto prompt = encode_prompt(input, formatted.prompt);
    if (prompt.tokens.size() + input.generation_tokens() > context_tokens()) {
        throw std::runtime_error("reasoning or generation context budget exceeded");
    }
    state->execution_frontiers.erase(std::upper_bound(state->execution_frontiers.begin(), state->execution_frontiers.end(), prompt.tokens.size()), state->execution_frontiers.end());
    record_frontier(prompt.tokens.size());
    state->prefix = std::make_shared<const encoded_prompt>(prompt);
    state->partial_marker.clear(); state->partial_prefix.clear(); state->partial_rows.clear();
    const size_t available = context_tokens() - prompt.tokens.size();
    const size_t maximum = std::min(available, input.answer_limit ? size_t(input.answer_limit) + input.reasoning_budget + 16 : available);
    ninfer::RequestOptions options;
    const auto sampling = frankie_brain_sampling(input.chat.enable_thinking);
    options.execution.sampling.temperature = sampling.temp;
    options.execution.sampling.top_p = sampling.top_p;
    options.execution.sampling.top_k = sampling.top_k;
    options.execution.sampling.min_p = sampling.min_p;
    options.execution.sampling.presence_penalty = this->options.presence_penalty;
    options.execution.sampling.seed = 42;
    options.execution.requested_output_tokens = maximum;
    if (input.reasoning_budget) { options.execution.thinking.budget = input.reasoning_budget; }
    options.output.raw = true; options.output.preserve_special_tokens = true;
    ninfer::GenerationObservationOptions observations;
    observations.token_ids = true; observations.hidden_features = true;
    observations.prompt_progress = true; observations.max_queued_tokens = 128;
    frankie_ninfer::voice_output sink(llama_model_get_vocab(model.get()), formatted, input.answer_limit, on_text, on_stage);
    auto handle = engine.submit(prepare(prompt, input.chat.enable_thinking), options, ninfer::OutputConsumerMode::Streaming, observations);
    const auto output = handle.wait(&sink, cancellation(cancelled));
    if (cancelled.load() || output.finish_reason == ninfer::FinishReason::Cancelled) { throw std::runtime_error("cancelled"); }
    const auto * vocab = llama_model_get_vocab(model.get());
    for (const auto token : sink.generated) {
        if (!llama_vocab_is_eog(vocab, token)) { prompt.tokens.push_back(token); }
    }
    for (const auto frontier : output.execution_frontiers) { record_frontier(frontier); }
    if (sink.needs_tail()) {
        const auto tail = engine.evaluate_prompt(prepare(prompt, input.chat.enable_thinking, state->prefix->tokens.size()), {}, cancellation(cancelled), true);
        if (!tail.features.token_ids.empty()) { sink.features(tail.features); }
        record_frontier(prompt.tokens.size());
    }
    sink.finish();
    if (sink.result.message.tool_calls.size() > 1) { throw std::runtime_error("parallel calls not enabled"); }
    for (const auto & call : sink.result.message.tool_calls) {
        if (std::none_of(input.chat.tools.begin(), input.chat.tools.end(), [&](const auto & tool) { return tool.name == call.name; })) {
            throw std::runtime_error("unadvertised tool");
        }
    }
    state->prefix = std::make_shared<const encoded_prompt>(std::move(prompt));
    std::cerr << "ninfer_voice generated_tokens=" << sink.generated.size() << " reasoning_budget=" << input.reasoning_budget
              << " cached_tokens=" << output.reused_prompt_tokens << " mtp_accepted=" << output.speculative.accepted_tokens << "\n";
    if (output.finish_reason == ninfer::FinishReason::OutputLimit || output.finish_reason == ninfer::FinishReason::ContextCapacity) {
        throw output_limit("max_output_tokens");
    }
    return std::move(sink.result);
}

void ninfer_brain_session::report_memory() const {
    brain_session::report_memory();
    const auto memory = engine.memory_summary();
    std::cerr << "memory ninfer brain_weights=" << memory.weights.used_bytes << " kv=" << memory.kv_payload_bytes
              << " workspace=" << memory.workspace.capacity_bytes << "\n";
}
