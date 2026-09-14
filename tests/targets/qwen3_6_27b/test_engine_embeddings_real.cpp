// Opt-in integration test against an actual Frankie artifact and CUDA target. Synthetic input
// rows exercise cache/state semantics; this is not an acoustic-quality or layer-math oracle.
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

struct Sink final : ninfer::OutputSink {
    std::map<std::uint32_t, ninfer::TokenId> tokens_by_position;
    std::map<std::uint32_t, ninfer::TokenId> features_by_position;
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta) override {}
    void tokens(ninfer::CommittedTokens event) override {
        for (std::size_t n = 0; n < event.token_ids.size(); ++n) {
            require(tokens_by_position.emplace(event.begin + n, event.token_ids[n]).second,
                    "duplicate committed output token");
        }
    }
    void features(ninfer::CommittedTokenFeatures event) override {
        require(event.width == 5120 && event.layer == 16, "incorrect feature geometry");
        require(event.values.size() == event.token_ids.size() * event.width,
                "incorrect feature row count");
        for (float value : event.values) { require(std::isfinite(value), "non-finite target feature"); }
        for (std::size_t n = 0; n < event.token_ids.size(); ++n) {
            require(features_by_position.emplace(event.begin + n, event.token_ids[n]).second,
                    "duplicate executed target row");
        }
    }
};

void run(const char* artifact, unsigned drafts) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 512;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.max_concurrency = 2;
    options.prefill_chunk = 128;
    options.hidden_layer = 16;
    options.context_cache.device_state_slots = 4;
    // Short device-resident cache checks do not need the default 8 GiB host KV reserve.
    options.context_cache.host_state_slots = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.speculative.backend = drafts ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = drafts;
    ninfer::Engine engine(options);
    auto ids = engine.tokenize_text("Count slowly from one to twenty, separated by spaces.\n");
    require(ids.size() > 8, "test prompt too short");
    ninfer::InputEmbeddingSpan audio{2, 5120, std::vector<float>(2 * 5120)};
    for (std::size_t n = 0; n < audio.values.size(); ++n) {
        audio.values[n] = static_cast<float>(static_cast<int>(n % 29) - 14) / 128.0F;
    }
    const auto prepared = [&](const std::vector<ninfer::TokenId>& tokens,
                              const ninfer::InputEmbeddingSpan& rows,
                              std::vector<std::uint32_t> frontiers = {}) {
        ninfer::RawPromptOptions raw;
        raw.session_key = "embedded-runtime-test";
        raw.rewrite_execution_frontiers = std::move(frontiers);
        return engine.prepare_embeddings(tokens, {rows}, std::move(raw));
    };
    const std::vector<ninfer::TokenId> picks{0, 1, 2, 3, 4};
    const auto first = engine.evaluate_prompt(prepared(ids, audio), picks, {}, true);
    require(first.logits.size() == picks.size(), "listener logits missing");
    require(first.features.token_ids == std::vector<ninfer::TokenId>{ids.back()},
            "prefill tail features do not name the executed input token");
    const auto repeat = engine.evaluate_prompt(prepared(ids, audio), picks);
    require(repeat.generation.reused_prompt_tokens > 0, "prefill checkpoint was not retained");
    require(first.logits == repeat.logits, "exact cached target tail changed listener logits");
    auto changed = audio;
    changed.values[0] += 1.0F;
    const auto revised = engine.evaluate_prompt(prepared(ids, changed), picks);
    require(revised.generation.reused_prompt_tokens <= audio.begin,
            "same placeholder tokens reused different audio content");
    require(engine.discard_session("embedded-runtime-test"), "idle session discard failed");
    const auto cold = engine.evaluate_prompt(prepared(ids, changed), picks);
    require(cold.generation.reused_prompt_tokens == 0, "discard retained private audio state");
    require(revised.logits == cold.logits, "changed-audio replay differs from cold evaluation");

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 16;
    request.execution.sampling.temperature = 0.0F;
    request.output.raw = true;
    ninfer::GenerationObservationOptions observation;
    observation.token_ids = true;
    observation.hidden_features = true;
    observation.max_queued_tokens = 4;
    const std::vector<std::uint32_t> frontiers{static_cast<std::uint32_t>(ids.size())};
    auto generation = engine.submit(prepared(ids, changed, frontiers), request,
                                    ninfer::OutputConsumerMode::Streaming, observation);
    // A separate HTTP-shaped request uses the same loaded brain while the voice lane is bounded.
    auto ordinary = engine.submit(engine.prepare_tokens(engine.tokenize_text("One plus one is"), false),
                                  request);
    require(!ordinary.wait().generated_token_ids.empty(), "concurrent text request failed");
    Sink sink;
    const auto result = generation.wait(&sink);
    require(!result.generated_token_ids.empty(), "no voice output tokens");
    for (std::size_t n = 0; n < result.generated_token_ids.size(); ++n) {
        const auto position = static_cast<std::uint32_t>(ids.size() + n);
        require(sink.tokens_by_position.at(position) == result.generated_token_ids[n],
                "stream/result token mismatch");
    }
    // The final sampled bonus has not executed. Replaying the extended prompt executes that
    // exact token; rejected MTP draft rows must never appear as committed feature events.
    auto extended = ids;
    extended.insert(extended.end(), result.generated_token_ids.begin(), result.generated_token_ids.end());
    auto tail = engine.evaluate_prompt(prepared(extended, changed, frontiers), {}, {}, true);
    if (!tail.features.token_ids.empty()) { sink.features(std::move(tail.features)); }
    for (const auto& [position, token] : sink.features_by_position) {
        require(position < extended.size() && extended[position] == token,
                "feature names rejected draft or wrong absolute input position");
    }
    for (const auto& [position, token] : sink.tokens_by_position) {
        require(sink.features_by_position.contains(position), "generated token lost its executed feature");
        require(sink.features_by_position.at(position) == token, "generated feature alignment mismatch");
    }
    require(engine.discard_session("embedded-runtime-test"), "terminal voice session not releasable");
    std::cout << "embedded runtime MTP " << drafts << ": passed\n";
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FRANKIE_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set NINFER_FRANKIE_ARTIFACT to an actual Frankie .ninfer artifact\n";
        return 77;
    }
    try {
        run(artifact, 0);
        run(artifact, 3);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
