#pragma once
#include "brain-session.h"
#include "ninfer/types.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace frankie_ninfer {
inline constexpr size_t hidden_width = 5120;
inline constexpr uint32_t hidden_layer = 16;
// Audible output waits for each sampled token's committed feature row;
// reasoning/control tokens may advance the parser without changing speech.
// In particular, an MTP bonus must never borrow its predecessor's row.
class voice_output final : public ninfer::OutputSink {
    const llama_vocab * vocab;
    common_chat_parser_params parser;
    brain_session::stream_callback on_text;
    brain_session::stage_callback on_stage;
    uint32_t limit;
    size_t answer_tokens = 0;
    uint32_t next = 0;
    uint32_t published = 0;
    std::map<uint32_t, ninfer::TokenId> pending;
    struct row { ninfer::TokenId token; std::array<float, hidden_width> values; };
    std::map<uint32_t, row> rows;
    bool started = false;
    bool first = true;
  public:
    brain_session::response result;
    std::vector<ninfer::TokenId> generated;
    bool limited = false;
    voice_output(const llama_vocab * vocabulary, const common_chat_params & format,
                 uint32_t answer_limit, brain_session::stream_callback text,
                 brain_session::stage_callback stage) :
        vocab(vocabulary), parser(format), on_text(std::move(text)), on_stage(std::move(stage)), limit(answer_limit) {
        parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        if (!format.parser.empty()) { parser.parser.load(format.parser); }
    }
    void start(ninfer::GenerationStart value) override {
        next = published = value.prompt.prompt_tokens;
        started = true;
        if (on_stage) { on_stage(value.reused_prompt_tokens ? "cache_append" : "cache_replay"); }
    }
    void progress(ninfer::PromptProgress value) override {
        if (value.processed_prompt_tokens >= value.total_prompt_tokens && on_stage) {
            on_stage("brain_prefill_done");
        }
    }
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta) override {} // Preserve the existing native tool/speech parser.
    void tokens(ninfer::CommittedTokens value) override {
        if (!started) { throw std::runtime_error("voice tokens arrived before prompt admission"); }
        if (value.begin != published) { throw std::runtime_error("noncontiguous committed voice tokens"); }
        published += value.token_ids.size();
        for (size_t i = 0; i < value.token_ids.size(); ++i) {
            const auto position = value.begin + uint32_t(i);
            if (position < next || !pending.emplace(position, value.token_ids[i]).second) {
                throw std::runtime_error("duplicate or retracted voice token");
            }
            generated.push_back(value.token_ids[i]);
        }
        drain();
    }
    void features(ninfer::CommittedTokenFeatures value) override {
        if (value.width != hidden_width || value.layer != hidden_layer ||
            value.values.size() != value.token_ids.size() * hidden_width) {
            throw std::runtime_error("voice feature layer or shape mismatch");
        }
        for (size_t i = 0; i < value.token_ids.size(); ++i) {
            const auto position = value.begin + uint32_t(i);
            if (position < next) { continue; } // Prefill tail or an already-consumed finalization row.
            row entry;
            entry.token = value.token_ids[i];
            std::copy_n(value.values.data() + i * hidden_width, hidden_width, entry.values.begin());
            if (!std::all_of(entry.values.begin(), entry.values.end(), [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("nonfinite voice features");
            }
            if (!rows.emplace(position, std::move(entry)).second) {
                throw std::runtime_error("duplicate voice feature row");
            }
        }
        drain();
    }
    void drain() {
        bool changed = false;
        for (;;) {
            auto token = pending.find(next);
            if (token == pending.end()) { break; }
            if (llama_vocab_is_eog(vocab, token->second)) {
                pending.erase(token); rows.erase(next++); continue;
            }
            auto feature = rows.find(next);
            const auto text = result.raw.text + common_token_to_piece(vocab, token->second, true);
            auto message = common_chat_parse(text, true, parser);
            // Injected reasoning-close tokens have no speech feature row. They can
            // advance the parser only while leaving the audible content unchanged.
            if (feature == rows.end() && message.content != result.message.content) { break; }
            if (feature != rows.end() && feature->second.token != token->second) { throw std::runtime_error("voice feature token mismatch"); }
            if (first) { first = false; if (on_stage) { on_stage("brain_first_token"); } }
            result.raw.text = text;
            result.message = std::move(message);
            result.message.role = "assistant";
            if (!result.message.content.empty() || !result.message.tool_calls.empty()) {
                if (++answer_tokens > limit && limit) { limited = true; throw brain_session::output_limit("max_output_tokens"); }
                if (feature != rows.end()) {
                    result.raw.ends.push_back(result.raw.text.size());
                    result.raw.hidden.push_back(feature->second.values);
                }
            }
            if (!result.message.content.empty() && (result.content_offset == std::string::npos ||
                    result.raw.text.compare(result.content_offset, result.message.content.size(), result.message.content) != 0)) {
                result.content_offset = result.raw.text.rfind(result.message.content);
            }
            pending.erase(token); rows.erase(next); ++next; changed = true;
        }
        if (changed && on_text) { on_text(result, false); }
    }
    bool needs_tail() const {
        for (const auto & [position, token] : pending) { if (!llama_vocab_is_eog(vocab, token)) { return true; } }
        return false;
    }
    void finish() {
        drain();
        if (needs_tail()) { throw std::runtime_error("NInfer did not execute the final spoken token"); }
        if (!rows.empty()) { throw std::runtime_error("NInfer supplied features for uncommitted voice tokens"); }
        result.message = common_chat_parse(result.raw.text, false, parser);
        result.message.role = "assistant";
        if (on_text) { on_text(result, true); }
    }
};
} // namespace frankie_ninfer
