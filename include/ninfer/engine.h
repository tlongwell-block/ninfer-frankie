#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class PreparedImage {
public:
    PreparedImage() noexcept = default;
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return impl_ != nullptr; }
private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
    friend class Engine;
};

struct RawImageSpan {
    std::uint32_t begin = 0;
    PreparedImage image;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] PreparedPrompt prepare_embeddings(std::vector<TokenId> token_ids,
                                                     std::vector<InputEmbeddingSpan> embeddings,
                                                     RawPromptOptions options = {},
                                                     std::vector<RawImageSpan> images = {}) const;
    [[nodiscard]] PreparedImage prepare_image(OwnedMedia media,
                                              const PreparationControl& control = {}) const;

    // Evaluate and retain the exact prompt frontier (one sampled bonus is discarded). Unlike
    // generate(...requested_output_tokens=0), this actually executes a prefill.
    // Tail features describe newly executed rows only; an exact cache hit returns no features.
    [[nodiscard]] PromptEvaluation evaluate_prompt(PreparedPrompt prompt,
                                                   std::vector<TokenId> logit_ids = {},
                                                   const CancellationView& cancellation = {},
                                                   bool capture_tail_features = false);

    // Run linked speech CUDA work between brain execution units. The callback must complete its
    // own GPU work before returning, and must not recursively invoke this Engine.
    void with_device_idle(const std::function<void()>& work);
    // Cancel/join this session's requests before discarding. False means a checkpoint is still
    // borrowed or a resource transaction is in flight; retry after yielding to the worker.
    [[nodiscard]] bool discard_session(std::string_view session_key);

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
