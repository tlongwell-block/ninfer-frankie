#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "targets/registry.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedImage::Impl {
public:
    targets::qwen3_6::PreparedPromptData data;
};

std::span<const TokenId> PreparedImage::token_ids() const noexcept {
    return impl_ ? std::span<const TokenId>(impl_->data.token_ids) : std::span<const TokenId>{};
}

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using Core27      = runtime::EngineCore<targets::Qwen3_6_27BInstance>;
    using Core35      = runtime::EngineCore<targets::Qwen3_6_35BA3BInstance>;
    using ScoreCore27 = runtime::CausalScoreCore<targets::Qwen3_6_27BInstance>;
    using ScoreCore35 = runtime::CausalScoreCore<targets::Qwen3_6_35BA3BInstance>;
    using Core = std::variant<std::monostate, std::unique_ptr<Core27>, std::unique_ptr<Core35>,
                              std::unique_ptr<ScoreCore27>, std::unique_ptr<ScoreCore35>>;

    explicit Impl(EngineOptions engine_options)
        : options(normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed  = targets::construct_target(options, device);
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        core = std::visit(
            [&](auto& target_ptr) -> Core {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore27>(*target_ptr, device);
                    }
                    return std::make_unique<Core27>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                } else {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore35>(*target_ptr, device);
                    }
                    return std::make_unique<Core35>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                }
            },
            active);
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            if (token_ids.size() > target_ptr->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   context_capacity_error(token_ids.size(), target_ptr->capacity));
            }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_embeddings(std::vector<TokenId> token_ids,
                                          std::vector<InputEmbeddingSpan> embeddings,
                                          RawPromptOptions options, std::vector<RawImageSpan> images) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto sampling_mode = options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit([&](const auto& target) -> PreparedPrompt {
        auto prepared = target->loaded->frontend.prepare_embeddings(
            std::move(token_ids), std::move(embeddings), std::move(options));
        auto& data = targets::qwen3_6::PreparedPromptAccess::mutable_view(prepared);
        std::uint32_t previous_end = 0;
        std::int32_t rope_delta = 0;
        std::size_t patch_offset = 0;
        const auto tokens = data.token_ids.size();
        for (const auto& span : images) {
            if (!span.image.impl_) { throw std::invalid_argument("empty prepared image"); }
            const auto& source = span.image.impl_->data;
            const auto count = source.token_ids.size();
            if (span.begin < previous_end || span.begin > tokens || count > tokens - span.begin ||
                !std::equal(source.token_ids.begin(), source.token_ids.end(), data.token_ids.begin() + span.begin)) {
                throw std::invalid_argument("image spans must match ordered disjoint image token runs");
            }
            for (const auto& row : data.embedding_identity) {
                if (row.position >= span.begin && row.position < span.begin + count) {
                    throw std::invalid_argument("image and external embedding spans overlap");
                }
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                for (std::size_t pos = previous_end; pos < span.begin; ++pos) {
                    data.positions[axis * tokens + pos] = static_cast<std::int32_t>(pos) + rope_delta;
                }
                for (std::size_t pos = 0; pos < count; ++pos) {
                    data.positions[axis * tokens + span.begin + pos] =
                        static_cast<std::int32_t>(span.begin) + rope_delta + source.positions[axis * count + pos];
                }
            }
            std::copy(source.token_types.begin(), source.token_types.end(), data.token_types.begin() + span.begin);
            for (auto item : source.vision_items) {
                item.patch_begin += patch_offset;
                for (auto& range : item.token_spans) { range.begin += span.begin; }
                patch_offset += item.patch_count;
                data.vision_items.push_back(std::move(item));
            }
            data.media_payloads.insert(data.media_payloads.end(), source.media_payloads.begin(), source.media_payloads.end());
            data.prepare.media_items += source.prepare.media_items;
            data.prepare.media_bytes += source.prepare.media_bytes;
            data.prepare.raw_patches += source.prepare.raw_patches;
            data.prepare.vision_tokens += source.prepare.vision_tokens;
            data.prepare.patch_bytes += source.prepare.patch_bytes;
            target->loaded->frontend.validate_media_budget(prepared.preparation_stats());
            previous_end = static_cast<std::uint32_t>(span.begin + count);
            rope_delta += source.rope_delta;
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (std::size_t pos = previous_end; pos < tokens; ++pos) {
                data.positions[axis * tokens + pos] = static_cast<std::int32_t>(pos) + rope_delta;
            }
        }
        data.rope_delta = rope_delta;
        const auto summary = prepared.summary();
        const auto stats = prepared.preparation_stats();
        return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(summary, stats, sampling_mode,
                                                                    std::move(prepared)));
    }, impl_->active);
}

PreparedImage Engine::prepare_image(OwnedMedia media, const PreparationControl& control) const {
    if (media.kind != MediaKind::Image) { throw std::invalid_argument("raw image input must be an image"); }
    PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(ChatMessage{.role=ChatRole::User,
        .parts={MessagePart{.kind=MessagePartKind::Media, .media=std::move(media)}}});
    auto prepared = prepare(std::move(input), control);
    auto data = targets::qwen3_6::PreparedPromptAccess::take(std::move(prepared.impl_->value));
    if (data.vision_items.size() != 1 || data.vision_items.front().token_spans.size() != 1) {
        throw std::logic_error("single raw image produced an unexpected vision layout");
    }
    const auto span = data.vision_items.front().token_spans.front();
    if (span.begin == 0 || span.begin + span.count >= data.token_ids.size()) {
        throw std::logic_error("raw image is missing its delimiter tokens");
    }
    const auto start = span.begin - 1;
    const auto count = span.count + 2;
    const auto vision_start = tokenize_text("<|vision_start|>");
    const auto vision_end = tokenize_text("<|vision_end|>");
    if (vision_start.size() != 1 || vision_end.size() != 1 ||
        data.token_ids[start] != vision_start.front() ||
        data.token_ids[start + count - 1] != vision_end.front()) {
        throw std::logic_error("raw image delimiters do not match the artifact tokenizer");
    }
    const auto original_tokens = data.token_ids.size();
    const auto position_base = data.positions[start];
    std::vector<std::int32_t> positions(3 * count);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t n = 0; n < count; ++n) {
            positions[axis * count + n] = data.positions[axis * original_tokens + start + n] - position_base;
        }
    }
    data.token_ids = std::vector<TokenId>(data.token_ids.begin() + start, data.token_ids.begin() + start + count);
    data.token_types = std::vector<std::uint8_t>(data.token_types.begin() + start, data.token_types.begin() + start + count);
    data.positions = std::move(positions);
    const auto last_position = std::max({data.positions[count-1], data.positions[2*count-1], data.positions[3*count-1]});
    data.rope_delta = last_position + 1 - static_cast<std::int32_t>(count);
    data.vision_items.front().token_spans.front().begin -= start;
    data.identity = {};
    data.context_cache = {};
    PreparedImage result;
    auto storage = std::make_shared<PreparedImage::Impl>();
    storage->data = std::move(data);
    result.impl_ = std::move(storage);
    return result;
}

PromptEvaluation Engine::evaluate_prompt(PreparedPrompt prompt, std::vector<TokenId> logit_ids,
                                         const CancellationView& cancellation, bool capture_tail_features) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (!prompt.impl_) { throw std::invalid_argument("prepared prompt is empty"); }
    auto probe = std::make_shared<targets::qwen3_6::PromptLogitProbe>();
    probe->capture_tail = capture_tail_features;
    if (capture_tail_features && !impl_->options.hidden_layer) {
        throw std::invalid_argument("tail features require EngineOptions.hidden_layer");
    }
    probe->token_ids = std::move(logit_ids);
    if (!probe->token_ids.empty()) { (void)prepare_tokens(probe->token_ids, false); }
    probe->logits.resize(probe->token_ids.size());
    {
        auto& data = targets::qwen3_6::PreparedPromptAccess::mutable_view(prompt.impl_->value);
        const auto frontier = static_cast<std::uint32_t>(data.token_ids.size());
        if (frontier == 0) { throw std::invalid_argument("cannot evaluate an empty prompt"); }
        data.logit_probe = probe;
        // Retain the prompt state, independently of the sampled, unexecuted bonus.
        data.identity.rewrite_checkpoint = targets::qwen3_6::RewriteCheckpointSpec{
            targets::qwen3_6::RewriteCheckpointKind::ResponseReplay, frontier};
        auto& boundaries = data.identity.rewrite_execution_frontiers;
        if (boundaries.empty() || boundaries.back() != frontier) { boundaries.push_back(frontier); }
    }
    RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.output.raw = true;
    auto generation = generate(std::move(prompt), options, nullptr, cancellation);
    return PromptEvaluation{std::move(generation), std::move(probe->logits), std::move(probe->features)};
}

bool Engine::discard_session(std::string_view session_key) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit([&](auto& core) -> bool {
        using T = std::remove_reference_t<decltype(core)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<Impl::Core27>> ||
                      std::is_same_v<T, std::unique_ptr<Impl::Core35>>) {
            return core->discard_session(session_key);
        } else { throw std::logic_error("session discard requires a Generation Engine"); }
    }, impl_->core);
}

void Engine::with_device_idle(const std::function<void()>& work) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::visit([&](auto& core) {
        using T = std::remove_reference_t<decltype(core)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<Impl::Core27>> ||
                      std::is_same_v<T, std::unique_ptr<Impl::Core35>>) {
            core->with_device_idle(work);
        } else { throw std::logic_error("external work requires a Generation Engine"); }
    }, impl_->core);
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.tokenize_text(text);
        },
        impl_->active);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (observation.max_queued_tokens != 0) { observation.token_ids = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress || observation.token_ids || observation.hidden_features)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    targets::qwen3_6::PreparedPromptAccess::mutable_view(prompt.impl_->value).capture_hidden =
        observation.hidden_features;
    if (observation.hidden_features && !impl_->options.hidden_layer) {
        throw std::invalid_argument("hidden_features requires EngineOptions.hidden_layer");
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
