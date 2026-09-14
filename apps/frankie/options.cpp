#include "options.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::frankie {

Options parse_options(int argc, char** argv) {
    Options options;
    options.voice.max_output_audio_seconds = 120;
    bool voice_context_explicit = false;
    std::vector<char*> args;
    std::vector<std::string> defaults{
        "--prefill-chunk", "128", "--max-concurrency", "2",
        "--default-max-tokens", "32768", "--kv-capacity", "16384"};
    if (argc > 0) { args.push_back(argv[0]); }
    if (argc > 1) {
        args.push_back(argv[1]);
        for (auto& value : defaults) { args.push_back(value.data()); }
        args.insert(args.end(), argv + 2, argv + argc);
    }
    options.http = serve::parse_serve_options(static_cast<int>(args.size()), args.data(),
        [&](std::string_view flag, int& index, int count, char** values) {
            const auto known = flag == "--frankie-package" || flag == "--voice" ||
                flag == "--voice-text-file" || flag == "--voice-thinking" ||
                flag == "--voice-context" || flag == "--voice-max-tokens" ||
                flag == "--speech-context-words" || flag == "--speech-threads" ||
                flag == "--speech-device" || flag == "--text-encoder-device" ||
                flag == "--max-output-audio-seconds" || flag == "--max-utterance-seconds";
            if (!known) { return false; }
            if (++index >= count) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            const std::string value(values[index]);
            if (flag == "--frankie-package") { options.package = value; }
            else if (flag == "--voice") { options.voice.voice = value; }
            else if (flag == "--voice-text-file") { options.voice.voice_text = value; }
            else if (flag == "--voice-thinking") { options.voice.thinking = value; }
            else if (flag == "--voice-context") {
                options.voice.context_tokens = frankie_unsigned(value);
                voice_context_explicit = true;
            } else if (flag == "--voice-max-tokens") {
                options.voice.max_output_tokens = frankie_unsigned(value);
            } else if (flag == "--speech-context-words") {
                options.voice.speech_context_words = frankie_unsigned(value);
            } else if (flag == "--speech-threads") {
                const auto threads = frankie_unsigned(value);
                if (threads < 1 || threads > 256) { throw std::invalid_argument("--speech-threads must be 1..256"); }
                options.voice.threads = static_cast<int>(threads);
            } else if (flag == "--speech-device" || flag == "--text-encoder-device") {
                if (value != "cpu" && value != "gpu") { throw std::invalid_argument(std::string(flag) + " must be cpu or gpu"); }
                (flag == "--speech-device" ? options.voice.use_gpu : options.voice.text_encoder_gpu) = value == "gpu";
            } else if (flag == "--max-output-audio-seconds") {
                options.voice.max_output_audio_seconds = frankie_unsigned(value);
            } else if (flag == "--max-utterance-seconds") {
                options.voice.max_utterance_seconds = frankie_unsigned(value);
            }
            return true;
        });
    if (options.http.help_requested) { return options; }
    if (options.http.device != 0) {
        throw std::invalid_argument("Frankie requires --device 0 so the brain and speech use the same GPU");
    }
    if (options.package.empty()) { throw std::invalid_argument("--frankie-package is required"); }
    if (options.http.kv_capacity.mode == KvCapacityMode::Automatic) {
        throw std::invalid_argument("use an explicit --kv-capacity to leave GPU memory for speech components");
    }
    if (options.http.speculative.backend != SpeculativeBackend::None &&
        options.http.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("Frankie supports --spec mtp; its artifact has no DFlash weights");
    }
    if (options.http.max_concurrency < 2) {
        throw std::invalid_argument("Frankie needs --max-concurrency of at least 2 for voice and HTTP");
    }
    if (!options.http.allow_prefix_reuse) { throw std::invalid_argument("Frankie requires prefix reuse for audio continuation"); }
    if (!voice_context_explicit) { options.voice.context_tokens = options.http.max_context; }
    if (options.voice.context_tokens > options.http.max_context) {
        throw std::invalid_argument("--voice-context cannot exceed --max-context");
    }
    options.http.hidden_layer = 16;
    options.http.enable_vision = true;
    options.voice.resolve_context();
    frankie_thinking_budget(options.voice.thinking);
    options.voice.mtp_tokens = options.http.speculative.draft_tokens;
    if (options.voice.max_utterance_seconds < 2 || options.voice.max_utterance_seconds > 120 ||
        options.voice.max_output_audio_seconds < 1 || options.voice.max_output_audio_seconds > 3600 ||
        options.voice.speech_context_words > 1000 || !options.voice.max_output_tokens ||
        (!options.voice.voice_text.empty() && options.voice.voice.empty())) {
        throw std::invalid_argument("invalid voice bounds or --voice-text-file without --voice");
    }
    return options;
}

std::string usage(const char* program) {
    return serve::serve_usage_text(program) +
        "\nFrankie Breeze options:\n"
        "  --frankie-package PATH          Breeze v3 GGUF speech components (required)\n"
        "  --voice WAV                     Override the packaged reference voice\n"
        "  --voice-text-file PATH          Optional transcript for the voice reference\n"
        "  --voice-thinking LEVEL          none|minimal|low|medium|high|xhigh|max (none)\n"
        "  --voice-context N               Voice limit within --max-context\n"
        "  --voice-max-tokens N            Maximum generated answer tokens (32768)\n"
        "  --speech-context-words N        Breeze prosody cache limit (100; 0 disables)\n"
        "  --speech-threads N              Speech CPU worker count (4)\n"
        "  --speech-device cpu|gpu         Speech execution device (gpu)\n"
        "  --text-encoder-device cpu|gpu   Breeze text encoder device (gpu)\n"
        "  --max-output-audio-seconds N    Per-reply audio cap (120)\n"
        "  --max-utterance-seconds N       Input utterance cap (90)\n"
        "Frankie uses CUDA device 0 for the brain and GPU speech components.\n"
        "One Engine and listener serve realtime and HTTP. --api-key applies to both.\n"
        "Frankie defaults: --prefill-chunk 128 --max-concurrency 2 --default-max-tokens 32768.\n"
        "--max-context is per request; --kv-capacity is the shared pool (16384 by default).\n"
        "Choose a larger explicit pool for simultaneous long requests; automatic allocation is disabled.\n";
}

} // namespace ninfer::frankie
