#include "options.h"
#include <stdexcept>
#include <iostream>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { throw std::runtime_error(#condition); } } while (false)

static ninfer::frankie::Options parse(std::vector<std::string> values) {
    std::vector<char*> args;
    for (auto& value : values) { args.push_back(value.data()); }
    return ninfer::frankie::parse_options(static_cast<int>(args.size()), args.data());
}

int main() {
    using namespace ninfer;
    const auto ordinary = parse({"ninfer-frankie", "brain.ninfer", "--frankie-package", "speech.gguf"});
    CHECK(ordinary.http.max_concurrency == 2 && ordinary.http.prefill_chunk == 128);
    CHECK(ordinary.http.default_max_tokens == 32768 && ordinary.http.enable_thinking);
    CHECK(ordinary.voice.thinking == "none" && ordinary.voice.max_output_audio_seconds == 120);
    CHECK(ordinary.voice.http_slots == 0 && ordinary.voice.context_tokens == ordinary.http.max_context);
    CHECK(ordinary.voice.speech_context_words == 100);
    CHECK(ordinary.http.hidden_layer == 16 && ordinary.http.enable_vision);
    CHECK(ordinary.http.kv_capacity.explicit_tokens == 2 * ordinary.http.max_context);
    const auto explicit_options = parse({"ninfer-frankie", "brain.ninfer", "--frankie-package", "speech.gguf",
        "--voice", "reference.wav", "--voice-text-file", "reference.txt", "--voice-thinking", "low",
        "--voice-context", "32768", "--max-context", "131072", "--kv-capacity", "196608",
        "--spec", "mtp", "--draft-tokens", "3", "--kv-dtype", "nvfp4", "--max-concurrency", "3",
        "--prefill-chunk", "256", "--speech-context-words", "0", "--api-key", "--voice"});
    CHECK(explicit_options.http.kv_capacity.explicit_tokens == 196608);
    CHECK(explicit_options.http.max_context == 131072 && explicit_options.voice.context_tokens == 32768);
    CHECK(explicit_options.http.max_concurrency == 3 && explicit_options.http.prefill_chunk == 256);
    CHECK(explicit_options.voice.mtp_tokens == 3 && explicit_options.voice.thinking == "low");
    CHECK(explicit_options.voice.voice == "reference.wav" && explicit_options.http.api_key == "--voice");
    CHECK(explicit_options.http.startup_argv.back() == "<redacted>");
    CHECK(parse({"ninfer-frankie", "--help"}).http.help_requested);
    const std::vector<std::vector<std::string>> invalid{
        {"--voice"}, {"--kv-capacity", "auto"}, {"--max-concurrency", "1"},
        {"--voice-context", "131072"}, {"--voice-thinking", "bogus"},
        {"--voice-max-tokens", "0"}, {"--speech-context-words", "1001"},
        {"--max-output-audio-seconds", "0"}, {"--max-utterance-seconds", "121"},
        {"--voice-text-file", "transcript.txt"}, {"--speech-threads", "4294967295"},
        {"--no-prefix-reuse"}, {"--unknown", "value"}, {"--device", "1"}
    };
    for (const auto& suffix : invalid) {
        std::vector<std::string> args{"ninfer-frankie", "brain.ninfer", "--frankie-package", "speech.gguf"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        bool rejected = false;
        try { parse(args); } catch (const std::exception&) { rejected = true; }
        CHECK(rejected);
    }
    std::cout << "PASS: CLI delegation, authentication redaction, shared context, voice controls and 14 rejected configurations\n";
}
