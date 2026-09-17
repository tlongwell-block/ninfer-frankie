#include "voice-output.h"
#include "component.h"
#include "common.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace {
void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
ninfer::CommittedTokenFeatures features(uint32_t begin, const std::vector<llama_token> & ids) {
    ninfer::CommittedTokenFeatures result;
    result.begin = begin; result.width = 5120; result.layer = 16; result.token_ids = ids;
    result.values.resize(ids.size() * 5120);
    for (size_t i = 0; i < ids.size(); ++i) { result.values[i * 5120] = float(ids[i]); }
    return result;
}
template<class F> void rejects(F work, const char * message) {
    try { work(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error(message);
}
}

int main(int argc, char ** argv) {
    try {
        const char * package = argc == 2 ? argv[1] : std::getenv("NINFER_FRANKIE_SPEECH");
        if (!package || !*package) { std::cout << "SKIP: set NINFER_FRANKIE_SPEECH or supply SPEECH_GGUF\n"; return 77; }
        component brain(package, "brain");
        auto params = llama_model_default_params(); params.vocab_only = true; params.n_gpu_layers = 0; ggml_backend_dev_t devices[] = {nullptr}; params.devices = devices;
        llama_model_ptr model(llama_model_init_from_user(brain.metadata(), component::set_tensor, &brain, params));
        require(bool(model), "vocabulary load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto templates = common_chat_templates_init(model.get(), "");
        common_chat_templates_inputs input;
        common_chat_msg user; user.role = "user"; user.content = "Greet me briefly."; input.messages.push_back(user);
        input.enable_thinking = false; input.add_generation_prompt = true; input.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        const auto format = common_chat_templates_apply(templates.get(), input);
        const auto ids = common_tokenize(vocab, "Hello there, friend.", false, true);
        require(ids.size() >= 3, "fixture needs an MTP-sized chunk");
        constexpr uint32_t prompt = 42;
        auto start = [&](frankie_ninfer::voice_output & sink) { ninfer::GenerationStart value; value.prompt.prompt_tokens = prompt; sink.start(value); };
        frankie_ninfer::voice_output sink(vocab, format, 128, {}, {}); start(sink);
        sink.tokens({prompt, ids});
        require(sink.result.raw.text.empty(), "unexecuted sampled words were published");
        std::vector<llama_token> head(ids.begin(), ids.end()-1), tail(ids.end()-1, ids.end());
        sink.features(features(prompt, head));
        require(sink.needs_tail(), "sampled MTP bonus lost its pending row");
        require(sink.result.raw.hidden.size() == head.size(), "accepted rows were not streamed");
        sink.features(features(prompt + head.size(), tail)); sink.finish();
        require(sink.result.message.content == "Hello there, friend.", "plain speech changed");
        require(sink.result.raw.hidden.size() == ids.size(), "final word has no feature row");
        for (size_t i = 0; i < ids.size(); ++i) { require(sink.result.raw.hidden[i][0] == float(ids[i]), "MTP row aligned to a different token"); }
        auto eog = llama_vocab_eos(vocab);
        require(llama_vocab_is_eog(vocab, eog), "fixture EOS is not an EOG");
        frankie_ninfer::voice_output stopped(vocab, format, 128, {}, {}); start(stopped);
        stopped.tokens({prompt, {eog}}); stopped.finish();
        require(stopped.result.raw.text.empty(), "EOG entered speech");
        frankie_ninfer::voice_output missing(vocab, format, 128, {}, {}); start(missing);
        missing.tokens({prompt, {ids.front()}});
        rejects([&] { missing.finish(); }, "missing final feature was silently accepted");
        frankie_ninfer::voice_output wrong(vocab, format, 128, {}, {}); start(wrong);
        wrong.tokens({prompt, {ids.front()}});
        rejects([&] { wrong.features(features(prompt, {ids.back()})); }, "wrong token's feature was accepted");
        frankie_ninfer::voice_output rejected(vocab, format, 128, {}, {}); start(rejected);
        rejected.features(features(prompt, {ids.front()}));
        rejects([&] { rejected.finish(); }, "uncommitted speculative row was accepted");
        frankie_ninfer::voice_output layer(vocab, format, 128, {}, {}); start(layer);
        auto bad = features(prompt, {ids.front()}); bad.layer = 17;
        rejects([&] { layer.features(bad); }, "wrong layer was accepted");
        input.enable_thinking = true;
        const auto reasoning_format = common_chat_templates_apply(templates.get(), input);
        const std::string thought = "Let me consider this.</think>\n\nHello there.";
        const auto thought_ids = common_tokenize(vocab, thought, false, true);
        frankie_ninfer::voice_output reasoning(vocab, reasoning_format, 128, {}, {}); start(reasoning);
        reasoning.tokens({prompt, thought_ids});
        const auto thought_prefix = common_tokenize(vocab, "Let me consider this.</think>\n\n", false, true);
        require(std::equal(thought_prefix.begin(), thought_prefix.end(), thought_ids.begin()), "reasoning fixture crosses token boundary");
        reasoning.features(features(prompt + thought_prefix.size(), {thought_ids.begin() + thought_prefix.size(), thought_ids.end()}));
        reasoning.finish();
        require(reasoning.result.message.content == "Hello there.", "reasoning entered speech");
        require(!reasoning.result.message.reasoning_content.empty(), "reasoning was lost");
        require(reasoning.result.raw.hidden.size() < thought_ids.size(), "reasoning retained mouth feature rows");
        input.enable_thinking = false;
        input.tools.push_back({"lookup", "Look up a value.", R"({"type":"object","properties":{"key":{"type":"string"}},"required":["key"]})"});
        const auto tool_format = common_chat_templates_apply(templates.get(), input);
        const std::string tool_text = "<tool_call>\n<function=lookup>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>";
        const auto tool_ids = common_tokenize(vocab, tool_text, false, true);
        frankie_ninfer::voice_output tool(vocab, tool_format, 128, {}, {}); start(tool);
        tool.tokens({prompt, tool_ids}); tool.finish();
        require(tool.result.message.content.empty(), "tool markup entered speech");
        require(tool.result.message.tool_calls.size() == 1 && tool.result.message.tool_calls.front().name == "lookup", "tool call was not preserved");
        require(tool.result.message.tool_calls.front().arguments.find("value") != std::string::npos, "tool arguments were lost");
        std::cout << "PASS: MTP token/row alignment, delayed bonus, EOG, missing/rejected/wrong-layer rows, reasoning/control separation, tool calls\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << "\n"; return 1; }
}
