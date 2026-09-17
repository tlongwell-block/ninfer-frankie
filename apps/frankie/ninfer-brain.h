#pragma once
#include "brain-session.h"
#include "ninfer/engine.h"

// NInfer executes the one resident brain; the base owns only the original native
// ear/bridge and vocabulary. HTTP and voice submit to the same Engine worker.
class ninfer_brain_session final : public brain_session {
    struct logical_state;
    struct encoded_prompt;
    ninfer::Engine & engine;
    std::shared_ptr<logical_state> state;
    encoded_prompt encode_prompt(const request &, const std::string & prompt,
                                 const std::map<std::string, size_t> & pending_audio = {}) const;
    ninfer::PreparedPrompt prepare(const encoded_prompt &, bool thinking, std::optional<uint32_t> checkpoint = {}) const;
    void discard_session();
    void record_frontier(uint32_t frontier);
    void cache_prefix(encoded_prompt prompt);
  public:
    ninfer_brain_session(ninfer::Engine &, const std::string & package, const frankie_options &);
    ~ninfer_brain_session() override;
    size_t prompt_tokens(const request &, const std::map<std::string, size_t> &) const override;
    void reset(bool preserve_checkpoint = false) override;
    void report_memory() const override;
    saved_state suspend_state() override;
    void restore_state(saved_state) override;
    void precommit(const request &, const std::string & marker, const std::vector<float> &, size_t count) override;
    listener_reaction probe_listener(const request &) override;
    void finish_audio(const request &, const std::string & marker, std::vector<float> &) override;
    void warm_prefix(common_chat_templates_inputs) override;
    response generate(const request &, const stream_callback & = {}, const stage_callback & = {}) override;
};
