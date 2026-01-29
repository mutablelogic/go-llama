#include "grammar.h"
#include "model.h"
#include "error.h"
#include <llama.h>

///////////////////////////////////////////////////////////////////////////////
// Grammar Sampler

void* llama_go_grammar_sampler_new(
    void* model_handle,
    const char* grammar_str,
    const char* grammar_root) {
    
    if (!model_handle || !grammar_str || !grammar_root) {
        llama_go_set_error("invalid parameters");
        return nullptr;
    }

    auto model = reinterpret_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        llama_go_set_error("failed to get model vocabulary");
        return nullptr;
    }

    auto sampler = llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    if (!sampler) {
        llama_go_set_error("failed to initialize grammar sampler (invalid grammar?)");
        return nullptr;
    }

    return sampler;
}

///////////////////////////////////////////////////////////////////////////////
// Lazy Grammar Sampler

void* llama_go_grammar_sampler_new_lazy(
    void* model_handle,
    const char* grammar_str,
    const char* grammar_root,
    const char** trigger_patterns,
    size_t num_trigger_patterns,
    const int32_t* trigger_tokens,
    size_t num_trigger_tokens) {
    
    if (!model_handle || !grammar_str || !grammar_root) {
        llama_go_set_error("invalid parameters");
        return nullptr;
    }

    auto model = reinterpret_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        llama_go_set_error("failed to get model vocabulary");
        return nullptr;
    }

    // Convert llama_go_token_t to llama_token
    const llama_token* tokens = reinterpret_cast<const llama_token*>(trigger_tokens);

    auto sampler = llama_sampler_init_grammar_lazy_patterns(
        vocab,
        grammar_str,
        grammar_root,
        trigger_patterns,
        num_trigger_patterns,
        tokens,
        num_trigger_tokens);
    
    if (!sampler) {
        llama_go_set_error("failed to initialize lazy grammar sampler (invalid grammar or patterns?)");
        return nullptr;
    }

    return sampler;
}
