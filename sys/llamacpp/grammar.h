#ifndef LLAMA_GO_GRAMMAR_H
#define LLAMA_GO_GRAMMAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

///////////////////////////////////////////////////////////////////////////////
// Grammar Sampler - Creates sampler from GBNF grammar string

// Creates a grammar sampler from a GBNF grammar string
// @param model - model handle to get vocab from
// @param grammar_str - GBNF grammar rules as string
// @param grammar_root - name of the start symbol (typically "root")
// @return sampler handle or NULL on error
void* llama_go_grammar_sampler_new(
    void* model,
    const char* grammar_str,
    const char* grammar_root);

///////////////////////////////////////////////////////////////////////////////
// Lazy Grammar Sampler - Grammar triggered by patterns/tokens

// Creates a lazy grammar sampler triggered by patterns or tokens
// @param model - model handle to get vocab from
// @param grammar_str - GBNF grammar rules as string
// @param grammar_root - name of the start symbol
// @param trigger_patterns - array of regex patterns (can be NULL)
// @param num_trigger_patterns - number of patterns
// @param trigger_tokens - array of token IDs (can be NULL)
// @param num_trigger_tokens - number of tokens
// @return sampler handle or NULL on error
void* llama_go_grammar_sampler_new_lazy(
    void* model,
    const char* grammar_str,
    const char* grammar_root,
    const char** trigger_patterns,
    size_t num_trigger_patterns,
    const int32_t* trigger_tokens,
    size_t num_trigger_tokens);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_GRAMMAR_H
