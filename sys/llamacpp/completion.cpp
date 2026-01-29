#include "completion.h"
#include "error.h"
#include "tokenizer.h"
#include "sampler.h"
#include "batch.h"

#include <llama.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

// Forward declaration of the Go callback
extern "C" bool goTokenCallback(void* handle, const char* token);

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMETERS

extern "C" llama_go_completion_params llama_go_completion_default_params() {
    llama_go_completion_params params{};
    
    // Default sampler params
    params.seed = 0;
    params.temperature = 0.8f;
    params.top_k = 40;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.repeat_penalty = 1.0f;
    params.repeat_last_n = 64;
    params.frequency_penalty = 0.0f;
    params.presence_penalty = 0.0f;
    
    params.max_tokens = 512;
    params.stop_words_count = 0;
    params.stop_words = nullptr;
    params.enable_prefix_caching = false;
    params.callback_handle = nullptr;
    
    return params;
}

///////////////////////////////////////////////////////////////////////////////
// GENERATION

extern "C" char* llama_go_completion_generate(
    void* ctx_handle,
    void* model_handle,
    const char* prompt,
    const llama_go_completion_params* gen_params
) {
    if (!ctx_handle || !model_handle || !prompt || !gen_params) {
        llama_go_set_error("Invalid arguments");
        return nullptr;
    }

    try {
        llama_context* ctx = static_cast<llama_context*>(ctx_handle);
        
        // Tokenize prompt using wrapper function
        std::vector<int32_t> prompt_tokens;
        prompt_tokens.resize(strlen(prompt) + 16); // Initial estimate
        
        int32_t n_prompt = llama_go_tokenize(
            model_handle,
            prompt,
            -1,  // null-terminated
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,   // add_special
            false   // parse_special
        );
        
        if (n_prompt < 0) {
            // Need larger buffer
            prompt_tokens.resize(-n_prompt);
            n_prompt = llama_go_tokenize(
                model_handle,
                prompt,
                -1,
                prompt_tokens.data(),
                prompt_tokens.size(),
                true,
                false
            );
        }
        
        if (n_prompt <= 0) {
            llama_go_set_error("Failed to tokenize prompt");
            return nullptr;
        }
        
        prompt_tokens.resize(n_prompt);
        
        // Check context size
        const int32_t n_ctx = llama_n_ctx(ctx);
        if (n_prompt + gen_params->max_tokens > n_ctx) {
            llama_go_set_error("Prompt + max_tokens exceeds context size");
            return nullptr;
        }
        
        // Create sampler with params
        llama_go_sampler_params sampler_params{};
        sampler_params.seed = gen_params->seed;
        sampler_params.temperature = gen_params->temperature;
        sampler_params.top_k = gen_params->top_k;
        sampler_params.top_p = gen_params->top_p;
        sampler_params.min_p = gen_params->min_p;
        sampler_params.repeat_penalty = gen_params->repeat_penalty;
        sampler_params.repeat_last_n = gen_params->repeat_last_n;
        sampler_params.frequency_penalty = gen_params->frequency_penalty;
        sampler_params.presence_penalty = gen_params->presence_penalty;
        
        void* sampler = llama_go_sampler_new(model_handle, sampler_params);
        if (!sampler) {
            llama_go_set_error("Failed to create sampler");
            return nullptr;
        }
        
        // Create batch using wrapper
        llama_go_batch* batch = llama_go_batch_init(n_ctx, 1);
        if (!batch) {
            llama_go_sampler_free(sampler);
            llama_go_set_error("Failed to create batch");
            return nullptr;
        }
        
        // Decode prompt
        llama_go_batch_clear(batch);
        for (int32_t i = 0; i < n_prompt; i++) {
            llama_go_batch_add(batch, prompt_tokens[i], i, 0, i == n_prompt - 1);
        }
        
        if (llama_go_batch_decode(ctx_handle, batch) != 0) {
            llama_go_batch_free(batch);
            llama_go_sampler_free(sampler);
            llama_go_set_error("Failed to decode prompt");
            return nullptr;
        }
        
        // Generation loop
        std::string generated_text;
        int32_t n_past = n_prompt;
        const int32_t max_tokens = gen_params->max_tokens;
        
        for (int32_t i = 0; i < max_tokens; i++) {
            // Sample next token
            int32_t new_token = llama_go_sampler_sample(sampler, ctx_handle, -1);
            
            // Accept token for repetition penalty
            llama_go_sampler_accept(sampler, new_token);
            
            // Check for end of generation
            const llama_model* llama_model_ptr = llama_get_model(ctx);
            const llama_vocab* vocab = llama_model_get_vocab(llama_model_ptr);
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }
            
            // Detokenize
            char piece[256];
            int32_t piece_len = llama_go_token_to_piece(
                model_handle,
                new_token,
                piece,
                sizeof(piece) - 1,
                false  // special
            );
            
            if (piece_len > 0) {
                piece[piece_len] = '\0';
                generated_text += piece;
                
                // Callback if provided
                if (gen_params->callback_handle) {
                    if (!goTokenCallback(gen_params->callback_handle, piece)) {
                        // Callback requested stop
                        break;
                    }
                }
                
                // Check stop words
                if (gen_params->stop_words) {
                    for (int j = 0; gen_params->stop_words[j] != nullptr; j++) {
                        const char* stop_word = gen_params->stop_words[j];
                        size_t stop_len = strlen(stop_word);
                        if (generated_text.length() >= stop_len &&
                            generated_text.substr(generated_text.length() - stop_len) == stop_word) {
                            // Found stop word - remove it and stop
                            generated_text.resize(generated_text.length() - stop_len);
                            llama_go_batch_free(batch);
                            llama_go_sampler_free(sampler);
                            
                            char* result = (char*)malloc(generated_text.length() + 1);
                            if (!result) {
                                llama_go_set_error("Failed to allocate result");
                                return nullptr;
                            }
                            strcpy(result, generated_text.c_str());
                            return result;
                        }
                    }
                }
            }
            
            // Prepare next iteration - decode new token
            llama_go_batch_clear(batch);
            llama_go_batch_add(batch, new_token, n_past, 0, true);
            n_past++;
            
            if (llama_go_batch_decode(ctx_handle, batch) != 0) {
                break;
            }
        }
        
        // Clean up
        llama_go_batch_free(batch);
        llama_go_sampler_free(sampler);
        
        // Allocate result
        char* result = (char*)malloc(generated_text.length() + 1);
        if (!result) {
            llama_go_set_error("Failed to allocate result");
            return nullptr;
        }
        strcpy(result, generated_text.c_str());
        return result;
        
    } catch (const std::exception& e) {
        llama_go_set_error(e.what());
        return nullptr;
    }
}

extern "C" void llama_go_completion_free_result(char* result) {
    if (result) {
        free(result);
    }
}
