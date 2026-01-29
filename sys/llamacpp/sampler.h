#ifndef GO_LLAMA_SAMPLER_H
#define GO_LLAMA_SAMPLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Sampler configuration for common use cases
typedef struct llama_go_sampler_params {
    // Seed for random sampling (0 = random)
    uint32_t seed;
    
    // Temperature (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
    float temperature;
    
    // Top-K sampling (0 = disabled)
    int32_t top_k;
    
    // Top-P (nucleus) sampling (1.0 = disabled)
    float top_p;
    
    // Min-P sampling (0.0 = disabled)
    float min_p;
    
    // Repetition penalty (1.0 = disabled)
    float repeat_penalty;
    
    // Number of tokens to consider for repetition penalty
    int32_t repeat_last_n;
    
    // Frequency penalty (0.0 = disabled)
    float frequency_penalty;
    
    // Presence penalty (0.0 = disabled)
    float presence_penalty;
} llama_go_sampler_params;

// Get default sampler parameters
llama_go_sampler_params llama_go_sampler_default_params(void);

// Create a sampler chain with common settings
// Returns NULL on error
void* llama_go_sampler_new(void* model, llama_go_sampler_params params);

// Free a sampler
void llama_go_sampler_free(void* sampler);

// Sample the next token from the context at the given position
// Returns the sampled token, or -1 on error
int32_t llama_go_sampler_sample(void* sampler, void* context, int32_t idx);

// Reset the sampler state (clears any accumulated state like repetition tracking)
void llama_go_sampler_reset(void* sampler);

// Accept a token (for repetition penalty tracking)
void llama_go_sampler_accept(void* sampler, int32_t token);

// Get the number of samplers in the chain
int32_t llama_go_sampler_chain_n(void* sampler);

// Create individual samplers for advanced usage
void* llama_go_sampler_init_greedy(void);
void* llama_go_sampler_init_dist(uint32_t seed);
void* llama_go_sampler_init_top_k(int32_t k);
void* llama_go_sampler_init_top_p(float p, size_t min_keep);
void* llama_go_sampler_init_min_p(float p, size_t min_keep);
void* llama_go_sampler_init_temp(float t);
void* llama_go_sampler_init_penalties(
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present
);

// Create a sampler chain
void* llama_go_sampler_chain_init(bool no_perf);

// Add a sampler to a chain (takes ownership of smpl)
void llama_go_sampler_chain_add(void* chain, void* smpl);

#ifdef __cplusplus
}
#endif

#endif // GO_LLAMA_SAMPLER_H
