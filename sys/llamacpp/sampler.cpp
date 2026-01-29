#include "sampler.h"
#include "model.h"
#include "error.h"
#include <llama.h>
#include <cstdlib>
#include <ctime>

// Get the underlying llama_model from our wrapper
extern "C" struct llama_model* llama_go_model_get_llama_model(void* model);

extern "C" {

llama_go_sampler_params llama_go_sampler_default_params(void) {
    llama_go_sampler_params params;
    params.seed = 0;              // Will use random seed
    params.temperature = 0.8f;
    params.top_k = 40;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.repeat_penalty = 1.1f;
    params.repeat_last_n = 64;
    params.frequency_penalty = 0.0f;
    params.presence_penalty = 0.0f;
    return params;
}

void* llama_go_sampler_new(void* model, llama_go_sampler_params params) {
    if (!model) {
        llama_go_set_error("invalid model");
        return nullptr;
    }

    // Create chain
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (!chain) {
        llama_go_set_error("failed to create sampler chain");
        return nullptr;
    }

    // Add samplers in the recommended order:
    // 1. Penalties (if enabled)
    if (params.repeat_penalty != 1.0f || params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            params.repeat_last_n,
            params.repeat_penalty,
            params.frequency_penalty,
            params.presence_penalty
        ));
    }

    // 2. Top-K (if enabled)
    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }

    // 3. Top-P (if enabled)
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }

    // 4. Min-P (if enabled)
    if (params.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1));
    }

    // 5. Temperature
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
    }

    // 6. Final sampling
    uint32_t seed = params.seed;
    if (seed == 0) {
        // Generate random seed
        seed = (uint32_t)time(nullptr);
    }
    
    if (params.temperature == 0.0f) {
        // Greedy sampling
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        // Probabilistic sampling
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    return chain;
}

void llama_go_sampler_free(void* sampler) {
    if (sampler) {
        llama_sampler_free((struct llama_sampler*)sampler);
    }
}

int32_t llama_go_sampler_sample(void* sampler, void* context, int32_t idx) {
    if (!sampler || !context) {
        llama_go_set_error("invalid sampler or context");
        return -1;
    }

    struct llama_sampler* smpl = (struct llama_sampler*)sampler;
    struct llama_context* ctx = (struct llama_context*)context;

    return llama_sampler_sample(smpl, ctx, idx);
}

void llama_go_sampler_reset(void* sampler) {
    if (sampler) {
        llama_sampler_reset((struct llama_sampler*)sampler);
    }
}

void llama_go_sampler_accept(void* sampler, int32_t token) {
    if (sampler) {
        llama_sampler_accept((struct llama_sampler*)sampler, token);
    }
}

int32_t llama_go_sampler_chain_n(void* sampler) {
    if (!sampler) return 0;
    return llama_sampler_chain_n((struct llama_sampler*)sampler);
}

// Individual sampler constructors

void* llama_go_sampler_init_greedy(void) {
    return llama_sampler_init_greedy();
}

void* llama_go_sampler_init_dist(uint32_t seed) {
    return llama_sampler_init_dist(seed);
}

void* llama_go_sampler_init_top_k(int32_t k) {
    return llama_sampler_init_top_k(k);
}

void* llama_go_sampler_init_top_p(float p, size_t min_keep) {
    return llama_sampler_init_top_p(p, min_keep);
}

void* llama_go_sampler_init_min_p(float p, size_t min_keep) {
    return llama_sampler_init_min_p(p, min_keep);
}

void* llama_go_sampler_init_temp(float t) {
    return llama_sampler_init_temp(t);
}

void* llama_go_sampler_init_penalties(
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present
) {
    return llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present);
}

void* llama_go_sampler_chain_init(bool no_perf) {
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = no_perf;
    return llama_sampler_chain_init(params);
}

void llama_go_sampler_chain_add(void* chain, void* smpl) {
    if (chain && smpl) {
        llama_sampler_chain_add((struct llama_sampler*)chain, (struct llama_sampler*)smpl);
    }
}

} // extern "C"
