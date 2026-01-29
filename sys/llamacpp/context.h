#ifndef LLAMA_GO_CONTEXT_H
#define LLAMA_GO_CONTEXT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML data types for KV cache quantization
typedef enum {
    LLAMA_GO_TYPE_F32     = 0,
    LLAMA_GO_TYPE_F16     = 1,
    LLAMA_GO_TYPE_Q4_0    = 2,
    LLAMA_GO_TYPE_Q4_1    = 3,
    LLAMA_GO_TYPE_Q5_0    = 6,
    LLAMA_GO_TYPE_Q5_1    = 7,
    LLAMA_GO_TYPE_Q8_0    = 8,
    LLAMA_GO_TYPE_Q8_1    = 9,
    LLAMA_GO_TYPE_BF16    = 30,
} llama_go_ggml_type;

// Context parameters (simplified subset of llama_context_params)
typedef struct {
    uint32_t n_ctx;           // text context size, 0 = from model
    uint32_t n_batch;         // logical maximum batch size
    uint32_t n_ubatch;        // physical maximum batch size
    uint32_t n_seq_max;       // max number of sequences
    int32_t  n_threads;       // threads for generation
    int32_t  n_threads_batch; // threads for batch processing
    float    rope_freq_base;  // RoPE base frequency, 0 = from model
    float    rope_freq_scale; // RoPE frequency scaling factor, 0 = from model
    int32_t  type_k;          // KV cache K type (llama_go_ggml_type), -1 = default (F16)
    int32_t  type_v;          // KV cache V type (llama_go_ggml_type), -1 = default (F16)
    bool     embeddings;      // if true, extract embeddings
    bool     offload_kqv;     // offload KQV ops to GPU
    bool     flash_attn;      // use flash attention
    bool     no_perf;         // disable performance timings
} llama_go_context_params;

// Get default context parameters
llama_go_context_params llama_go_context_default_params(void);

// Create a new context from a model
// Returns NULL on failure (check llama_go_last_error)
void* llama_go_context_new(void* model_handle, llama_go_context_params params);

// Free a context
void llama_go_context_free(void* ctx_handle);

// Get context info
uint32_t llama_go_context_n_ctx(void* ctx_handle);
uint32_t llama_go_context_n_batch(void* ctx_handle);
uint32_t llama_go_context_n_ubatch(void* ctx_handle);
uint32_t llama_go_context_n_seq_max(void* ctx_handle);
int32_t  llama_go_context_n_threads(void* ctx_handle);

// Get the model associated with this context
void* llama_go_context_get_model(void* ctx_handle);

// Get human-readable GGML type name (e.g., "f16", "q8_0")
const char* llama_go_ggml_type_name(int32_t type);

///////////////////////////////////////////////////////////////////////////////
// MEMORY / KV CACHE OPERATIONS (additional for prefix caching)

// Remove all tokens that do not belong to the specified sequence
void llama_go_memory_seq_keep(void* ctx_handle, int32_t seq_id);

///////////////////////////////////////////////////////////////////////////////
// STATE SAVE/LOAD (for persistent prefix caching)

// Get the size needed to save a sequence state
size_t llama_go_state_seq_get_size(void* ctx_handle, int32_t seq_id);

// Copy sequence state to buffer, returns bytes written
size_t llama_go_state_seq_get_data(void* ctx_handle, uint8_t* dst, size_t size, int32_t seq_id);

// Restore sequence state from buffer, returns bytes read (0 on failure)
size_t llama_go_state_seq_set_data(void* ctx_handle, const uint8_t* src, size_t size, int32_t dest_seq_id);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_CONTEXT_H
