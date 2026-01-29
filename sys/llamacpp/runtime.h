#ifndef LLAMA_GO_RUNTIME_H
#define LLAMA_GO_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runtime information about model
typedef struct {
    int32_t n_layer;         // Total number of layers
    int32_t n_head;          // Number of attention heads
    int32_t n_head_kv;       // Number of KV heads (for GQA/MQA)
    int32_t n_embd;          // Embedding dimension
    int32_t n_ctx_train;     // Training context length
    uint64_t n_params;       // Total parameter count
    uint64_t model_size;     // Model size in bytes
} llama_go_model_info;

// Runtime information about context
typedef struct {
    uint32_t n_ctx;          // Context size
    uint32_t n_batch;        // Batch size
    uint32_t n_ubatch;       // Micro-batch size
    uint32_t n_seq_max;      // Max sequences
    int32_t n_threads;       // Thread count
} llama_go_context_info;

// Performance timing data
typedef struct {
    double t_start_ms;       // Absolute start time (ms)
    double t_load_ms;        // Model loading time (ms)
    double t_p_eval_ms;      // Prompt processing time (ms)
    double t_eval_ms;        // Token generation time (ms)
    int32_t n_p_eval;        // Prompt tokens processed
    int32_t n_eval;          // Tokens generated
} llama_go_perf_data;

// Get model runtime information
bool llama_go_get_model_info(void* model_handle, llama_go_model_info* info);

// Get model description string
int32_t llama_go_model_desc(void* model_handle, char* buf, size_t buf_size);

// Get context runtime information
bool llama_go_get_context_info(void* ctx_handle, llama_go_context_info* info);

// Get performance timing data
bool llama_go_get_perf_data(void* ctx_handle, llama_go_perf_data* data);

// Reset performance counters
void llama_go_perf_reset(void* ctx_handle);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_RUNTIME_H
