#ifndef LLAMA_GO_CONTEXT_H
#define LLAMA_GO_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML data types for KV cache quantization
typedef enum {
  LLAMA_GO_TYPE_F32 = 0,
  LLAMA_GO_TYPE_F16 = 1,
  LLAMA_GO_TYPE_Q4_0 = 2,
  LLAMA_GO_TYPE_Q4_1 = 3,
  LLAMA_GO_TYPE_Q5_0 = 6,
  LLAMA_GO_TYPE_Q5_1 = 7,
  LLAMA_GO_TYPE_Q8_0 = 8,
  LLAMA_GO_TYPE_Q8_1 = 9,
  LLAMA_GO_TYPE_BF16 = 30,
} llama_go_ggml_type;

// Attention type for embedding models
typedef enum {
  LLAMA_GO_ATTENTION_TYPE_UNSPECIFIED = -1,
  LLAMA_GO_ATTENTION_TYPE_CAUSAL = 0,
  LLAMA_GO_ATTENTION_TYPE_NON_CAUSAL = 1,
} llama_go_attention_type;

// Flash attention type
typedef enum {
  LLAMA_GO_FLASH_ATTN_AUTO = -1,
  LLAMA_GO_FLASH_ATTN_DISABLED = 0,
  LLAMA_GO_FLASH_ATTN_ENABLED = 1,
} llama_go_flash_attn_type;

// Context parameters (simplified subset of llama_context_params)
typedef struct {
  uint32_t n_ctx;          // text context size, 0 = from model
  uint32_t n_batch;        // logical maximum batch size
  uint32_t n_ubatch;       // physical maximum batch size
  uint32_t n_seq_max;      // max number of sequences
  int32_t n_threads;       // threads for generation
  int32_t n_threads_batch; // threads for batch processing
  float rope_freq_base;    // RoPE base frequency, 0 = from model
  float rope_freq_scale;   // RoPE frequency scaling factor, 0 = from model
  int32_t type_k; // KV cache K type (llama_go_ggml_type), -1 = default (F16)
  int32_t type_v; // KV cache V type (llama_go_ggml_type), -1 = default (F16)
  int32_t attention_type; // attention type for embeddings (-1 = unspecified, 0
                          // = causal, 1 = non-causal)
  int32_t
      flash_attn; // flash attention type (-1 = auto, 0 = disabled, 1 = enabled)
  bool embeddings;  // if true, extract embeddings
  bool offload_kqv; // offload KQV ops to GPU
  bool kv_unified;  // use unified KV cache (required for encoder/BERT models)
  bool no_perf;     // disable performance timings
} llama_go_context_params;

// Get default context parameters
llama_go_context_params llama_go_context_default_params(void);

// Create a new context from a model
// Returns NULL on failure (check llama_go_last_error)
void *llama_go_context_new(void *model_handle, llama_go_context_params params);

// Free a context
void llama_go_context_free(void *ctx_handle);

// Get context info
uint32_t llama_go_context_n_ctx(void *ctx_handle);
uint32_t llama_go_context_n_batch(void *ctx_handle);
uint32_t llama_go_context_n_ubatch(void *ctx_handle);
uint32_t llama_go_context_n_seq_max(void *ctx_handle);
uint32_t llama_go_context_n_ctx_seq(void *ctx_handle);
int32_t llama_go_context_n_threads(void *ctx_handle);

// Get the model associated with this context
void *llama_go_context_get_model(void *ctx_handle);

// Get human-readable GGML type name (e.g., "f16", "q8_0")
const char *llama_go_ggml_type_name(int32_t type);

///////////////////////////////////////////////////////////////////////////////
// STATE SAVE/LOAD

// Get the size needed to save full context state
size_t llama_go_state_get_size(void *ctx_handle);

// Copy full context state to buffer, returns bytes written
size_t llama_go_state_get_data(void *ctx_handle, uint8_t *dst, size_t size);

// Restore full context state from buffer, returns bytes read (0 on failure)
size_t llama_go_state_set_data(void *ctx_handle, const uint8_t *src,
                               size_t size);

// Save full context state to file
// tokens_out: buffer to receive tokens (can be NULL)
// n_token_capacity: size of tokens_out buffer
// Returns true on success
bool llama_go_state_save_file(void *ctx_handle, const char *path,
                              const int32_t *tokens, size_t n_tokens);

// Load full context state from file
// tokens_out: buffer to receive tokens (can be NULL)
// n_token_capacity: size of tokens_out buffer
// n_tokens_out: receives number of tokens read (can be NULL)
// Returns true on success
bool llama_go_state_load_file(void *ctx_handle, const char *path,
                              int32_t *tokens_out, size_t n_token_capacity,
                              size_t *n_tokens_out);

// Get the size needed to save a sequence state
size_t llama_go_state_seq_get_size(void *ctx_handle, int32_t seq_id);

// Copy sequence state to buffer, returns bytes written
size_t llama_go_state_seq_get_data(void *ctx_handle, uint8_t *dst, size_t size,
                                   int32_t seq_id);

// Restore sequence state from buffer, returns bytes read (0 on failure)
size_t llama_go_state_seq_set_data(void *ctx_handle, const uint8_t *src,
                                   size_t size, int32_t dest_seq_id);

// Save sequence state to file, returns bytes written (0 on failure)
size_t llama_go_state_seq_save_file(void *ctx_handle, const char *path,
                                    int32_t seq_id, const int32_t *tokens,
                                    size_t n_tokens);

// Load sequence state from file, returns bytes read (0 on failure)
size_t llama_go_state_seq_load_file(void *ctx_handle, const char *path,
                                    int32_t dest_seq_id, int32_t *tokens_out,
                                    size_t n_token_capacity,
                                    size_t *n_token_count_out);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_CONTEXT_H
