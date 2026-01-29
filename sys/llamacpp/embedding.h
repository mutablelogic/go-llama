#ifndef LLAMA_GO_EMBEDDING_H
#define LLAMA_GO_EMBEDDING_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pooling types for embeddings
typedef enum {
    LLAMA_GO_POOLING_UNSPECIFIED = -1,
    LLAMA_GO_POOLING_NONE = 0,
    LLAMA_GO_POOLING_MEAN = 1,
    LLAMA_GO_POOLING_CLS = 2,
    LLAMA_GO_POOLING_LAST = 3,
    LLAMA_GO_POOLING_RANK = 4
} llama_go_pooling_type;

// Enable or disable embedding mode on a context
void llama_go_set_embeddings(void* ctx_handle, bool embeddings);

// Get the pooling type for the context
llama_go_pooling_type llama_go_get_pooling_type(void* ctx_handle);

// Get all embeddings as a contiguous buffer
// Returns pointer to embeddings array or NULL on error
// Size depends on n_outputs * n_embd
float* llama_go_get_all_embeddings(void* ctx_handle);

// Get embeddings for a specific sequence ID (pooled)
// Returns NULL if pooling_type is NONE
// Size is n_embd (or n_cls_out for RANK pooling)
float* llama_go_get_embeddings_seq(void* ctx_handle, int32_t seq_id);

// Normalize a vector in-place (L2 normalization)
void llama_go_normalize_embeddings(float* embd, int32_t n);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_EMBEDDING_H
