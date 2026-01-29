#ifndef LLAMA_GO_DECODE_H
#define LLAMA_GO_DECODE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get logits for a specific token index after decode
// idx: the index of the token in the batch that had logits=true
//      use -1 for the last token with logits
// Returns pointer to logits array of size n_vocab, or NULL on error
float* llama_go_get_logits(void* ctx_handle, int32_t idx);

// Get the number of vocabulary tokens (size of logits array) from context
int32_t llama_go_ctx_n_vocab(void* ctx_handle);

// Get embeddings for a specific token index (for embedding models)
// Returns pointer to embeddings array of size n_embd, or NULL on error
float* llama_go_get_embeddings(void* ctx_handle, int32_t idx);

// Get the embedding dimension
int32_t llama_go_n_embd(void* ctx_handle);

// Clear the memory (KV cache) contents
// If clear_data is true, the data buffers will also be cleared
void llama_go_memory_clear(void* ctx_handle, bool clear_data);

// Remove tokens from memory
// seq_id: sequence ID (-1 for all sequences)
// p0: start position (inclusive, -1 for 0)
// p1: end position (exclusive, -1 for end)
// Returns false if partial removal not supported
bool llama_go_memory_seq_rm(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1);

// Copy sequence in memory
void llama_go_memory_seq_cp(void* ctx_handle, int32_t seq_id_src, int32_t seq_id_dst, int32_t p0, int32_t p1);

// Remove all tokens that do not belong to the specified sequence
void llama_go_memory_seq_keep(void* ctx_handle, int32_t seq_id);

// Divide positions in a sequence range [p0, p1) by d (integer division)
void llama_go_memory_seq_div(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1, int32_t d);

// Shift positions in memory (for context shifting)
void llama_go_memory_seq_add(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1, int32_t delta);

// Get minimum position for a sequence (-1 if empty)
int32_t llama_go_memory_seq_pos_min(void* ctx_handle, int32_t seq_id);

// Get maximum position for a sequence (-1 if empty)
int32_t llama_go_memory_seq_pos_max(void* ctx_handle, int32_t seq_id);

// Check if memory supports context shifting
bool llama_go_memory_can_shift(void* ctx_handle);

// Synchronize computation (wait for GPU to finish)
void llama_go_synchronize(void* ctx_handle);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_DECODE_H
