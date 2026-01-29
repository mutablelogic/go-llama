#ifndef LLAMA_GO_BATCH_H
#define LLAMA_GO_BATCH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque batch handle
typedef struct llama_go_batch llama_go_batch;

// Allocate a new batch that can hold up to n_tokens
// n_seq_max is the maximum number of sequence IDs per token
// Returns NULL on failure
llama_go_batch* llama_go_batch_init(int32_t n_tokens, int32_t n_seq_max);

// Free a batch
void llama_go_batch_free(llama_go_batch* batch);

// Clear the batch (reset n_tokens to 0)
void llama_go_batch_clear(llama_go_batch* batch);

// Get current number of tokens in the batch
int32_t llama_go_batch_n_tokens(llama_go_batch* batch);

// Get capacity of the batch
int32_t llama_go_batch_capacity(llama_go_batch* batch);

// Add a single token to the batch
// pos: position in the sequence
// seq_id: sequence ID
// logits: whether to compute logits for this token
// Returns false if batch is full
bool llama_go_batch_add(llama_go_batch* batch, int32_t token, int32_t pos, int32_t seq_id, bool logits);

// Add a single token with multiple sequence IDs
// seq_ids: array of sequence IDs
// n_seq: number of sequence IDs
// Returns false if batch is full
bool llama_go_batch_add_seq(llama_go_batch* batch, int32_t token, int32_t pos,
                            const int32_t* seq_ids, int32_t n_seq, bool logits);

// Add multiple tokens at once (all with same sequence ID)
// tokens: array of token IDs
// n_tokens: number of tokens
// pos_start: starting position
// seq_id: sequence ID for all tokens
// logits_last: if true, only compute logits for last token
// Returns number of tokens added (may be less than n_tokens if batch fills up)
int32_t llama_go_batch_add_tokens(llama_go_batch* batch, const int32_t* tokens, int32_t n_tokens,
                                   int32_t pos_start, int32_t seq_id, bool logits_last);

// Set logits flag for a specific token index
void llama_go_batch_set_logits(llama_go_batch* batch, int32_t idx, bool logits);

// Decode the batch using the given context
// Returns: 0 on success, 1 if no KV slot, -1 on error
int32_t llama_go_batch_decode(void* ctx_handle, llama_go_batch* batch);

// Encode the batch (for encoder-decoder models)
// Returns: 0 on success, negative on error
int32_t llama_go_batch_encode(void* ctx_handle, llama_go_batch* batch);

// Get a pointer to the underlying llama_batch for direct access
// This is mainly for internal use
void* llama_go_batch_get_native(llama_go_batch* batch);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_BATCH_H
