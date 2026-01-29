#include "batch.h"
#include "error.h"
#include "llama.h"
#include <cstdlib>
#include <cstring>

// Internal batch structure
struct llama_go_batch {
    llama_batch batch;      // The underlying llama.cpp batch
    int32_t capacity;       // Maximum tokens this batch can hold
    int32_t n_seq_max;      // Maximum sequence IDs per token
};

extern "C" {

llama_go_batch* llama_go_batch_init(int32_t n_tokens, int32_t n_seq_max) {
    if (n_tokens <= 0 || n_seq_max <= 0) {
        llama_go_set_error("batch init: invalid parameters");
        return nullptr;
    }

    auto* b = new (std::nothrow) llama_go_batch;
    if (!b) {
        llama_go_set_error("batch init: allocation failed");
        return nullptr;
    }

    // Initialize the llama batch (0 for embd means we use tokens, not embeddings)
    b->batch = llama_batch_init(n_tokens, 0, n_seq_max);
    b->capacity = n_tokens;
    b->n_seq_max = n_seq_max;

    // Start with empty batch
    b->batch.n_tokens = 0;

    return b;
}

void llama_go_batch_free(llama_go_batch* batch) {
    if (batch) {
        llama_batch_free(batch->batch);
        delete batch;
    }
}

void llama_go_batch_clear(llama_go_batch* batch) {
    if (batch) {
        batch->batch.n_tokens = 0;
    }
}

int32_t llama_go_batch_n_tokens(llama_go_batch* batch) {
    return batch ? batch->batch.n_tokens : 0;
}

int32_t llama_go_batch_capacity(llama_go_batch* batch) {
    return batch ? batch->capacity : 0;
}

bool llama_go_batch_add(llama_go_batch* batch, int32_t token, int32_t pos, int32_t seq_id, bool logits) {
    if (!batch) {
        return false;
    }

    if (batch->batch.n_tokens >= batch->capacity) {
        return false;
    }

    int32_t idx = batch->batch.n_tokens;

    batch->batch.token[idx] = token;
    batch->batch.pos[idx] = pos;
    batch->batch.n_seq_id[idx] = 1;
    batch->batch.seq_id[idx][0] = seq_id;
    batch->batch.logits[idx] = logits ? 1 : 0;

    batch->batch.n_tokens++;
    return true;
}

bool llama_go_batch_add_seq(llama_go_batch* batch, int32_t token, int32_t pos,
                            const int32_t* seq_ids, int32_t n_seq, bool logits) {
    if (!batch || !seq_ids || n_seq <= 0) {
        return false;
    }

    if (batch->batch.n_tokens >= batch->capacity) {
        return false;
    }

    if (n_seq > batch->n_seq_max) {
        n_seq = batch->n_seq_max;  // Clamp to max
    }

    int32_t idx = batch->batch.n_tokens;

    batch->batch.token[idx] = token;
    batch->batch.pos[idx] = pos;
    batch->batch.n_seq_id[idx] = n_seq;
    for (int32_t i = 0; i < n_seq; i++) {
        batch->batch.seq_id[idx][i] = seq_ids[i];
    }
    batch->batch.logits[idx] = logits ? 1 : 0;

    batch->batch.n_tokens++;
    return true;
}

int32_t llama_go_batch_add_tokens(llama_go_batch* batch, const int32_t* tokens, int32_t n_tokens,
                                   int32_t pos_start, int32_t seq_id, bool logits_last) {
    if (!batch || !tokens || n_tokens <= 0) {
        return 0;
    }

    int32_t added = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        if (batch->batch.n_tokens >= batch->capacity) {
            break;
        }

        int32_t idx = batch->batch.n_tokens;
        bool output_logits = logits_last ? (i == n_tokens - 1) : false;

        batch->batch.token[idx] = tokens[i];
        batch->batch.pos[idx] = pos_start + i;
        batch->batch.n_seq_id[idx] = 1;
        batch->batch.seq_id[idx][0] = seq_id;
        batch->batch.logits[idx] = output_logits ? 1 : 0;

        batch->batch.n_tokens++;
        added++;
    }

    return added;
}

void llama_go_batch_set_logits(llama_go_batch* batch, int32_t idx, bool logits) {
    if (batch && idx >= 0 && idx < batch->batch.n_tokens) {
        batch->batch.logits[idx] = logits ? 1 : 0;
    }
}

int32_t llama_go_batch_decode(void* ctx_handle, llama_go_batch* batch) {
    if (!ctx_handle || !batch) {
        llama_go_set_error("batch decode: invalid context or batch");
        return -1;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    int32_t result = llama_decode(ctx, batch->batch);

    if (result < 0) {
        llama_go_set_error("batch decode failed");
    } else if (result == 1) {
        llama_go_set_error("batch decode: no KV slot available");
    }

    return result;
}

int32_t llama_go_batch_encode(void* ctx_handle, llama_go_batch* batch) {
    if (!ctx_handle || !batch) {
        llama_go_set_error("batch encode: invalid context or batch");
        return -1;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    int32_t result = llama_encode(ctx, batch->batch);

    if (result < 0) {
        llama_go_set_error("batch encode failed");
    }

    return result;
}

void* llama_go_batch_get_native(llama_go_batch* batch) {
    return batch ? &batch->batch : nullptr;
}

} // extern "C"
