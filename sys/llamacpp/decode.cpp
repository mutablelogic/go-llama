#include "decode.h"
#include "error.h"
#include "llama.h"

extern "C" {

float* llama_go_get_logits(void* ctx_handle, int32_t idx) {
    if (!ctx_handle) {
        llama_go_set_error("get_logits: invalid context");
        return nullptr;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    float* logits = llama_get_logits_ith(ctx, idx);

    if (!logits) {
        llama_go_set_error("get_logits: no logits for index");
        return nullptr;
    }

    return logits;
}

int32_t llama_go_ctx_n_vocab(void* ctx_handle) {
    if (!ctx_handle) {
        return 0;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    const llama_model* model = llama_get_model(ctx);
    if (!model) {
        return 0;
    }

    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}

float* llama_go_get_embeddings(void* ctx_handle, int32_t idx) {
    if (!ctx_handle) {
        llama_go_set_error("get_embeddings: invalid context");
        return nullptr;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    float* embd = llama_get_embeddings_ith(ctx, idx);

    if (!embd) {
        llama_go_set_error("get_embeddings: no embeddings for index");
        return nullptr;
    }

    return embd;
}

int32_t llama_go_n_embd(void* ctx_handle) {
    if (!ctx_handle) {
        return 0;
    }

    auto* ctx = static_cast<llama_context*>(ctx_handle);
    const llama_model* model = llama_get_model(ctx);
    if (!model) {
        return 0;
    }

    return llama_model_n_embd(model);
}

void llama_go_memory_clear(void* ctx_handle, bool clear_data) {
    if (ctx_handle) {
        auto* ctx = static_cast<llama_context*>(ctx_handle);
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, clear_data);
        }
    }
}

bool llama_go_memory_seq_rm(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx_handle) {
        return false;
    }
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return false;
    }
    return llama_memory_seq_rm(mem, seq_id, p0, p1);
}

void llama_go_memory_seq_cp(void* ctx_handle, int32_t seq_id_src, int32_t seq_id_dst, int32_t p0, int32_t p1) {
    if (ctx_handle) {
        auto* ctx = static_cast<llama_context*>(ctx_handle);
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
        }
    }
}

void llama_go_memory_seq_keep(void* ctx_handle, int32_t seq_id) {
    if (ctx_handle) {
        auto* ctx = static_cast<llama_context*>(ctx_handle);
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_keep(mem, seq_id);
        }
    }
}

void llama_go_memory_seq_div(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1, int32_t d) {
    if (ctx_handle) {
        auto* ctx = static_cast<llama_context*>(ctx_handle);
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_div(mem, seq_id, p0, p1, d);
        }
    }
}

void llama_go_memory_seq_add(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) {
    if (ctx_handle) {
        auto* ctx = static_cast<llama_context*>(ctx_handle);
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        }
    }
}

int32_t llama_go_memory_seq_pos_min(void* ctx_handle, int32_t seq_id) {
    if (!ctx_handle) {
        return -1;
    }
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return -1;
    }
    return llama_memory_seq_pos_min(mem, seq_id);
}

int32_t llama_go_memory_seq_pos_max(void* ctx_handle, int32_t seq_id) {
    if (!ctx_handle) {
        return -1;
    }
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return -1;
    }
    return llama_memory_seq_pos_max(mem, seq_id);
}

bool llama_go_memory_can_shift(void* ctx_handle) {
    if (!ctx_handle) {
        return false;
    }
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return false;
    }
    return llama_memory_can_shift(mem);
}

void llama_go_synchronize(void* ctx_handle) {
    if (ctx_handle) {
        llama_synchronize(static_cast<llama_context*>(ctx_handle));
    }
}

} // extern "C"
