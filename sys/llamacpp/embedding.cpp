#include "embedding.h"
#include "context.h"
#include "error.h"
#include "llama.h"
#include <cmath>

extern "C" {

void llama_go_set_embeddings(void *ctx_handle, bool embeddings) {
  if (ctx_handle == nullptr) {
    llama_go_set_error("set_embeddings: context handle is null");
    return;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  if (ctx == nullptr) {
    llama_go_set_error("set_embeddings: context is null");
    return;
  }

  llama_set_embeddings(ctx, embeddings);
}

llama_go_pooling_type llama_go_get_pooling_type(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return LLAMA_GO_POOLING_UNSPECIFIED;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  if (ctx == nullptr) {
    return LLAMA_GO_POOLING_UNSPECIFIED;
  }

  enum llama_pooling_type ptype = ::llama_pooling_type(ctx);
  return static_cast<llama_go_pooling_type>(ptype);
}

float *llama_go_get_all_embeddings(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    llama_go_set_error("get_all_embeddings: context handle is null");
    return nullptr;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  if (ctx == nullptr) {
    llama_go_set_error("get_all_embeddings: context is null");
    return nullptr;
  }

  float *embd = llama_get_embeddings(ctx);
  if (embd == nullptr) {
    llama_go_set_error("get_all_embeddings: no embeddings available (ensure "
                       "embeddings mode is enabled and batch was decoded)");
  }
  return embd;
}

float *llama_go_get_embeddings_seq(void *ctx_handle, int32_t seq_id) {
  if (ctx_handle == nullptr) {
    llama_go_set_error("get_embeddings_seq: context handle is null");
    return nullptr;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  if (ctx == nullptr) {
    llama_go_set_error("get_embeddings_seq: context is null");
    return nullptr;
  }

  float *embd = llama_get_embeddings_seq(ctx, seq_id);
  if (embd == nullptr) {
    llama_go_set_error("get_embeddings_seq: no embeddings for sequence (ensure "
                       "pooling_type != NONE)");
  }
  return embd;
}

void llama_go_normalize_embeddings(float *embd, int32_t n) {
  if (embd == nullptr || n <= 0) {
    return;
  }

  // L2 normalization
  double sum = 0.0;
  for (int32_t i = 0; i < n; i++) {
    sum += static_cast<double>(embd[i]) * static_cast<double>(embd[i]);
  }

  if (sum > 0.0) {
    double norm = std::sqrt(sum);
    for (int32_t i = 0; i < n; i++) {
      embd[i] = static_cast<float>(static_cast<double>(embd[i]) / norm);
    }
  }
}

} // extern "C"
