#include "context.h"
#include "error.h"
#include "ggml.h"
#include "llama.h"
#include "model.h"

#include <cstdlib>

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMS

extern "C" llama_go_context_params llama_go_context_default_params(void) {
  llama_context_params defaults = llama_context_default_params();

  llama_go_context_params params;
  params.n_ctx = defaults.n_ctx;
  params.n_batch = defaults.n_batch;
  params.n_ubatch = defaults.n_ubatch;
  params.n_seq_max = defaults.n_seq_max;
  params.n_threads = defaults.n_threads;
  params.n_threads_batch = defaults.n_threads_batch;
  params.rope_freq_base = defaults.rope_freq_base;
  params.rope_freq_scale = defaults.rope_freq_scale;
  params.type_k = -1; // -1 means use default (F16)
  params.type_v = -1; // -1 means use default (F16)
  params.attention_type = static_cast<int32_t>(defaults.attention_type);
  params.flash_attn = static_cast<int32_t>(defaults.flash_attn_type);
  params.embeddings = defaults.embeddings;
  params.offload_kqv = defaults.offload_kqv;
  params.kv_unified = defaults.kv_unified;
  params.no_perf = defaults.no_perf;

  return params;
}

///////////////////////////////////////////////////////////////////////////////
// CONTEXT LIFECYCLE

extern "C" void *llama_go_context_new(void *model_handle,
                                      llama_go_context_params params) {
  if (model_handle == nullptr) {
    llama_go_set_error("model handle is null");
    return nullptr;
  }

  // Get the underlying llama_model from the model wrapper
  llama_model *model = llama_go_model_get_llama_model(model_handle);
  if (model == nullptr) {
    llama_go_set_error("failed to get llama_model from handle");
    return nullptr;
  }

  // Convert our params to llama_context_params
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = params.n_ctx;
  ctx_params.n_batch = params.n_batch;
  ctx_params.n_ubatch = params.n_ubatch;
  ctx_params.n_seq_max = params.n_seq_max;
  ctx_params.n_threads = params.n_threads;
  ctx_params.n_threads_batch = params.n_threads_batch;
  ctx_params.rope_freq_base = params.rope_freq_base;
  ctx_params.rope_freq_scale = params.rope_freq_scale;
  ctx_params.attention_type =
      static_cast<enum llama_attention_type>(params.attention_type);
  ctx_params.flash_attn_type =
      static_cast<enum llama_flash_attn_type>(params.flash_attn);
  ctx_params.embeddings = params.embeddings;
  ctx_params.offload_kqv = params.offload_kqv;
  ctx_params.kv_unified = params.kv_unified;
  ctx_params.no_perf = params.no_perf;

  // Set KV cache quantization types if specified
  if (params.type_k >= 0) {
    ctx_params.type_k = static_cast<ggml_type>(params.type_k);
  }
  if (params.type_v >= 0) {
    ctx_params.type_v = static_cast<ggml_type>(params.type_v);
  }

  // Create the context
  llama_context *ctx = llama_init_from_model(model, ctx_params);
  if (ctx == nullptr) {
    llama_go_set_error("failed to create context");
    return nullptr;
  }

  return static_cast<void *>(ctx);
}

extern "C" void llama_go_context_free(void *ctx_handle) {
  if (ctx_handle != nullptr) {
    llama_context *ctx = static_cast<llama_context *>(ctx_handle);
    llama_free(ctx);
  }
}

///////////////////////////////////////////////////////////////////////////////
// CONTEXT INFO

extern "C" uint32_t llama_go_context_n_ctx(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_n_ctx(ctx);
}

extern "C" uint32_t llama_go_context_n_batch(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_n_batch(ctx);
}

extern "C" uint32_t llama_go_context_n_ubatch(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_n_ubatch(ctx);
}

extern "C" uint32_t llama_go_context_n_seq_max(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_n_seq_max(ctx);
}

extern "C" uint32_t llama_go_context_n_ctx_seq(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_n_ctx_seq(ctx);
}

extern "C" int32_t llama_go_context_n_threads(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  // llama doesn't expose n_threads getter, return 0 for now
  // Could store it in a wrapper struct if needed
  (void)ctx;
  return 0;
}

extern "C" void *llama_go_context_get_model(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return nullptr;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  // Note: This returns the llama_model*, not our model wrapper
  // The caller should be aware of this
  return const_cast<llama_model *>(llama_get_model(ctx));
}

///////////////////////////////////////////////////////////////////////////////
// GGML TYPE UTILITIES

extern "C" const char *llama_go_ggml_type_name(int32_t type) {
  return ggml_type_name(static_cast<ggml_type>(type));
}

///////////////////////////////////////////////////////////////////////////////
// Note: Performance monitoring functions are implemented in runtime.cpp
// Note: All memory/KV cache operations are implemented in decode.cpp

///////////////////////////////////////////////////////////////////////////////
// STATE SAVE/LOAD

extern "C" size_t llama_go_state_get_size(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_get_size(ctx);
}

extern "C" size_t llama_go_state_get_data(void *ctx_handle, uint8_t *dst,
                                          size_t size) {
  if (ctx_handle == nullptr || dst == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_get_data(ctx, dst, size);
}

extern "C" size_t llama_go_state_set_data(void *ctx_handle, const uint8_t *src,
                                          size_t size) {
  if (ctx_handle == nullptr || src == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_set_data(ctx, src, size);
}

extern "C" bool llama_go_state_save_file(void *ctx_handle, const char *path,
                                         const int32_t *tokens,
                                         size_t n_tokens) {
  if (ctx_handle == nullptr || path == nullptr) {
    return false;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_save_file(
      ctx, path, reinterpret_cast<const llama_token *>(tokens), n_tokens);
}

extern "C" bool llama_go_state_load_file(void *ctx_handle, const char *path,
                                         int32_t *tokens_out,
                                         size_t n_token_capacity,
                                         size_t *n_tokens_out) {
  if (ctx_handle == nullptr || path == nullptr) {
    return false;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_load_file(ctx, path,
                               reinterpret_cast<llama_token *>(tokens_out),
                               n_token_capacity, n_tokens_out);
}

extern "C" size_t llama_go_state_seq_get_size(void *ctx_handle,
                                              int32_t seq_id) {
  if (ctx_handle == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_seq_get_size(ctx, seq_id);
}

extern "C" size_t llama_go_state_seq_get_data(void *ctx_handle, uint8_t *dst,
                                              size_t size, int32_t seq_id) {
  if (ctx_handle == nullptr || dst == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_seq_get_data(ctx, dst, size, seq_id);
}

extern "C" size_t llama_go_state_seq_set_data(void *ctx_handle,
                                              const uint8_t *src, size_t size,
                                              int32_t dest_seq_id) {
  if (ctx_handle == nullptr || src == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_seq_set_data(ctx, src, size, dest_seq_id);
}

extern "C" size_t llama_go_state_seq_save_file(void *ctx_handle,
                                               const char *path, int32_t seq_id,
                                               const int32_t *tokens,
                                               size_t n_tokens) {
  if (ctx_handle == nullptr || path == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_seq_save_file(
      ctx, path, seq_id, reinterpret_cast<const llama_token *>(tokens),
      n_tokens);
}

extern "C" size_t llama_go_state_seq_load_file(
    void *ctx_handle, const char *path, int32_t dest_seq_id,
    int32_t *tokens_out, size_t n_token_capacity, size_t *n_token_count_out) {
  if (ctx_handle == nullptr || path == nullptr) {
    return 0;
  }
  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  return llama_state_seq_load_file(ctx, path, dest_seq_id,
                                   reinterpret_cast<llama_token *>(tokens_out),
                                   n_token_capacity, n_token_count_out);
}
