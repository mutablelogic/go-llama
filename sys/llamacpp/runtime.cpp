#include "runtime.h"
#include "context.h"
#include "error.h"
#include "llama.h"
#include "model.h"

// Note: Context handle is a raw llama_context*, not a wrapper struct

extern "C" {

bool llama_go_get_model_info(void *model_handle, llama_go_model_info *info) {
  if (model_handle == nullptr || info == nullptr) {
    llama_go_set_error("get_model_info: null handle or info");
    return false;
  }

  llama_model *model = llama_go_model_get_llama_model(model_handle);
  if (model == nullptr) {
    llama_go_set_error("get_model_info: model is null");
    return false;
  }

  info->n_layer = llama_model_n_layer(model);
  info->n_head = llama_model_n_head(model);
  info->n_head_kv = llama_model_n_head_kv(model);
  info->n_embd = llama_model_n_embd(model);
  info->n_ctx_train = llama_model_n_ctx_train(model);
  info->n_params = llama_model_n_params(model);
  info->model_size = llama_model_size(model);

  return true;
}

int32_t llama_go_model_desc(void *model_handle, char *buf, size_t buf_size) {
  if (model_handle == nullptr || buf == nullptr || buf_size == 0) {
    return -1;
  }

  llama_model *model = llama_go_model_get_llama_model(model_handle);
  if (model == nullptr) {
    return -1;
  }

  return llama_model_desc(model, buf, buf_size);
}

bool llama_go_get_context_info(void *ctx_handle, llama_go_context_info *info) {
  if (ctx_handle == nullptr || info == nullptr) {
    llama_go_set_error("get_context_info: null handle or info");
    return false;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);

  info->n_ctx = llama_n_ctx(ctx);
  info->n_batch = llama_n_batch(ctx);
  info->n_ubatch = llama_n_ubatch(ctx);
  info->n_seq_max = llama_n_seq_max(ctx);
  info->n_threads = llama_n_threads(ctx);

  return true;
}

bool llama_go_get_perf_data(void *ctx_handle, llama_go_perf_data *data) {
  if (ctx_handle == nullptr || data == nullptr) {
    llama_go_set_error("get_perf_data: null handle or data");
    return false;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  struct llama_perf_context_data perf = llama_perf_context(ctx);

  data->t_start_ms = perf.t_start_ms;
  data->t_load_ms = perf.t_load_ms;
  data->t_p_eval_ms = perf.t_p_eval_ms;
  data->t_eval_ms = perf.t_eval_ms;
  data->n_p_eval = perf.n_p_eval;
  data->n_eval = perf.n_eval;

  return true;
}

void llama_go_perf_reset(void *ctx_handle) {
  if (ctx_handle == nullptr) {
    return;
  }

  llama_context *ctx = static_cast<llama_context *>(ctx_handle);
  llama_perf_context_reset(ctx);
}

} // extern "C"
