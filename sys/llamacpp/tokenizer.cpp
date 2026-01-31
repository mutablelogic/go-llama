#include "tokenizer.h"
#include "error.h"
#include "model.h"
#include <cstring>
#include <llama.h>
#include <string>
#include <vector>

// Get the underlying llama_model from our wrapper
extern "C" struct llama_model *llama_go_model_get_llama_model(void *model);

extern "C" {

int32_t llama_go_tokenize(void *model, const char *text, int32_t text_len,
                          int32_t *tokens, int32_t tokens_capacity,
                          bool add_special, bool parse_special) {
  if (!model || !text) {
    llama_go_set_error("invalid model or text");
    return -1;
  }

  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel) {
    llama_go_set_error("failed to get llama model");
    return -1;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab) {
    llama_go_set_error("failed to get vocabulary");
    return -1;
  }

  // If tokens is NULL, calculate required size
  if (!tokens) {
    // Estimate: worst case is 1 token per byte + special tokens
    return text_len + 2;
  }

  int32_t effective_len = text_len;
  if (effective_len < 0) {
    effective_len = (int32_t)strlen(text);
  }

  int32_t n_tokens =
      llama_tokenize(vocab, text, effective_len, tokens, tokens_capacity,
                     add_special, parse_special);

  if (n_tokens < 0) {
    // Buffer too small, return required size (negated)
    return n_tokens;
  }

  return n_tokens;
}

int32_t llama_go_token_to_piece(void *model, int32_t token, char *buf,
                                int32_t buf_size, bool special) {
  if (!model) {
    llama_go_set_error("invalid model");
    return -1;
  }

  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel) {
    llama_go_set_error("failed to get llama model");
    return -1;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab) {
    llama_go_set_error("failed to get vocabulary");
    return -1;
  }

  return llama_token_to_piece(vocab, token, buf, buf_size, 0, special);
}

int32_t llama_go_detokenize(void *model, const int32_t *tokens,
                            int32_t n_tokens, char *text, int32_t text_capacity,
                            bool remove_special, bool unparse_special) {
  if (!model || !tokens) {
    llama_go_set_error("invalid model or tokens");
    return -1;
  }

  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel) {
    llama_go_set_error("failed to get llama model");
    return -1;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab) {
    llama_go_set_error("failed to get vocabulary");
    return -1;
  }

  return llama_detokenize(vocab, tokens, n_tokens, text, text_capacity,
                          remove_special, unparse_special);
}

int32_t llama_go_token_bos(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_bos(vocab);
}

int32_t llama_go_token_eos(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_eos(vocab);
}

int32_t llama_go_token_eot(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_eot(vocab);
}

int32_t llama_go_token_nl(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_nl(vocab);
}

int32_t llama_go_token_pad(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_pad(vocab);
}

bool llama_go_token_is_eog(void *model, int32_t token) {
  if (!model)
    return false;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return false;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return false;
  return llama_vocab_is_eog(vocab, token);
}

bool llama_go_token_is_control(void *model, int32_t token) {
  if (!model)
    return false;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return false;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return false;
  return llama_vocab_is_control(vocab, token);
}

int32_t llama_go_n_vocab(void *model) {
  if (!model)
    return -1;
  struct llama_model *lmodel = llama_go_model_get_llama_model(model);
  if (!lmodel)
    return -1;
  const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
  if (!vocab)
    return -1;
  return llama_vocab_n_tokens(vocab);
}

} // extern "C"
