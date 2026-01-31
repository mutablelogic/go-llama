#include "completion.h"
#include "batch.h"
#include "context.h"
#include "error.h"
#include "sampler.h"
#include "tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <llama.h>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

// Forward declaration of the Go callback
extern "C" bool goTokenCallback(void *handle, const char *token);

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMETERS

extern "C" llama_go_completion_params llama_go_completion_default_params() {
  llama_go_completion_params params{};

  // Default sampler params
  params.seed = 0;
  params.temperature = 0.8f;
  params.top_k = 40;
  params.top_p = 0.95f;
  params.min_p = 0.05f;
  params.repeat_penalty = 1.0f;
  params.repeat_last_n = 64;
  params.frequency_penalty = 0.0f;
  params.presence_penalty = 0.0f;

  params.max_tokens = 512;
  params.stop_words_count = 0;
  params.stop_words = nullptr;
  params.enable_prefix_caching = false;
  params.callback_handle = nullptr;

  return params;
}

///////////////////////////////////////////////////////////////////////////////
// GENERATION

extern "C" struct llama_go_completion_result *
llama_go_completion_generate(void *ctx_handle, void *model_handle,
                             const char *prompt,
                             const llama_go_completion_params *gen_params) {
  if (!ctx_handle || !model_handle || !prompt || !gen_params) {
    llama_go_set_error("Invalid arguments");
    return nullptr;
  }

  int stage = 0;
  int32_t last_token = -1;
  int32_t last_piece_len = 0;
  int64_t last_needed = 0;
  size_t last_generated_len = 0;
  bool stop_word_hit = false;
  int32_t stop_word_index = -1;

  try {
    llama_context *ctx = static_cast<llama_context *>(ctx_handle);

    stage = 1;
    // Tokenize prompt using wrapper function
    std::vector<int32_t> prompt_tokens;
    prompt_tokens.resize(strlen(prompt) + 16); // Initial estimate

    int32_t n_prompt =
        llama_go_tokenize(model_handle, prompt,
                          -1, // null-terminated
                          prompt_tokens.data(), prompt_tokens.size(),
                          true, // add_special
                          false // parse_special
        );

    stage = 2;
    if (n_prompt < 0) {
      // Need larger buffer
      if (n_prompt == std::numeric_limits<int32_t>::min()) {
        llama_go_set_error("Tokenize failed: invalid required size");
        return nullptr;
      }
      const int64_t needed = -static_cast<int64_t>(n_prompt);
      if (needed <= 0 || needed > std::numeric_limits<int32_t>::max()) {
        llama_go_set_error("Tokenize failed: required size too large");
        return nullptr;
      }
      prompt_tokens.resize(static_cast<size_t>(needed));
      n_prompt =
          llama_go_tokenize(model_handle, prompt, -1, prompt_tokens.data(),
                            prompt_tokens.size(), true, false);
    }

    stage = 3;
    if (n_prompt <= 0) {
      llama_go_set_error("Failed to tokenize prompt");
      return nullptr;
    }

    prompt_tokens.resize(n_prompt);

    // Check context and batch sizes
    const int32_t n_ctx = llama_n_ctx(ctx);
    const uint32_t n_batch = llama_go_context_n_batch(ctx_handle);
    if (n_batch == 0) {
      llama_go_set_error("Invalid batch size (n_batch=0)");
      return nullptr;
    }
    if (n_prompt > (int32_t)n_batch) {
      llama_go_set_error("Prompt exceeds batch size (n_prompt > n_batch)");
      return nullptr;
    }
    if (n_prompt + gen_params->max_tokens > n_ctx) {
      llama_go_set_error("Prompt + max_tokens exceeds context size");
      return nullptr;
    }

    stage = 4;
    // Create sampler with params
    llama_go_sampler_params sampler_params{};
    sampler_params.seed = gen_params->seed;
    sampler_params.temperature = gen_params->temperature;
    sampler_params.top_k = gen_params->top_k;
    sampler_params.top_p = gen_params->top_p;
    sampler_params.min_p = gen_params->min_p;
    sampler_params.repeat_penalty = gen_params->repeat_penalty;
    sampler_params.repeat_last_n = gen_params->repeat_last_n;
    sampler_params.frequency_penalty = gen_params->frequency_penalty;
    sampler_params.presence_penalty = gen_params->presence_penalty;

    void *sampler = llama_go_sampler_new(model_handle, sampler_params);
    if (!sampler) {
      llama_go_set_error("Failed to create sampler");
      return nullptr;
    }

    stage = 5;
    // Create batch using wrapper
    llama_go_batch *batch = llama_go_batch_init((int32_t)n_batch, 1);
    if (!batch) {
      llama_go_sampler_free(sampler);
      llama_go_set_error("Failed to create batch");
      return nullptr;
    }

    stage = 6;
    // Decode prompt
    llama_go_batch_clear(batch);
    for (int32_t i = 0; i < n_prompt; i++) {
      llama_go_batch_add(batch, prompt_tokens[i], i, 0, i == n_prompt - 1);
    }

    if (llama_go_batch_decode(ctx_handle, batch) != 0) {
      llama_go_batch_free(batch);
      llama_go_sampler_free(sampler);
      llama_go_set_error("Failed to decode prompt");
      return nullptr;
    }

    stage = 7;
    // Generation loop
    std::string generated_text;
    int32_t n_past = n_prompt;
    const int32_t max_tokens = gen_params->max_tokens;
    last_token = -1;
    last_piece_len = 0;
    last_needed = 0;

    for (int32_t i = 0; i < max_tokens; i++) {
      stage = 8;
      // Sample next token
      int32_t new_token = llama_go_sampler_sample(sampler, ctx_handle, -1);
      last_token = new_token;
      if (new_token < 0) {
        llama_go_batch_free(batch);
        llama_go_sampler_free(sampler);
        llama_go_set_error("Failed to sample token");
        return nullptr;
      }

      stage = 9;
      // Accept token for repetition penalty
      llama_go_sampler_accept(sampler, new_token);

      stage = 10;
      // Check for end of generation
      const llama_model *llama_model_ptr = llama_get_model(ctx);
      const llama_vocab *vocab = llama_model_get_vocab(llama_model_ptr);
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }

      stage = 11;
      // Detokenize with safe buffer handling
      const char *piece_ptr = nullptr;
      std::string piece_str;
      char piece[256];
      int32_t piece_len = llama_go_token_to_piece(model_handle, new_token,
                                                  piece, sizeof(piece) - 1,
                                                  false // special
      );
      last_piece_len = piece_len;

      if (piece_len < 0) {
        if (piece_len == std::numeric_limits<int32_t>::min()) {
          llama_go_batch_free(batch);
          llama_go_sampler_free(sampler);
          llama_go_set_error("Failed to detokenize token: invalid size");
          return nullptr;
        }
        const int64_t needed64 = -static_cast<int64_t>(piece_len) + 1;
        last_needed = needed64;
        if (needed64 <= 0 || needed64 > std::numeric_limits<size_t>::max()) {
          llama_go_batch_free(batch);
          llama_go_sampler_free(sampler);
          llama_go_set_error("Failed to detokenize token: size too large");
          return nullptr;
        }
        piece_str.resize(static_cast<size_t>(needed64));
        int32_t ret =
            llama_go_token_to_piece(model_handle, new_token, &piece_str[0],
                                    piece_str.size() - 1, false);
        if (ret < 0) {
          llama_go_batch_free(batch);
          llama_go_sampler_free(sampler);
          llama_go_set_error("Failed to detokenize token");
          return nullptr;
        }
        piece_str.resize(ret);
        piece_ptr = piece_str.c_str();
      } else if (piece_len >= (int32_t)sizeof(piece)) {
        last_needed = static_cast<int64_t>(piece_len) + 1;
        piece_str.resize(piece_len + 1);
        int32_t ret =
            llama_go_token_to_piece(model_handle, new_token, &piece_str[0],
                                    piece_str.size() - 1, false);
        if (ret < 0) {
          llama_go_batch_free(batch);
          llama_go_sampler_free(sampler);
          llama_go_set_error("Failed to detokenize token");
          return nullptr;
        }
        piece_str.resize(ret);
        piece_ptr = piece_str.c_str();
      } else if (piece_len > 0) {
        piece[piece_len] = '\0';
        piece_ptr = piece;
      }

      stage = 12;
      if (piece_ptr) {
        generated_text += piece_ptr;
        last_generated_len = generated_text.size();

        // Callback if provided
        if (gen_params->callback_handle) {
          if (!goTokenCallback(gen_params->callback_handle, piece_ptr)) {
            // Callback requested stop
            break;
          }
        }

        // Check stop words
        if (gen_params->stop_words) {
          for (int j = 0; gen_params->stop_words[j] != nullptr; j++) {
            const char *stop_word = gen_params->stop_words[j];
            size_t stop_len = strlen(stop_word);
            if (generated_text.length() >= stop_len &&
                generated_text.compare(generated_text.length() - stop_len,
                                       stop_len, stop_word) == 0) {
              // Found stop word - remove it and stop
              generated_text.resize(generated_text.length() - stop_len);
              stop_word_hit = true;
              stop_word_index = j;
              llama_go_batch_free(batch);
              llama_go_sampler_free(sampler);

              struct llama_go_completion_result *result =
                  (struct llama_go_completion_result *)malloc(
                      sizeof(struct llama_go_completion_result));
              if (!result) {
                llama_go_set_error("Failed to allocate result");
                return nullptr;
              }
              result->text = (char *)malloc(generated_text.length() + 1);
              if (!result->text) {
                llama_go_set_error("Failed to allocate result text");
                free(result);
                return nullptr;
              }
              strcpy(result->text, generated_text.c_str());
              result->stop_word_hit = true;
              result->index = stop_word_index;
              return result;
            }
          }
        }
      }

      stage = 13;
      // Prepare next iteration - decode new token
      llama_go_batch_clear(batch);
      llama_go_batch_add(batch, new_token, n_past, 0, true);
      n_past++;

      if (llama_go_batch_decode(ctx_handle, batch) != 0) {
        break;
      }
    }

    stage = 14;
    // Clean up
    llama_go_batch_free(batch);
    llama_go_sampler_free(sampler);

    stage = 15;
    // Allocate result
    struct llama_go_completion_result *result =
        (struct llama_go_completion_result *)malloc(
            sizeof(struct llama_go_completion_result));
    if (!result) {
      llama_go_set_error("Failed to allocate result");
      return nullptr;
    }
    result->text = (char *)malloc(generated_text.length() + 1);
    if (!result->text) {
      llama_go_set_error("Failed to allocate result text");
      free(result);
      return nullptr;
    }
    strcpy(result->text, generated_text.c_str());
    result->stop_word_hit = stop_word_hit;
    result->index = stop_word_index;
    return result;

  } catch (const std::exception &e) {
    std::string msg = "completion exception: ";
    msg += typeid(e).name();
    msg += ": ";
    msg += e.what();
    msg += " (stage=";
    msg += std::to_string(stage);
    msg += ", token=";
    msg += std::to_string(last_token);
    msg += ", piece_len=";
    msg += std::to_string(last_piece_len);
    msg += ", needed=";
    msg += std::to_string(last_needed);
    msg += ", generated_len=";
    msg += std::to_string(last_generated_len);
    msg += ")";
    llama_go_set_error(msg);
    return nullptr;
  } catch (...) {
    llama_go_set_error("completion exception: unknown");
    return nullptr;
  }
}

extern "C" void
llama_go_completion_free_result(struct llama_go_completion_result *result) {
  if (result) {
    if (result->text) {
      free(result->text);
    }
    free(result);
  }
}
