#ifndef GO_LLAMA_COMPLETION_H
#define GO_LLAMA_COMPLETION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Completion parameters
struct llama_go_completion_params {
  // Sampler parameters
  uint32_t seed;
  float temperature;
  int32_t top_k;
  float top_p;
  float min_p;
  float repeat_penalty;
  int32_t repeat_last_n;
  float frequency_penalty;
  float presence_penalty;

  // Generation parameters
  int32_t max_tokens;
  int32_t stop_words_count;
  const char **stop_words;

  // Options
  bool enable_prefix_caching;

  // Callback
  void *callback_handle;
};

// Completion result with generation info
struct llama_go_completion_result {
  char *text;         // Generated text (owned by this struct)
  bool stop_word_hit; // True if generation stopped due to a stop sequence
  int32_t index;      // Index of which stop word was hit (-1 if none)
};

// Default completion parameters
struct llama_go_completion_params llama_go_completion_default_params();

// Generate text from prompt (C++ implementation)
// Returns allocated result struct (caller must free with
// llama_go_completion_free_result) Returns NULL on error (check
// llama_go_last_error)
struct llama_go_completion_result *
llama_go_completion_generate(void *ctx_handle, void *model_handle,
                             const char *prompt,
                             const struct llama_go_completion_params *params);

// Free result from generation
void llama_go_completion_free_result(struct llama_go_completion_result *result);

#ifdef __cplusplus
}
#endif

#endif // GO_LLAMA_COMPLETION_H
