#ifndef LLAMA_GO_METADATA_H
#define LLAMA_GO_METADATA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get the number of metadata key-value pairs in the model
int32_t llama_go_model_meta_count(void* model);

// Get a metadata key by index
// Returns the key string, or NULL if index is out of range
// The returned string is valid until the model is freed
const char* llama_go_model_meta_key(void* model, int32_t index);

// Get a metadata value by key
// Returns the value string, or NULL if key not found
// The caller must free the returned string with llama_go_free_string
char* llama_go_model_meta_value(void* model, const char* key);

// Free a string returned by llama_go_model_meta_value
void llama_go_free_string(char* str);

// Get common model metadata
const char* llama_go_model_name(void* model);
const char* llama_go_model_arch(void* model);
const char* llama_go_model_description(void* model);

// Additional model dimensions (n_layer, n_embd, n_ctx_train are in model.h)
int32_t llama_go_model_n_head(void* model);
int32_t llama_go_model_n_head_kv(void* model);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_METADATA_H
