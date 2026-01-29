#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

///////////////////////////////////////////////////////////////////////////////
// MODEL CACHE
//
// Models are expensive to load, so we cache them by path.
// Multiple contexts can share the same model.

// Model loading parameters
typedef struct {
    int32_t n_gpu_layers;    // Number of layers to offload to GPU (-1 = all)
    int32_t main_gpu;        // Main GPU device index
    bool use_mmap;           // Use memory mapping (default: true)
    bool use_mlock;          // Lock model in memory (default: false)
} llama_go_model_params;

// Get default model parameters
llama_go_model_params llama_go_model_default_params(void);

// Load or get cached model
// Returns opaque handle, or NULL on error (check llama_go_last_error)
// The model is reference-counted; call llama_go_model_release when done
void* llama_go_model_load(const char* path, llama_go_model_params params);

// Release a model reference
// Model is freed when reference count reaches zero
void llama_go_model_release(void* model);

// Get model info
int32_t llama_go_model_n_ctx_train(void* model);   // Training context length
int32_t llama_go_model_n_embd(void* model);        // Embedding dimension
int32_t llama_go_model_n_layer(void* model);       // Number of layers
int32_t llama_go_model_n_vocab(void* model);       // Vocabulary size

// Get model metadata
// Returns NULL if key not found
const char* llama_go_model_meta_val_str(void* model, const char* key);

// Get chat template from model metadata
// template_name can be NULL for default, or a specific name like "tool_use"
// Returns NULL if no template available
const char* llama_go_model_chat_template(void* model, const char* template_name);

// Get the underlying llama_model pointer from a model handle
// This is used internally by context creation
struct llama_model* llama_go_model_get_llama_model(void* model);

// Cache management
int32_t llama_go_model_cache_count(void);  // Number of cached models
void llama_go_model_cache_clear(void);     // Clear all cached models (releases refs)

#ifdef __cplusplus
}
#endif
