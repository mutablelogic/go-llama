#include "model.h"
#include "error.h"
#include "init.h"
#include "llama.h"

#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////
// MODEL CACHE

// Cached model entry with reference counting
struct CachedModel {
    llama_model* model;
    std::string path;
    int32_t ref_count;
    
    CachedModel(llama_model* m, const std::string& p)
        : model(m), path(p), ref_count(1) {}
    
    ~CachedModel() {
        if (model) {
            llama_model_free(model);
        }
    }
};

// Global model cache
static std::unordered_map<std::string, std::unique_ptr<CachedModel>> g_model_cache;
static std::mutex g_cache_mutex;

///////////////////////////////////////////////////////////////////////////////
// MODEL PARAMETERS

extern "C" llama_go_model_params llama_go_model_default_params(void) {
    llama_go_model_params params;
    params.n_gpu_layers = -1;   // All layers on GPU
    params.main_gpu = 0;
    params.use_mmap = true;
    params.use_mlock = false;
    return params;
}

///////////////////////////////////////////////////////////////////////////////
// MODEL LOADING

extern "C" void* llama_go_model_load(const char* path, llama_go_model_params params) {
    if (!path) {
        llama_go_set_error("Model path cannot be null");
        return nullptr;
    }
    
    std::string path_str(path);
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    
    // Check if model is already cached
    auto it = g_model_cache.find(path_str);
    if (it != g_model_cache.end()) {
        it->second->ref_count++;
        return it->second.get();
    }
    
    // Initialize backend if needed
    llama_go_init();
    
    // Convert parameters
    llama_model_params model_params = llama_model_default_params();
    
    if (params.n_gpu_layers >= 0) {
        model_params.n_gpu_layers = params.n_gpu_layers;
    } else {
        model_params.n_gpu_layers = 999; // All layers
    }
    
    model_params.main_gpu = params.main_gpu;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;
    
    // Load model
    llama_model* model = llama_model_load_from_file(path, model_params);
    if (!model) {
        llama_go_set_error("Failed to load model: " + path_str);
        return nullptr;
    }
    
    // Create and cache using constructor
    auto cached = std::make_unique<CachedModel>(model, path_str);
    
    void* result = cached.get();
    g_model_cache[path_str] = std::move(cached);
    
    return result;
}

///////////////////////////////////////////////////////////////////////////////
// MODEL RELEASE

extern "C" void llama_go_model_release(void* handle) {
    if (!handle) return;
    
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    CachedModel* cached = static_cast<CachedModel*>(handle);
    cached->ref_count--;
    
    if (cached->ref_count <= 0) {
        g_model_cache.erase(cached->path);
    }
}

///////////////////////////////////////////////////////////////////////////////
// MODEL INFO

extern "C" int32_t llama_go_model_n_ctx_train(void* handle) {
    if (!handle) return 0;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    return llama_model_n_ctx_train(cached->model);
}

extern "C" int32_t llama_go_model_n_embd(void* handle) {
    if (!handle) return 0;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    return llama_model_n_embd(cached->model);
}

extern "C" int32_t llama_go_model_n_layer(void* handle) {
    if (!handle) return 0;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    return llama_model_n_layer(cached->model);
}

extern "C" int32_t llama_go_model_n_vocab(void* handle) {
    if (!handle) return 0;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    const llama_vocab* vocab = llama_model_get_vocab(cached->model);
    return llama_vocab_n_tokens(vocab);
}

///////////////////////////////////////////////////////////////////////////////
// MODEL METADATA

extern "C" const char* llama_go_model_meta_val_str(void* handle, const char* key) {
    if (!handle || !key) return nullptr;
    
    CachedModel* cached = static_cast<CachedModel*>(handle);
    
    // Buffer for metadata value
    static thread_local char buffer[4096];
    
    int32_t len = llama_model_meta_val_str(cached->model, key, buffer, sizeof(buffer));
    if (len < 0) {
        return nullptr;
    }
    
    return buffer;
}

extern "C" const char* llama_go_model_chat_template(void* handle, const char* template_name) {
    if (!handle) return nullptr;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    return llama_model_chat_template(cached->model, template_name);
}

extern "C" llama_model* llama_go_model_get_llama_model(void* handle) {
    if (!handle) return nullptr;
    CachedModel* cached = static_cast<CachedModel*>(handle);
    return cached->model;
}

///////////////////////////////////////////////////////////////////////////////
// CACHE MANAGEMENT

extern "C" int32_t llama_go_model_cache_count(void) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return static_cast<int32_t>(g_model_cache.size());
}

extern "C" void llama_go_model_cache_clear(void) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_model_cache.clear();
}
