#include "metadata.h"
#include "model.h"
#include "error.h"
#include "llama.h"
#include <cstring>
#include <cstdlib>
#include <string>

extern "C" {

int32_t llama_go_model_meta_count(void* model) {
    if (!model) {
        return 0;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return 0;
    }
    return llama_model_meta_count(m);
}

const char* llama_go_model_meta_key(void* model, int32_t index) {
    if (!model) {
        return nullptr;
    }

    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return nullptr;
    }
    
    int32_t count = llama_model_meta_count(m);
    if (index < 0 || index >= count) {
        return nullptr;
    }

    // Buffer for key - metadata keys are typically short
    static thread_local char key_buffer[256];
    int32_t len = llama_model_meta_key_by_index(m, index, key_buffer, sizeof(key_buffer));
    if (len < 0) {
        return nullptr;
    }

    return key_buffer;
}

char* llama_go_model_meta_value(void* model, const char* key) {
    if (!model || !key) {
        return nullptr;
    }

    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return nullptr;
    }

    // First call to get required buffer size
    char small_buf[1];
    int32_t len = llama_model_meta_val_str(m, key, small_buf, 0);
    if (len < 0) {
        // Key not found
        return nullptr;
    }

    // Allocate buffer for value (+1 for null terminator)
    char* value = static_cast<char*>(malloc(len + 1));
    if (!value) {
        llama_go_set_error("failed to allocate memory for metadata value");
        return nullptr;
    }

    // Get the actual value
    llama_model_meta_val_str(m, key, value, len + 1);
    return value;
}

void llama_go_free_string(char* str) {
    if (str) {
        free(str);
    }
}

const char* llama_go_model_name(void* model) {
    if (!model) {
        return nullptr;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return nullptr;
    }
    static thread_local char name_buffer[256];
    int32_t len = llama_model_meta_val_str(m, "general.name", name_buffer, sizeof(name_buffer));
    if (len < 0) {
        return nullptr;
    }
    return name_buffer;
}

const char* llama_go_model_arch(void* model) {
    if (!model) {
        return nullptr;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return nullptr;
    }
    static thread_local char arch_buffer[128];
    int32_t len = llama_model_meta_val_str(m, "general.architecture", arch_buffer, sizeof(arch_buffer));
    if (len < 0) {
        return nullptr;
    }
    return arch_buffer;
}

const char* llama_go_model_description(void* model) {
    if (!model) {
        return nullptr;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return nullptr;
    }
    static thread_local char desc_buffer[512];
    
    // Try description first
    int32_t len = llama_model_meta_val_str(m, "general.description", desc_buffer, sizeof(desc_buffer));
    if (len < 0) {
        // Fall back to quantization info
        len = llama_model_meta_val_str(m, "general.quantization", desc_buffer, sizeof(desc_buffer));
    }
    if (len < 0) {
        return nullptr;
    }
    return desc_buffer;
}

// n_head and n_head_kv are defined here since they're not in model.cpp
int32_t llama_go_model_n_head(void* model) {
    if (!model) {
        return 0;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return 0;
    }
    return llama_model_n_head(m);
}

int32_t llama_go_model_n_head_kv(void* model) {
    if (!model) {
        return 0;
    }
    llama_model* m = llama_go_model_get_llama_model(model);
    if (!m) {
        return 0;
    }
    return llama_model_n_head_kv(m);
}

} // extern "C"
