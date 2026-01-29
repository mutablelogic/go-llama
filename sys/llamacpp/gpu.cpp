#include "gpu.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstring>

// Include backend-specific headers based on what's available
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

extern "C" {

int32_t llama_go_gpu_count(void) {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count();
#elif defined(GGML_USE_METAL)
    // Metal always has 1 device on macOS
    return 1;
#elif defined(GGML_USE_VULKAN)
    return ggml_backend_vk_get_device_count();
#else
    return 0;
#endif
}

bool llama_go_gpu_get_info(int32_t device_id, llama_go_gpu_info* info) {
    if (!info) {
        return false;
    }

    memset(info, 0, sizeof(llama_go_gpu_info));
    info->device_id = device_id;

#ifdef GGML_USE_CUDA
    int count = ggml_backend_cuda_get_device_count();
    if (device_id < 0 || device_id >= count) {
        return false;
    }

    ggml_backend_cuda_get_device_description(device_id, info->device_name, sizeof(info->device_name));

    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(device_id, &free_mem, &total_mem);
    info->free_memory_bytes = static_cast<int64_t>(free_mem);
    info->total_memory_bytes = static_cast<int64_t>(total_mem);

    return true;

#elif defined(GGML_USE_METAL)
    if (device_id != 0) {
        return false;  // Metal only has device 0
    }

    // Get Metal device name - use a generic name since ggml-metal doesn't expose device query
    strncpy(info->device_name, "Apple Metal GPU", sizeof(info->device_name) - 1);

    // Metal doesn't expose memory info through ggml, but we can try to get system info
    // For now, report -1 to indicate unknown
    info->free_memory_bytes = -1;
    info->total_memory_bytes = -1;

    return true;

#elif defined(GGML_USE_VULKAN)
    int count = ggml_backend_vk_get_device_count();
    if (device_id < 0 || device_id >= count) {
        return false;
    }

    ggml_backend_vk_get_device_description(device_id, info->device_name, sizeof(info->device_name));

    size_t free_mem, total_mem;
    ggml_backend_vk_get_device_memory(device_id, &free_mem, &total_mem);
    info->free_memory_bytes = static_cast<int64_t>(free_mem);
    info->total_memory_bytes = static_cast<int64_t>(total_mem);

    return true;

#else
    (void)device_id;
    return false;
#endif
}

const char* llama_go_gpu_backend_name(void) {
#ifdef GGML_USE_CUDA
    return "CUDA";
#elif defined(GGML_USE_METAL)
    return "Metal";
#elif defined(GGML_USE_VULKAN)
    return "Vulkan";
#else
    return "CPU";
#endif
}

bool llama_go_has_metal(void) {
#ifdef GGML_USE_METAL
    return true;
#else
    return false;
#endif
}

bool llama_go_has_cuda(void) {
#ifdef GGML_USE_CUDA
    return true;
#else
    return false;
#endif
}

bool llama_go_has_vulkan(void) {
#ifdef GGML_USE_VULKAN
    return true;
#else
    return false;
#endif
}

} // extern "C"
