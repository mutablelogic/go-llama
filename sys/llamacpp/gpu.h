#ifndef LLAMA_GO_GPU_H
#define LLAMA_GO_GPU_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU device information
typedef struct {
    int32_t device_id;
    char device_name[256];
    int64_t free_memory_bytes;
    int64_t total_memory_bytes;
} llama_go_gpu_info;

// Get the number of available GPU devices
// Returns 0 if no GPU backend is available
int32_t llama_go_gpu_count(void);

// Get information about a specific GPU device
// Returns true on success, false if device_id is invalid or GPU not available
bool llama_go_gpu_get_info(int32_t device_id, llama_go_gpu_info* info);

// Get the name of the GPU backend (e.g., "Metal", "CUDA", "Vulkan", "CPU")
const char* llama_go_gpu_backend_name(void);

// Check if a specific backend is available
bool llama_go_has_metal(void);
bool llama_go_has_cuda(void);
bool llama_go_has_vulkan(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_GPU_H
