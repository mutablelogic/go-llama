package llamacpp

/*
#include "gpu.h"
*/
import "C"

// GPUInfo contains information about a GPU device
type GPUInfo struct {
	DeviceID         int32
	DeviceName       string
	FreeMemoryBytes  int64 // -1 if unknown
	TotalMemoryBytes int64 // -1 if unknown
}

// FreeMemoryMB returns free memory in megabytes, or -1 if unknown
func (g GPUInfo) FreeMemoryMB() int64 {
	if g.FreeMemoryBytes < 0 {
		return -1
	}
	return g.FreeMemoryBytes / (1024 * 1024)
}

// TotalMemoryMB returns total memory in megabytes, or -1 if unknown
func (g GPUInfo) TotalMemoryMB() int64 {
	if g.TotalMemoryBytes < 0 {
		return -1
	}
	return g.TotalMemoryBytes / (1024 * 1024)
}

// GPUCount returns the number of available GPU devices.
// Returns 0 if no GPU backend is available.
func GPUCount() int {
	return int(C.llama_go_gpu_count())
}

// GPUGetInfo returns information about a specific GPU device.
// Returns nil if the device_id is invalid or GPU is not available.
func GPUGetInfo(deviceID int) *GPUInfo {
	var cInfo C.llama_go_gpu_info

	if !C.llama_go_gpu_get_info(C.int32_t(deviceID), &cInfo) {
		return nil
	}

	return &GPUInfo{
		DeviceID:         int32(cInfo.device_id),
		DeviceName:       C.GoString(&cInfo.device_name[0]),
		FreeMemoryBytes:  int64(cInfo.free_memory_bytes),
		TotalMemoryBytes: int64(cInfo.total_memory_bytes),
	}
}

// GPUBackendName returns the name of the GPU backend.
// Returns "Metal", "CUDA", "Vulkan", or "CPU".
func GPUBackendName() string {
	return C.GoString(C.llama_go_gpu_backend_name())
}

// HasMetal returns true if Metal backend is available
func HasMetal() bool {
	return bool(C.llama_go_has_metal())
}

// HasCUDA returns true if CUDA backend is available
func HasCUDA() bool {
	return bool(C.llama_go_has_cuda())
}

// HasVulkan returns true if Vulkan backend is available
func HasVulkan() bool {
	return bool(C.llama_go_has_vulkan())
}

// GPUList returns information about all available GPU devices
func GPUList() []GPUInfo {
	count := GPUCount()
	if count == 0 {
		return nil
	}

	gpus := make([]GPUInfo, 0, count)
	for i := 0; i < count; i++ {
		if info := GPUGetInfo(i); info != nil {
			gpus = append(gpus, *info)
		}
	}
	return gpus
}
