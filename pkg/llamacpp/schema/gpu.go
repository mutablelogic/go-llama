package schema

// GPUDevice represents information about a single GPU device
type GPUDevice struct {
	ID               int32  `json:"id"`
	Name             string `json:"name"`
	FreeMemoryBytes  int64  `json:"free_memory_bytes"`  // -1 if unknown
	TotalMemoryBytes int64  `json:"total_memory_bytes"` // -1 if unknown
}

// GPUInfo represents the GPU/accelerator configuration
type GPUInfo struct {
	Backend string      `json:"backend"` // "Metal", "CUDA", "Vulkan", "CPU"
	Devices []GPUDevice `json:"devices"`
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (d GPUDevice) String() string {
	return stringify(d)
}

func (i GPUInfo) String() string {
	return stringify(i)
}
