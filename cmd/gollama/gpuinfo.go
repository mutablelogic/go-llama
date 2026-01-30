package main

import (
	"encoding/json"
	"fmt"
	"os"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type GpuInfoCmd struct{}

type GPUInfoResponse struct {
	Backend string    `json:"backend"`
	Devices []GPUInfo `json:"devices"`
}

type MemoryInfo struct {
	Bytes     int64 `json:"bytes"`
	Megabytes int64 `json:"megabytes"`
}

type GPUInfo struct {
	DeviceID    int32       `json:"device_id"`
	DeviceName  string      `json:"device_name"`
	TotalMemory *MemoryInfo `json:"total_memory,omitempty"`
	FreeMemory  *MemoryInfo `json:"free_memory,omitempty"`
}

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func (cmd *GpuInfoCmd) Run(globals *Globals) error {
	// Initialize llamacpp
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Create response structure
	response := GPUInfoResponse{
		Backend: llamacpp.GPUBackendName(),
	}

	// Get GPU device information
	gpus := llamacpp.GPUList()
	response.Devices = make([]GPUInfo, 0, len(gpus))

	for _, gpu := range gpus {
		deviceInfo := GPUInfo{
			DeviceID:   gpu.DeviceID,
			DeviceName: gpu.DeviceName,
		}

		// Only include memory info if available and non-zero
		if gpu.TotalMemoryBytes > 0 {
			deviceInfo.TotalMemory = &MemoryInfo{
				Bytes:     gpu.TotalMemoryBytes,
				Megabytes: gpu.TotalMemoryMB(),
			}
		}

		if gpu.FreeMemoryBytes > 0 {
			deviceInfo.FreeMemory = &MemoryInfo{
				Bytes:     gpu.FreeMemoryBytes,
				Megabytes: gpu.FreeMemoryMB(),
			}
		}

		response.Devices = append(response.Devices, deviceInfo)
	}

	// Output as JSON
	data, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal GPU info: %w", err)
	}

	fmt.Fprintln(os.Stdout, string(data))
	return nil
}
