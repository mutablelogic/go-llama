package main

import (
	"fmt"
	"os"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
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
	// Initialize llamacpp, with a temporary model path
	llama, err := llamacpp.New(os.TempDir())
	if err != nil {
		return err
	}
	defer llama.Close()

	// Print GPU info
	fmt.Println(llama.GPUInfo(globals.ctx))
	return nil
}
