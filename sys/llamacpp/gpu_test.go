package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestGPUBackendName(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	backend := llamacpp.GPUBackendName()
	t.Logf("GPU Backend: %s", backend)

	// Should be one of the known backends
	switch backend {
	case "Metal", "CUDA", "Vulkan", "CPU":
		// OK
	default:
		t.Errorf("unexpected backend name: %s", backend)
	}
}

func TestGPUCount(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	count := llamacpp.GPUCount()
	t.Logf("GPU Count: %d", count)

	// On macOS with Metal, should have at least 1
	if llamacpp.HasMetal() && count < 1 {
		t.Error("expected at least 1 GPU with Metal backend")
	}
}

func TestGPUGetInfo(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	count := llamacpp.GPUCount()
	if count == 0 {
		t.Skip("no GPU devices available")
	}

	info := llamacpp.GPUGetInfo(0)
	if info == nil {
		t.Fatal("failed to get GPU info for device 0")
	}

	t.Logf("GPU 0: %s", info.DeviceName)
	t.Logf("  Device ID: %d", info.DeviceID)
	if info.TotalMemoryBytes > 0 {
		t.Logf("  Total Memory: %d MB", info.TotalMemoryMB())
		t.Logf("  Free Memory: %d MB", info.FreeMemoryMB())
	} else {
		t.Log("  Memory info: not available")
	}
}

func TestGPUGetInfoInvalid(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Invalid device ID should return nil
	info := llamacpp.GPUGetInfo(999)
	if info != nil {
		t.Error("expected nil for invalid device ID")
	}

	info = llamacpp.GPUGetInfo(-1)
	if info != nil {
		t.Error("expected nil for negative device ID")
	}
}

func TestGPUList(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	gpus := llamacpp.GPUList()
	t.Logf("Found %d GPU(s)", len(gpus))

	for _, gpu := range gpus {
		t.Logf("  [%d] %s", gpu.DeviceID, gpu.DeviceName)
	}
}

func TestHasBackends(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	t.Logf("Has Metal: %v", llamacpp.HasMetal())
	t.Logf("Has CUDA: %v", llamacpp.HasCUDA())
	t.Logf("Has Vulkan: %v", llamacpp.HasVulkan())

	// At least one should be true on most systems, or all false for CPU-only
	backend := llamacpp.GPUBackendName()
	switch backend {
	case "Metal":
		if !llamacpp.HasMetal() {
			t.Error("backend is Metal but HasMetal() is false")
		}
	case "CUDA":
		if !llamacpp.HasCUDA() {
			t.Error("backend is CUDA but HasCUDA() is false")
		}
	case "Vulkan":
		if !llamacpp.HasVulkan() {
			t.Error("backend is Vulkan but HasVulkan() is false")
		}
	case "CPU":
		if llamacpp.HasMetal() || llamacpp.HasCUDA() || llamacpp.HasVulkan() {
			t.Error("backend is CPU but a GPU backend is reported available")
		}
	}
}
