package llamacpp_test

import (
	"context"
	"path/filepath"
	"testing"

	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testdataPath = "../../testdata"

func TestLlamaNewAndClose(t *testing.T) {
	path, err := filepath.Abs(testdataPath)
	require.NoError(t, err)

	llama, err := llamacpp.New(path)
	require.NoError(t, err)
	require.NotNil(t, llama)

	assert.NoError(t, llama.Close())
}

func TestLlamaCloseNil(t *testing.T) {
	var llama *llamacpp.Llama
	require.NoError(t, llama.Close())
}

func TestLlamaGPUInfo(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	info := l.GPUInfo(context.Background())
	require.NotNil(info)

	// Backend should be one of the known values
	assert.Contains([]string{"Metal", "CUDA", "Vulkan", "CPU"}, info.Backend)

	// Devices slice should exist (may be empty on CPU-only systems)
	assert.NotNil(info.Devices)

	// If we have devices, check they have valid data
	for _, d := range info.Devices {
		assert.GreaterOrEqual(d.ID, int32(0))
		assert.NotEmpty(d.Name)
	}
}
