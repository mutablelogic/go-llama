package llamacpp

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

func TestLlama_PullModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Create temporary directory for models
	tempDir := t.TempDir()

	// Create Llama instance
	llama, err := New(tempDir)
	require.NoError(err)
	require.NotNil(llama)
	defer llama.Close()

	// Test pulling a small model from HuggingFace
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var progressUpdates int
	progressCallback := func(filename string, bytesReceived, totalBytes uint64) error {
		progressUpdates++
		t.Logf("Progress: %s - %d/%d bytes (%.1f%%)",
			filename, bytesReceived, totalBytes,
			float64(bytesReceived)*100.0/float64(totalBytes))
		return nil
	}

	model, err := llama.PullModel(ctx, schema.PullModelRequest{
		URL: "hf://ggml-org/models-moved@main/tinyllamas/stories260K.gguf",
	}, progressCallback)
	assert.NoError(err, "PullModel should succeed")
	assert.NotNil(model, "PullModel should return a model")
	assert.NotEmpty(model.Path, "Model should have a path")
	assert.NotEmpty(model.Name, "Model should have a name")
	assert.Equal("llama", model.Architecture, "Model architecture should be llama")
	assert.True(model.LoadedAt.IsZero(), "Model should not be loaded into memory")
	assert.Greater(progressUpdates, 0, "Progress callback should be called at least once")
}
