package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testdataPath = "../../../testdata"

func TestNew(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Test with valid directory
	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	store, err := New(path)
	require.NoError(err)
	assert.NotNil(store)
	assert.Equal(path, store.Path())
}

func TestNew_InvalidPath(t *testing.T) {
	assert := assert.New(t)

	// Test with non-existent directory
	_, err := New("/nonexistent/path")
	assert.Error(err)
}

func TestNew_NotADirectory(t *testing.T) {
	assert := assert.New(t)

	// Test with a file instead of directory
	path, err := filepath.Abs(filepath.Join(testdataPath, "stories260K.gguf"))
	assert.NoError(err)

	_, err = New(path)
	assert.Error(err)
}

func TestListModels(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	store, err := New(path)
	require.NoError(err)

	// List models
	models, err := store.ListModels(context.Background())
	require.NoError(err)
	assert.GreaterOrEqual(len(models), 2) // At least stories260K.gguf and all-MiniLM-L6-v2-Q4_K_M.gguf

	// Check that models have expected fields populated
	for _, m := range models {
		assert.NotEmpty(m.Path)
		assert.NotEmpty(m.Architecture)
	}

	// Check we have both architectures
	archs := make(map[string]bool)
	for _, m := range models {
		archs[m.Architecture] = true
	}
	assert.True(archs["llama"])
	assert.True(archs["bert"])
}

func TestStore_PullModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Create a temporary directory for this test
	tempDir := t.TempDir()

	store, err := New(tempDir)
	require.NoError(err)

	// Test pulling the tiny model from Hugging Face
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	modelURL := "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories260K.gguf?download=true"

	progressCallback := func(filename string, bytes_received uint64, total_bytes uint64) {
		// Default progress reporting
		displayName := filename
		if displayName == "" {
			displayName = filepath.Base(modelURL)
		}
		fmt.Printf("Downloading %s: %d/%d bytes\n", displayName, bytes_received, total_bytes)
	}

	model, err := store.PullModel(ctx, modelURL, progressCallback)
	assert.NoError(err, "PullModel should succeed")
	assert.NotNil(model, "PullModel should return a model")

	// Verify the temporary directory is clean (no leftover temp files)
	entries, err := os.ReadDir(tempDir)
	require.NoError(err)

	for _, entry := range entries {
		assert.False(strings.HasSuffix(entry.Name(), ".tmp"), "No temporary files should remain: %s", entry.Name())
	}
}
