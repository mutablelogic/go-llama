package store

import (
	"context"
	"path/filepath"
	"testing"

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
	assert.Len(models, 2) // stories260K.gguf and all-MiniLM-L6-v2-Q4_K_M.gguf

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
