package gguf_test

import (
	"path/filepath"
	"testing"

	"github.com/mutablelogic/go-llama/sys/gguf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testdataDir = "../../testdata"
)

func TestOpen_InvalidPath(t *testing.T) {
	assert := assert.New(t)

	ctx, err := gguf.Open("/nonexistent/path/model.gguf")
	assert.Error(err)
	assert.Nil(ctx)
	assert.ErrorIs(err, gguf.ErrOpenFailed)
}

func TestOpen_Stories(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	require.NotNil(ctx)
	defer ctx.Close()

	// Check we have some metadata
	assert.Greater(ctx.MetaCount(), 0)
	t.Logf("stories260K.gguf has %d metadata keys", ctx.MetaCount())

	// Check architecture
	arch := ctx.Architecture()
	assert.NotEmpty(arch)
	t.Logf("Architecture: %s", arch)
}

func TestOpen_MiniLM(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "all-MiniLM-L6-v2-Q4_K_M.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	require.NotNil(ctx)
	defer ctx.Close()

	// Check we have some metadata
	assert.Greater(ctx.MetaCount(), 0)
	t.Logf("all-MiniLM-L6-v2-Q4_K_M.gguf has %d metadata keys", ctx.MetaCount())

	// Check architecture (should be "bert" for MiniLM)
	arch := ctx.Architecture()
	assert.NotEmpty(arch)
	t.Logf("Architecture: %s", arch)
}

func TestMetaKey(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	defer ctx.Close()

	// Get all keys
	n := ctx.MetaCount()
	keys := make([]string, 0, n)
	for i := 0; i < n; i++ {
		key, err := ctx.MetaKey(i)
		require.NoError(err)
		assert.NotEmpty(key)
		keys = append(keys, key)
	}

	t.Logf("Metadata keys: %v", keys)

	// Test out of range
	_, err = ctx.MetaKey(-1)
	assert.ErrorIs(err, gguf.ErrIndexOutOfRange)

	_, err = ctx.MetaKey(n)
	assert.ErrorIs(err, gguf.ErrIndexOutOfRange)
}

func TestMetaValue(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	defer ctx.Close()

	// general.architecture should exist and be a string
	arch, err := ctx.MetaValue("general.architecture")
	require.NoError(err)
	assert.IsType("", arch)
	t.Logf("general.architecture = %v", arch)

	// Non-existent key should return error
	_, err = ctx.MetaValue("nonexistent.key")
	assert.ErrorIs(err, gguf.ErrKeyNotFound)
}

func TestAllMetadata(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	defer ctx.Close()

	meta, err := ctx.AllMetadata()
	require.NoError(err)
	assert.NotEmpty(meta)

	// Should have general.architecture
	_, hasArch := meta["general.architecture"]
	assert.True(hasArch)

	// Log some interesting metadata
	for key, value := range meta {
		if value != nil {
			t.Logf("%s = %v (%T)", key, value, value)
		}
	}
}

func TestAccessors(t *testing.T) {
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)
	defer ctx.Close()

	// Test convenience accessors - they should not panic even if values are missing
	t.Logf("Name: %q", ctx.Name())
	t.Logf("Architecture: %q", ctx.Architecture())
	t.Logf("Description: %q", ctx.Description())
	t.Logf("ChatTemplate: %q", ctx.ChatTemplate())
}

func TestClose_Idempotent(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	require.NoError(err)

	// Close should be safe to call multiple times
	assert.NoError(ctx.Close())
	assert.NoError(ctx.Close())

	// After close, MetaCount should return 0
	assert.Equal(0, ctx.MetaCount())
}

func TestNilContext(t *testing.T) {
	assert := assert.New(t)

	// Create a context and close it
	path := filepath.Join(testdataDir, "stories260K.gguf")
	ctx, err := gguf.Open(path)
	assert.NoError(err)
	ctx.Close()

	// Operations on closed context should handle gracefully
	assert.Equal(0, ctx.MetaCount())

	_, err = ctx.MetaKey(0)
	assert.ErrorIs(err, gguf.ErrInvalidContext)

	_, err = ctx.MetaValue("general.architecture")
	assert.ErrorIs(err, gguf.ErrInvalidContext)

	_, err = ctx.AllMetadata()
	assert.ErrorIs(err, gguf.ErrInvalidContext)
}
