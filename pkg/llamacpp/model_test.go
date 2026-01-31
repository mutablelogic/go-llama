package llamacpp_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	llama "github.com/mutablelogic/go-llama"
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLlamaListModels(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	models, err := l.ListModels(context.Background())
	require.NoError(err)
	assert.Len(models, 3)

	// Check models are sorted by path
	assert.Equal("Qwen3-8B-Q8_0.gguf", models[0].Path)
	assert.Equal("all-MiniLM-L6-v2-Q4_K_M.gguf", models[1].Path)
	assert.Equal("stories260K.gguf", models[2].Path)

	// Check uncached models have zero timestamp and nil handle
	for _, m := range models {
		assert.True(m.LoadedAt.IsZero())
		assert.Nil(m.Handle)
	}
}

func TestLlamaGetModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Get by full path
	model, err := l.GetModel(context.Background(), "stories260K.gguf")
	require.NoError(err)
	require.NotNil(model)
	assert.Equal("llama", model.Architecture)
	assert.True(model.LoadedAt.IsZero())
	assert.Nil(model.Handle)

	// Get by filename (same as path in flat directory)
	model, err = l.GetModel(context.Background(), "all-MiniLM-L6-v2-Q4_K_M.gguf")
	require.NoError(err)
	require.NotNil(model)
	assert.Equal("bert", model.Architecture)

	// Non-existent model returns ErrNotFound
	_, err = l.GetModel(context.Background(), "nonexistent.gguf")
	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestLlamaLoadModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load a model
	beforeLoad := time.Now()
	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)
	require.NotNil(cached)

	// Check cached model has timestamp and handle
	assert.Equal("stories260K.gguf", cached.Path)
	assert.Equal("llama", cached.Architecture)
	assert.False(cached.LoadedAt.IsZero())
	assert.True(cached.LoadedAt.After(beforeLoad) || cached.LoadedAt.Equal(beforeLoad))
	assert.NotNil(cached.Handle)
}

func TestLlamaLoadModelTwice(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load a model
	cached1, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	// Load the same model again
	cached2, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	// Should return the same cached instance
	assert.Equal(cached1.LoadedAt, cached2.LoadedAt)
	assert.Equal(cached1.Handle, cached2.Handle)
}

func TestLlamaLoadModelNotFound(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Try to load non-existent model
	_, err = l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "nonexistent.gguf",
	})
	require.Error(err)
	assert.True(t, errors.Is(err, llama.ErrNotFound))
}

func TestLlamaGetModelAfterLoad(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load a model
	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	// Get the same model - should return cached version
	model, err := l.GetModel(context.Background(), "stories260K.gguf")
	require.NoError(err)
	assert.Equal(cached.LoadedAt, model.LoadedAt)
	assert.Equal(cached.Handle, model.Handle)
}

func TestLlamaListModelsAfterLoad(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load one model
	_, err = l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	// List all models
	models, err := l.ListModels(context.Background())
	require.NoError(err)
	assert.Len(models, 3)

	// Check one is loaded, one is not
	for _, m := range models {
		if m.Path == "stories260K.gguf" {
			assert.False(m.LoadedAt.IsZero())
			assert.NotNil(m.Handle)
		} else {
			assert.True(m.LoadedAt.IsZero())
			assert.Nil(m.Handle)
		}
	}
}

func TestLlamaUnloadModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load a model
	_, err = l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	// Unload the model
	unloaded, err := l.UnloadModel(context.Background(), "stories260K.gguf")
	require.NoError(err)
	assert.Equal("stories260K.gguf", unloaded.Path)
	assert.True(unloaded.LoadedAt.IsZero())
	assert.Nil(unloaded.Handle)

	// Get the model - should be uncached now
	model, err := l.GetModel(context.Background(), "stories260K.gguf")
	require.NoError(err)
	assert.True(model.LoadedAt.IsZero())
	assert.Nil(model.Handle)
}

func TestLlamaUnloadModelNotLoaded(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Unload a model that was never loaded
	unloaded, err := l.UnloadModel(context.Background(), "stories260K.gguf")
	require.NoError(err)
	assert.Equal("stories260K.gguf", unloaded.Path)
	assert.True(unloaded.LoadedAt.IsZero())
	assert.Nil(unloaded.Handle)
}

func TestLlamaUnloadModelNotFound(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Try to unload non-existent model
	_, err = l.UnloadModel(context.Background(), "nonexistent.gguf")
	require.Error(err)
	assert.True(t, errors.Is(err, llama.ErrNotFound))
}

func TestLlamaLoadModelWithParams(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Load with custom params
	layers := int32(0)
	gpu := int32(0)
	mmap := true
	mlock := false

	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name:   "stories260K.gguf",
		Layers: &layers,
		Gpu:    &gpu,
		Mmap:   &mmap,
		Mlock:  &mlock,
	})
	require.NoError(err)
	assert.NotNil(cached.Handle)
}
