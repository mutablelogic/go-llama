package llamacpp_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	llama "github.com/mutablelogic/go-llama"
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	var callbackCalled bool
	err = l.WithModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	}, func(ctx context.Context, task *llamacpp.Task) error {
		callbackCalled = true

		// Task should have a valid model
		assert.NotNil(task.Model())
		assert.NotNil(task.CachedModel())

		// Context should be nil (WithModel doesn't create context)
		assert.Nil(task.Context())

		// CachedModel should have valid metadata
		assert.Equal("stories260K.gguf", task.CachedModel().Path)
		assert.Equal("llama", task.CachedModel().Architecture)
		assert.NotNil(task.CachedModel().Handle)

		return nil
	})

	require.NoError(err)
	assert.True(callbackCalled)
}

func TestWithModelError(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Test that callback error is propagated
	expectedErr := errors.New("callback error")
	err = l.WithModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	}, func(ctx context.Context, task *llamacpp.Task) error {
		return expectedErr
	})

	require.Error(err)
	require.Equal(expectedErr, err)
}

func TestWithModelNotFound(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	err = l.WithModel(context.Background(), schema.LoadModelRequest{
		Name: "nonexistent.gguf",
	}, func(ctx context.Context, task *llamacpp.Task) error {
		t.Fatal("callback should not be called")
		return nil
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestWithModelCaching(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	var firstModel, secondModel *schema.CachedModel

	// First call loads the model
	err = l.WithModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	}, func(ctx context.Context, task *llamacpp.Task) error {
		firstModel = task.CachedModel()
		return nil
	})
	require.NoError(err)

	// Second call should return the same cached model
	err = l.WithModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	}, func(ctx context.Context, task *llamacpp.Task) error {
		secondModel = task.CachedModel()
		return nil
	})
	require.NoError(err)

	// Should be the same pointer (cached)
	assert.Same(firstModel, secondModel)
}

func TestWithContext(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	var callbackCalled bool
	err = l.WithContext(context.Background(), schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: "stories260K.gguf",
		},
	}, func(ctx context.Context, task *llamacpp.Task) error {
		callbackCalled = true

		// Task should have both model and context
		assert.NotNil(task.Model())
		assert.NotNil(task.CachedModel())
		assert.NotNil(task.Context())

		return nil
	})

	require.NoError(err)
	assert.True(callbackCalled)
}

func TestWithContextParams(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	ctxSize := uint32(256)
	batchSize := uint32(64)
	threads := int32(2)

	err = l.WithContext(context.Background(), schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: "stories260K.gguf",
		},
		ContextSize: &ctxSize,
		BatchSize:   &batchSize,
		Threads:     &threads,
	}, func(ctx context.Context, task *llamacpp.Task) error {
		// Context was created successfully with custom params
		assert.NotNil(task.Context())
		return nil
	})

	require.NoError(err)
}

func TestWithContextError(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Test that callback error is propagated
	expectedErr := errors.New("context callback error")
	err = l.WithContext(context.Background(), schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: "stories260K.gguf",
		},
	}, func(ctx context.Context, task *llamacpp.Task) error {
		return expectedErr
	})

	require.Error(err)
	require.Equal(expectedErr, err)
}

func TestWithContextNotFound(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	err = l.WithContext(context.Background(), schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: "nonexistent.gguf",
		},
	}, func(ctx context.Context, task *llamacpp.Task) error {
		t.Fatal("callback should not be called")
		return nil
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestWithModelConcurrent(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Run multiple concurrent WithModel calls on the same model
	const numGoroutines = 10
	var wg sync.WaitGroup
	var successCount atomic.Int32

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := l.WithModel(context.Background(), schema.LoadModelRequest{
				Name: "stories260K.gguf",
			}, func(ctx context.Context, task *llamacpp.Task) error {
				// Lock for thread-safety as we would in real operations
				task.CachedModel().Lock()
				defer task.CachedModel().Unlock()

				// Access model data
				if task.Model() != nil && task.CachedModel() != nil {
					successCount.Add(1)
				}
				return nil
			})
			if err != nil {
				t.Errorf("WithModel failed: %v", err)
			}
		}()
	}

	wg.Wait()
	require.Equal(int32(numGoroutines), successCount.Load())
}
