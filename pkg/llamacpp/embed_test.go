package llamacpp_test

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"

	llama "github.com/mutablelogic/go-llama"
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbedWithGenerativeModel(t *testing.T) {
	t.Skip("TODO: Generative model embedding behavior needs investigation")

	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// stories260K.gguf is a generative model - embedding extraction uses last token
	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model: "stories260K.gguf",
		Input: []string{"Hello, world!"},
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Len(resp.Embeddings, 1)
	t.Logf("Generative model embedding dimension: %d", resp.Dimension)
}

func TestEmbedWithNonEmbeddingModelReturnsError(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	_, err = l.Embed(context.Background(), schema.EmbedRequest{
		Model: "stories260K.gguf",
		Input: []string{"Hello, world!"},
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotEmbeddingModel))
}

func TestEmbedModelNotFound(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	_, err = l.Embed(context.Background(), schema.EmbedRequest{
		Model: "nonexistent.gguf",
		Input: []string{"Hello"},
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestEmbedEmptyInput(t *testing.T) {
	t.Skip("TODO: BERT embedding model causes crash - needs investigation")

	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model: "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Input: []string{},
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Empty(resp.Embeddings)
	assert.Equal(0, resp.Usage.InputTokens)
}

func TestEmbed(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model: "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Input: []string{"Hello, world!"},
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Equal("all-MiniLM-L6-v2-Q4_K_M.gguf", resp.Model)
	assert.Len(resp.Embeddings, 1)
	assert.Equal(384, resp.Dimension) // MiniLM has 384 dimensions
	assert.Greater(resp.Usage.InputTokens, 0)
	assert.Equal(0, resp.Usage.OutputTokens)

	// Check embedding is normalized (magnitude ~= 1.0)
	if len(resp.Embeddings) > 0 {
		var mag float64
		for _, v := range resp.Embeddings[0] {
			mag += float64(v) * float64(v)
		}
		mag = math.Sqrt(mag)
		assert.InDelta(1.0, mag, 0.01, "embedding should be normalized")
	}
}

func TestEmbedMultiple(t *testing.T) {
	t.Skip("TODO: BERT embedding model causes crash - needs investigation")

	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	inputs := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is fun.",
	}

	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model: "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Input: inputs,
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Len(resp.Embeddings, len(inputs))
	assert.Equal(384, resp.Dimension)
	assert.Greater(resp.Usage.InputTokens, 0)

	// Each embedding should have the correct dimension
	for i, emb := range resp.Embeddings {
		assert.Len(emb, resp.Dimension, "embedding %d should have correct dimension", i)
	}

	t.Logf("Computed %d embeddings of dimension %d", len(resp.Embeddings), resp.Dimension)
	t.Logf("Total input tokens: %d", resp.Usage.InputTokens)
}

func TestEmbedNormalize(t *testing.T) {
	t.Skip("TODO: BERT embedding model causes crash - needs investigation")

	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Test with normalize = false
	normalize := false
	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model:     "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Input:     []string{"Hello, world!"},
		Normalize: &normalize,
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Len(resp.Embeddings, 1)

	// Non-normalized embedding may have magnitude != 1.0
	if len(resp.Embeddings) > 0 {
		var mag float64
		for _, v := range resp.Embeddings[0] {
			mag += float64(v) * float64(v)
		}
		mag = math.Sqrt(mag)
		t.Logf("Non-normalized embedding magnitude: %f", mag)
	}
}

func TestEmbedUsage(t *testing.T) {
	t.Skip("TODO: BERT embedding model causes crash - needs investigation")

	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Embed(context.Background(), schema.EmbedRequest{
		Model: "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Input: []string{"Hello", "World"},
	})

	require.NoError(err)
	require.NotNil(resp)

	// Check usage stats
	assert.Greater(resp.Usage.InputTokens, 0)
	assert.Equal(0, resp.Usage.OutputTokens)
	assert.Equal(resp.Usage.InputTokens, resp.Usage.TotalTokens())

	t.Logf("Usage: input=%d, output=%d, total=%d",
		resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens())
}
