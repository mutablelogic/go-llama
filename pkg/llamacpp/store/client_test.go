package store

import (
	"bytes"
	"context"
	"testing"
	"time"

	// Packages

	assert "github.com/stretchr/testify/assert"
	require "github.com/stretchr/testify/require"
)

func TestClient_PullModel(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Create client
	client, err := NewClient()
	require.NoError(err)
	require.NotNil(client)

	// Test pulling the tiny model from Hugging Face
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	modelURL := "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories260K.gguf?download=true"

	// Create a temporary buffer to capture the downloaded data
	var buf bytes.Buffer

	// Track progress through callback
	var lastFilename string
	var lastReceived, lastTotal uint64
	callback := func(filename string, received, total uint64) {
		lastFilename = filename
		lastReceived = received
		lastTotal = total
	}

	destPath, err := client.PullModel(ctx, &buf, modelURL, callback)
	assert.NoError(err, "PullModel should succeed in downloading the model")
	assert.NotEmpty(destPath, "PullModel should return a destination path")

	// Verify we got some data and it starts with GGUF magic
	assert.Greater(buf.Len(), 4, "Downloaded data should be larger than 4 bytes")
	assert.Equal("GGUF", string(buf.Bytes()[:4]), "Downloaded data should start with GGUF magic header")

	// Verify callback was called with progress
	assert.NotEmpty(lastFilename, "Callback should have received filename")
	assert.Greater(lastReceived, uint64(0), "Callback should have received bytes count")
	assert.Greater(lastTotal, uint64(0), "Callback should have received total size")
	assert.Equal(uint64(buf.Len()), lastReceived, "Final received count should exactly match buffer size")
}

func TestClient_PullModel_InvalidURL(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Create client
	client, err := NewClient()
	require.NoError(err)
	require.NotNil(client)

	// Test with invalid URL
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var buf bytes.Buffer
	_, err = client.PullModel(ctx, &buf, "invalid-url", nil)
	assert.Error(err)
}

func TestClient_PullModel_InvalidGGUF(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Create client pointing to a text file (not a GGUF file)
	client, err := NewClient()
	require.NoError(err)
	require.NotNil(client)

	// Test with a URL that returns non-GGUF content
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try to download the README.md from the same repo - this is not a GGUF file
	readmeURL := "https://huggingface.co/ggml-org/models-moved/resolve/main/README.md"

	var buf bytes.Buffer
	_, err = client.PullModel(ctx, &buf, readmeURL, nil)
	assert.Error(err, "PullModel should fail when downloading non-GGUF content")
	assert.Contains(err.Error(), "invalid GGUF file", "Error should mention invalid GGUF file")
}
