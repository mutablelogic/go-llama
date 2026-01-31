//go:build !client

package llamacpp

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const chatTestdataPath = "../../testdata"

func TestBuildChatPromptWithSystemMessage(t *testing.T) {
	require := require.New(t)
	assert := assert.New(t)

	path, err := filepath.Abs(chatTestdataPath)
	require.NoError(err)

	l, err := New(path)
	require.NoError(err)
	defer l.Close()

	// Load model to test prompt building
	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)
	require.NotNil(cached)
	require.NotNil(cached.Handle)

	model := cached.Handle

	if !model.HasChatTemplate() {
		t.Skip("Model has no chat template")
	}

	req := schema.ChatRequest{
		CompletionRequest: schema.CompletionRequest{
			Model: "stories260K.gguf",
		},
		Messages: []schema.ChatMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
		},
	}

	prompt, err := buildChatPrompt(model, req)
	require.NoError(err)
	assert.NotEmpty(prompt)
	assert.Contains(prompt, "Hello")
}

func TestBuildChatPromptWithoutTemplate(t *testing.T) {
	require := require.New(t)

	// Test with nil model
	req := schema.ChatRequest{
		CompletionRequest: schema.CompletionRequest{
			Model: "test",
		},
		Messages: []schema.ChatMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
		},
	}

	_, err := buildChatPrompt(nil, req)
	require.Error(err)
	assert.New(t).Contains(err.Error(), "model is required")
}

func TestBuildChatPromptEmpty(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(chatTestdataPath)
	require.NoError(err)

	l, err := New(path)
	require.NoError(err)
	defer l.Close()

	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	model := cached.Handle

	req := schema.ChatRequest{
		CompletionRequest: schema.CompletionRequest{
			Model: "stories260K.gguf",
		},
		Messages: []schema.ChatMessage{},
	}

	_, chatErr := buildChatPrompt(model, req)
	require.Error(chatErr)
	assert.New(t).Contains(chatErr.Error(), "no chat messages provided")
}

func TestChatLogsOutput(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(chatTestdataPath)
	require.NoError(err)

	l, err := New(path)
	require.NoError(err)
	defer l.Close()

	cached, err := l.LoadModel(context.Background(), schema.LoadModelRequest{
		Name: "stories260K.gguf",
	})
	require.NoError(err)

	if !cached.Handle.HasChatTemplate() {
		t.Skip("Model has no chat template")
	}

	maxTokens := int32(16)
	temperature := float32(0)
	seed := uint32(1)

	resp, err := l.Chat(context.Background(), schema.ChatRequest{
		CompletionRequest: schema.CompletionRequest{
			Model:       "stories260K.gguf",
			MaxTokens:   &maxTokens,
			Temperature: &temperature,
			Seed:        &seed,
		},
		Messages: []schema.ChatMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
		},
	}, nil)
	require.NoError(err)
	require.NotNil(resp)

	t.Logf("chat response: %q", resp.Message.Content)
	t.Logf("usage: input=%d output=%d total=%d", resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens())
}
