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

func TestFindPartialStop(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		stops    []string
		expected int
	}{
		{
			name:     "no partial match",
			text:     "hello world",
			stops:    []string{"</s>", "<|end|>"},
			expected: -1,
		},
		{
			name:     "full stop not detected as partial",
			text:     "hello</s>",
			stops:    []string{"</s>"},
			expected: 5, // partial match starts at position 5
		},
		{
			name:     "partial match single char",
			text:     "hello<",
			stops:    []string{"</s>"},
			expected: 5,
		},
		{
			name:     "partial match two chars",
			text:     "hello</",
			stops:    []string{"</s>"},
			expected: 5,
		},
		{
			name:     "partial match three chars",
			text:     "hello</s",
			stops:    []string{"</s>"},
			expected: 5,
		},
		{
			name:     "partial match with pipe",
			text:     "test<|",
			stops:    []string{"<|end|>"},
			expected: 4,
		},
		{
			name:     "partial match longer prefix",
			text:     "test<|end",
			stops:    []string{"<|end|>"},
			expected: 4,
		},
		{
			name:     "partial match almost complete",
			text:     "test<|end|",
			stops:    []string{"<|end|>"},
			expected: 4,
		},
		{
			name:     "multiple stops first matches",
			text:     "hello<",
			stops:    []string{"</s>", "<|end|>"},
			expected: 5,
		},
		{
			name:     "empty text",
			text:     "",
			stops:    []string{"</s>"},
			expected: -1,
		},
		{
			name:     "empty stops",
			text:     "hello",
			stops:    []string{},
			expected: -1,
		},
		{
			name:     "text is just partial",
			text:     "</",
			stops:    []string{"</s>"},
			expected: 0,
		},
		{
			name:     "no match when char not in stop",
			text:     "hellox",
			stops:    []string{"</s>"},
			expected: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findPartialStop(tt.text, tt.stops)
			if result != tt.expected {
				t.Errorf("findPartialStop(%q, %v) = %d, want %d", tt.text, tt.stops, result, tt.expected)
			}
		})
	}
}

func TestStopMarkerFilter(t *testing.T) {
	tests := []struct {
		name           string
		stops          []string
		tokens         []string
		expectedChunks []string
		expectedStop   bool
	}{
		{
			name:           "no stop sequences",
			stops:          []string{"</s>"},
			tokens:         []string{"hello", " ", "world"},
			expectedChunks: []string{"hello", " ", "world"},
			expectedStop:   false,
		},
		{
			name:           "full stop in single token",
			stops:          []string{"</s>"},
			tokens:         []string{"hello", "</s>", "extra"},
			expectedChunks: []string{"hello", ""},
			expectedStop:   true,
		},
		{
			name:           "stop split across tokens",
			stops:          []string{"</s>"},
			tokens:         []string{"hello<", "/s>more"},
			expectedChunks: []string{"hello", ""},
			expectedStop:   true,
		},
		{
			name:           "partial withholding then release",
			stops:          []string{"</s>"},
			tokens:         []string{"hello<", "notastop"},
			expectedChunks: []string{"hello", "<notastop"},
			expectedStop:   false,
		},
		{
			name:           "partial withholding with longer sequence",
			stops:          []string{"<|end|>"},
			tokens:         []string{"test<|", "end|>done"},
			expectedChunks: []string{"test", ""},
			expectedStop:   true,
		},
		{
			name:           "buffer accumulates partial",
			stops:          []string{"</s>"},
			tokens:         []string{"a<", "/", "s>"},
			expectedChunks: []string{"a", "", ""},
			expectedStop:   true,
		},
		{
			name:           "stop at very beginning",
			stops:          []string{"</s>"},
			tokens:         []string{"</s>rest"},
			expectedChunks: []string{""},
			expectedStop:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := newStopMarkerFilter(tt.stops)
			var chunks []string

			for _, token := range tt.tokens {
				chunk, stopped := filter.Process(token)
				chunks = append(chunks, chunk)
				if stopped {
					break
				}
			}

			// If not stopped, flush remaining buffer
			if !filter.Stopped() {
				if tail := filter.Flush(); tail != "" {
					chunks[len(chunks)-1] += tail
				}
			}

			if len(chunks) != len(tt.expectedChunks) {
				t.Errorf("got %d chunks %v, want %d chunks %v", len(chunks), chunks, len(tt.expectedChunks), tt.expectedChunks)
				return
			}

			for i, chunk := range chunks {
				if chunk != tt.expectedChunks[i] {
					t.Errorf("chunk[%d] = %q, want %q", i, chunk, tt.expectedChunks[i])
				}
			}

			if filter.Stopped() != tt.expectedStop {
				t.Errorf("Stopped() = %v, want %v", filter.Stopped(), tt.expectedStop)
			}
		})
	}
}
