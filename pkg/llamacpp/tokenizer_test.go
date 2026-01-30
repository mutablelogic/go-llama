package llamacpp_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	llama "github.com/mutablelogic/go-llama"
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	sysllamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTokenize(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "stories260K.gguf",
		Text:  "Hello, world!",
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.NotEmpty(resp.Tokens)
}

func TestTokenizeEmpty(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "stories260K.gguf",
		Text:  "",
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Empty(resp.Tokens)
}

func TestTokenizeWithOptions(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	addSpecial := true
	parseSpecial := false

	resp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model:        "stories260K.gguf",
		Text:         "Hello, world!",
		AddSpecial:   &addSpecial,
		ParseSpecial: &parseSpecial,
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.NotEmpty(resp.Tokens)
}

func TestTokenizeModelNotFound(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	_, err = l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "nonexistent.gguf",
		Text:  "Hello",
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestDetokenize(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// First tokenize some text
	tokenResp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "stories260K.gguf",
		Text:  "Hello, world!",
	})
	require.NoError(err)
	require.NotEmpty(tokenResp.Tokens)

	// Then detokenize
	detokResp, err := l.Detokenize(context.Background(), schema.DetokenizeRequest{
		Model:  "stories260K.gguf",
		Tokens: tokenResp.Tokens,
	})

	require.NoError(err)
	require.NotNil(detokResp)
	assert.NotEmpty(detokResp.Text)
}

func TestDetokenizeEmpty(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	resp, err := l.Detokenize(context.Background(), schema.DetokenizeRequest{
		Model:  "stories260K.gguf",
		Tokens: []sysllamacpp.Token{},
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.Empty(resp.Text)
}

func TestDetokenizeWithOptions(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// First tokenize
	tokenResp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "stories260K.gguf",
		Text:  "Hello, world!",
	})
	require.NoError(err)

	removeSpecial := true
	unparseSpecial := false

	resp, err := l.Detokenize(context.Background(), schema.DetokenizeRequest{
		Model:          "stories260K.gguf",
		Tokens:         tokenResp.Tokens,
		RemoveSpecial:  &removeSpecial,
		UnparseSpecial: &unparseSpecial,
	})

	require.NoError(err)
	require.NotNil(resp)
	assert.NotEmpty(resp.Text)
}

func TestDetokenizeModelNotFound(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	_, err = l.Detokenize(context.Background(), schema.DetokenizeRequest{
		Model:  "nonexistent.gguf",
		Tokens: []sysllamacpp.Token{1, 2, 3},
	})

	require.Error(err)
	assert.True(errors.Is(err, llama.ErrNotFound))
}

func TestTokenizeRoundTrip(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	originalText := "The quick brown fox jumps over the lazy dog."

	// Tokenize without special tokens for clean round-trip
	addSpecial := false
	tokenResp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model:      "stories260K.gguf",
		Text:       originalText,
		AddSpecial: &addSpecial,
	})
	require.NoError(err)
	require.NotEmpty(tokenResp.Tokens)

	// Detokenize
	detokResp, err := l.Detokenize(context.Background(), schema.DetokenizeRequest{
		Model:  "stories260K.gguf",
		Tokens: tokenResp.Tokens,
	})
	require.NoError(err)

	// Log the actual values for debugging
	t.Logf("Original: %q", originalText)
	t.Logf("Tokens:   %v", tokenResp.Tokens)
	t.Logf("Decoded:  %q", detokResp.Text)

	// Text should match exactly
	assert.Equal(originalText, detokResp.Text)
}

func TestTokenizeConcurrent(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Run multiple concurrent tokenize calls on the same model
	const numGoroutines = 10
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	texts := []string{
		"Hello, world!",
		"The quick brown fox",
		"Testing concurrent tokenization",
		"Multiple goroutines",
		"Thread safety test",
	}

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			text := texts[idx%len(texts)]
			resp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
				Model: "stories260K.gguf",
				Text:  text,
			})
			if err != nil {
				errCh <- err
				return
			}
			if len(resp.Tokens) == 0 && len(text) > 0 {
				errCh <- errors.New("expected non-empty tokens")
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("Concurrent tokenize error: %v", err)
	}
}

func TestDetokenizeConcurrent(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Pre-tokenize some texts
	addSpecial := false
	tokenSets := make([][]sysllamacpp.Token, 0)
	texts := []string{"Hello", "World", "Test", "Tokens", "Here"}

	for _, text := range texts {
		resp, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
			Model:      "stories260K.gguf",
			Text:       text,
			AddSpecial: &addSpecial,
		})
		require.NoError(err)
		tokenSets = append(tokenSets, resp.Tokens)
	}

	// Run concurrent detokenize
	const numGoroutines = 10
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			tokens := tokenSets[idx%len(tokenSets)]
			resp, err := l.Detokenize(context.Background(), schema.DetokenizeRequest{
				Model:  "stories260K.gguf",
				Tokens: tokens,
			})
			if err != nil {
				errCh <- err
				return
			}
			if len(resp.Text) == 0 && len(tokens) > 0 {
				errCh <- errors.New("expected non-empty text")
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("Concurrent detokenize error: %v", err)
	}
}

func TestTokenizeMultipleModels(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	path, err := filepath.Abs(testdataPath)
	require.NoError(err)

	l, err := llamacpp.New(path)
	require.NoError(err)
	defer l.Close()

	// Tokenize with first model
	resp1, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "stories260K.gguf",
		Text:  "Hello",
	})
	require.NoError(err)
	assert.NotEmpty(resp1.Tokens)

	// Tokenize with second model (BERT model)
	resp2, err := l.Tokenize(context.Background(), schema.TokenizeRequest{
		Model: "all-MiniLM-L6-v2-Q4_K_M.gguf",
		Text:  "Hello",
	})
	require.NoError(err)
	assert.NotEmpty(resp2.Tokens)

	// Tokens may differ between models (different vocabularies)
	t.Logf("stories260K tokens: %v", resp1.Tokens)
	t.Logf("MiniLM tokens: %v", resp2.Tokens)
}
