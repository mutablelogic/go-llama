package llamacpp

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	sysllamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const completionTestdataPath = "../../testdata"

func TestBuildCompletionOptionsDefaults(t *testing.T) {
	assert := assert.New(t)

	defaults := sysllamacpp.DefaultCompletionOptions()
	opts := buildCompletionOptions(nil, schema.CompletionRequest{})

	assert.Equal(defaults.MaxTokens, opts.MaxTokens)
	assert.Equal(defaults.EnablePrefixCaching, opts.EnablePrefixCaching)
	assert.Nil(opts.StopWords)
	assert.Equal(defaults.SamplerParams, opts.SamplerParams)
	assert.Nil(opts.AbortContext)
}

func TestBuildCompletionOptionsOverrides(t *testing.T) {
	assert := assert.New(t)

	maxTokens := int32(42)
	seed := uint32(123)
	temperature := float32(0.7)
	topP := float32(0.8)
	topK := int32(12)
	prefixCache := false
	stop := []string{"END"}

	opts := buildCompletionOptions(context.Background(), schema.CompletionRequest{
		MaxTokens:   &maxTokens,
		Seed:        &seed,
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
		PrefixCache: &prefixCache,
		Stop:        stop,
	})

	assert.Equal(int(maxTokens), opts.MaxTokens)
	assert.Equal(stop, opts.StopWords)
	assert.Equal(prefixCache, opts.EnablePrefixCaching)
	assert.Equal(seed, opts.SamplerParams.Seed)
	assert.Equal(temperature, opts.SamplerParams.Temperature)
	assert.Equal(topP, opts.SamplerParams.TopP)
	assert.Equal(topK, opts.SamplerParams.TopK)
	assert.NotNil(opts.AbortContext)
}

func TestBuildCompletionOptionsAbortContext(t *testing.T) {
	assert := assert.New(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := buildCompletionOptions(ctx, schema.CompletionRequest{})
	assert.Equal(ctx, opts.AbortContext)
}

func TestCompleteLogsOutput(t *testing.T) {
	require := require.New(t)

	path, err := filepath.Abs(completionTestdataPath)
	require.NoError(err)

	l, err := New(path)
	require.NoError(err)
	defer l.Close()

	maxTokens := int32(16)
	temperature := float32(0)
	seed := uint32(1)

	resp, err := l.Complete(context.Background(), schema.CompletionRequest{
		Model:       "stories260K.gguf",
		Prompt:      "Hello",
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		Seed:        &seed,
	}, nil)
	require.NoError(err)
	require.NotNil(resp)

	t.Logf("completion: %q", resp.Text)
	t.Logf("usage: input=%d output=%d total=%d", resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens())
}
