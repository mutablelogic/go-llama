package llamacpp_test

import (
	"strings"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelCompletion = "../../testdata/stories260K.gguf"

// TestBasicCompletion tests simple text completion
func TestBasicCompletion(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load model
	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelCompletion, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Create context
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	// Test completion
	opts := llamacpp.DefaultCompletionOptions()
	opts.MaxTokens = 20
	opts.SamplerParams = llamacpp.GreedySamplerParams() // Deterministic

	result, err := ctx.Complete("Once upon a time", opts)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if result == "" {
		t.Error("Got empty completion result")
	}

	t.Logf("Completion result: %q", result)
}

// TestCompletionWithCallback tests token streaming callback
func TestCompletionWithCallback(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load model
	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelCompletion, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Create context
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	// Track tokens
	var tokens []string
	opts := llamacpp.DefaultCompletionOptions()
	opts.MaxTokens = 10
	opts.SamplerParams = llamacpp.GreedySamplerParams()
	opts.OnToken = func(token string) bool {
		tokens = append(tokens, token)
		return true // Continue
	}

	result, err := ctx.Complete("Hello", opts)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("No tokens received in callback")
	}

	// Verify tokens reconstruct result
	reconstructed := strings.Join(tokens, "")
	if reconstructed != result {
		t.Errorf("Tokens don't match result: %q vs %q", reconstructed, result)
	}

	t.Logf("Received %d tokens: %v", len(tokens), tokens)
}

// TestCompletionWithStopWords tests stop word handling
func TestCompletionWithStopWords(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load model
	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelCompletion, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Create context
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	// Generate with stop words
	opts := llamacpp.DefaultCompletionOptions()
	opts.MaxTokens = 50
	opts.StopWords = []string{".", "\n"}
	opts.SamplerParams = llamacpp.GreedySamplerParams()

	result, err := ctx.Complete("Once upon a time", opts)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	hasStopWord := strings.Contains(result, ".") || strings.Contains(result, "\n")
	if hasStopWord {
		t.Logf("Correctly stopped at stop word: %q", result)
	} else {
		t.Logf("Result (may not have hit stop word): %q", result)
	}
}
