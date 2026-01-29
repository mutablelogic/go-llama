package llamacpp_test

import (
	"sync"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelConcurrency = "../../testdata/stories260K.gguf"

///////////////////////////////////////////////////////////////////////////////
// THREAD SAFETY NOTES
//
// llama.cpp has very limited thread safety. Based on testing:
//
// The following are NOT thread-safe in llama.cpp:
// - Model loading
// - Tokenization/detokenization on the same model
// - Metadata access on the same model
// - Context creation from the same model
// - Decoding (even on separate contexts from the same model)
// - Sampler creation
// - Log level changes (underlying C library uses global state)
// - Batch operations (even independent batches may use shared GPU state)
//
// For concurrent inference, use separate model instances loaded sequentially,
// or serialize all access to a shared model.

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL MODEL OPERATION TESTS

// TestSequentialTokenize tests tokenization operations.
func TestSequentialTokenize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	texts := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"Testing tokenization operations.",
		"1 + 1 = 2",
		"Lorem ipsum dolor sit amet.",
	}

	opts := llamacpp.DefaultTokenizeOptions()

	// Tokenize texts sequentially
	for _, text := range texts {
		tokens, err := model.Tokenize(text, opts)
		if err != nil {
			t.Errorf("Tokenize %q failed: %v", text, err)
			continue
		}
		if len(tokens) == 0 {
			t.Logf("Warning: empty tokens for %q", text)
		}
	}
}

// TestSequentialDetokenize tests detokenization operations.
func TestSequentialDetokenize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Pre-tokenize some texts
	texts := []string{
		"Hello, world!",
		"Testing detokenization.",
		"Multiple operations.",
	}
	tokOpts := llamacpp.TokenizeOptions{AddSpecial: false}
	detokOpts := llamacpp.DefaultDetokenizeOptions()

	for _, text := range texts {
		tokens, err := model.Tokenize(text, tokOpts)
		if err != nil {
			t.Fatalf("Failed to tokenize %q: %v", text, err)
		}

		_, err = model.Detokenize(tokens, detokOpts)
		if err != nil {
			t.Errorf("Detokenize failed: %v", err)
		}
	}
}

// TestSequentialMetadata tests metadata access operations.
func TestSequentialMetadata(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Access metadata sequentially
	for i := 0; i < 20; i++ {
		_ = model.MetaCount()
		_, _ = model.Arch()
		_, _ = model.Name()
		_, _ = model.AllMetadata()
		_ = model.VocabSize()
	}
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL CONTEXT TESTS

// TestSequentialContextCreation tests creating multiple contexts from the same model.
func TestSequentialContextCreation(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256 // Small context to allow multiple

	contexts := make([]*llamacpp.Context, 0)

	// Create multiple contexts sequentially
	for i := 0; i < 5; i++ {
		ctx, err := llamacpp.NewContext(model, ctxParams)
		if err != nil {
			t.Fatalf("Failed to create context %d: %v", i, err)
		}
		contexts = append(contexts, ctx)
	}

	// Clean up all contexts
	for _, ctx := range contexts {
		ctx.Close()
	}

	t.Logf("Successfully created %d contexts sequentially", len(contexts))
}

// TestSequentialDecode tests decoding on separate contexts from the same model.
func TestSequentialDecode(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	// Create and use contexts sequentially
	for i := 0; i < 3; i++ {
		ctx, err := llamacpp.NewContext(model, ctxParams)
		if err != nil {
			t.Fatalf("Failed to create context %d: %v", i, err)
		}

		// Run multiple decode operations
		for j := 0; j < 3; j++ {
			ctx.MemoryClear(true)

			tokens := []llamacpp.Token{1, 2, 3}
			batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
			if err != nil {
				t.Fatalf("Failed to create batch: %v", err)
			}

			err = batch.Decode(ctx)
			batch.Close()
			if err != nil {
				t.Fatalf("Decode failed on context %d, iteration %d: %v", i, j, err)
			}
		}

		ctx.Close()
	}
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL BATCH TESTS

// TestSequentialBatchOperations tests batch creation and manipulation.
func TestSequentialBatchOperations(t *testing.T) {
	t.Parallel() // Batch operations are thread-safe

	// Create and manipulate multiple batches sequentially
	for i := 0; i < 5; i++ {
		batch, err := llamacpp.NewBatch(100, 1)
		if err != nil {
			t.Fatalf("Failed to create batch %d: %v", i, err)
		}

		// Add tokens
		for j := int32(0); j < 50; j++ {
			err := batch.Add(llamacpp.Token(j), j, 0, j == 49)
			if err != nil {
				t.Fatalf("Failed to add token: %v", err)
			}
		}

		// Clear and reuse
		batch.Clear()

		tokens := []llamacpp.Token{1, 2, 3, 4, 5}
		_, err = batch.AddTokens(tokens, 0, 0, true)
		if err != nil {
			t.Fatalf("Failed to add tokens: %v", err)
		}

		batch.Close()
	}
}

///////////////////////////////////////////////////////////////////////////////
// CONCURRENT BATCH TESTS

// TestConcurrentBatchOperations tests concurrent batch creation and manipulation.
func TestConcurrentBatchOperations(t *testing.T) {
	t.Parallel()

	var wg sync.WaitGroup
	numGoroutines := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Each goroutine creates its own batches
			for j := 0; j < 3; j++ {
				batch, err := llamacpp.NewBatch(100, 1)
				if err != nil {
					t.Errorf("Goroutine %d: Failed to create batch: %v", id, err)
					return
				}

				// Add tokens
				for k := int32(0); k < 20; k++ {
					batch.Add(llamacpp.Token(k), k, 0, k == 19)
				}

				// Clear and reuse
				batch.Clear()

				tokens := []llamacpp.Token{1, 2, 3}
				batch.AddTokens(tokens, 0, 0, true)

				batch.Close()
			}
		}(i)
	}

	wg.Wait()
	t.Logf("Successfully completed %d concurrent batch operations", numGoroutines)
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL LOGGING TESTS

// TestSequentialLogCallback tests log callback access.
func TestSequentialLogCallback(t *testing.T) {
	t.Parallel() // Log callback operations are safe with proper locking

	var mu sync.Mutex
	messages := make([]string, 0)

	callback := func(level llamacpp.LogLevel, msg string) {
		mu.Lock()
		messages = append(messages, msg)
		mu.Unlock()
	}

	// Set and unset log callbacks sequentially
	for i := 0; i < 10; i++ {
		llamacpp.SetLogCallback(callback)
		llamacpp.SetLogCallback(nil)
	}

	// Reset to nil
	llamacpp.SetLogCallback(nil)
}

// TestSequentialLogLevel tests log level changes.
func TestSequentialLogLevel(t *testing.T) {
	t.Parallel() // Log level changes are atomic

	levels := []llamacpp.LogLevel{
		llamacpp.LogLevelDebug,
		llamacpp.LogLevelInfo,
		llamacpp.LogLevelWarn,
		llamacpp.LogLevelError,
		llamacpp.LogLevelNone,
	}

	for i := 0; i < 20; i++ {
		for _, level := range levels {
			llamacpp.SetLogLevel(level)
			_ = llamacpp.GetLogLevel()
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// CONCURRENT LOGGING TESTS

// TestConcurrentLogLevel tests concurrent log level changes.
func TestConcurrentLogLevel(t *testing.T) {
	t.Parallel()

	var wg sync.WaitGroup
	numGoroutines := 5

	levels := []llamacpp.LogLevel{
		llamacpp.LogLevelDebug,
		llamacpp.LogLevelInfo,
		llamacpp.LogLevelWarn,
		llamacpp.LogLevelError,
	}

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Each goroutine cycles through log levels
			for j := 0; j < 10; j++ {
				level := levels[j%len(levels)]
				llamacpp.SetLogLevel(level)
				_ = llamacpp.GetLogLevel()
			}
		}(i)
	}

	wg.Wait()
	t.Log("Successfully completed concurrent log level operations")
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL SAMPLER TESTS

// TestSamplerCreationSequential tests creating multiple samplers sequentially.
func TestSamplerCreationSequential(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	samplers := make([]*llamacpp.Sampler, 0)

	// Create multiple samplers sequentially
	for i := 0; i < 5; i++ {
		params := llamacpp.DefaultSamplerParams()
		params.Temperature = float32(i) * 0.1
		params.TopK = int32(10 + i)

		sampler, err := llamacpp.NewSampler(model, params)
		if err != nil {
			t.Fatalf("Failed to create sampler %d: %v", i, err)
		}
		samplers = append(samplers, sampler)
	}

	// Clean up
	for _, sampler := range samplers {
		sampler.Close()
	}

	t.Logf("Successfully created %d samplers sequentially", len(samplers))
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL MODEL CACHE TESTS

// TestSequentialModelCache tests model caching behavior.
func TestSequentialModelCache(t *testing.T) {
	llamacpp.Init()
	llamacpp.ClearCache()

	params := llamacpp.DefaultModelParams()

	// Load the same model multiple times sequentially
	models := make([]*llamacpp.Model, 0)
	for i := 0; i < 5; i++ {
		model, err := llamacpp.LoadModel(testModelConcurrency, params)
		if err != nil {
			t.Fatalf("Failed to load model %d: %v", i, err)
		}
		models = append(models, model)
	}

	// All loads should succeed and use the same cached model
	cacheCount := llamacpp.CacheCount()
	if cacheCount != 1 {
		t.Errorf("Expected cache count 1 (same model reused), got %d", cacheCount)
	}

	// Close all model references
	for _, model := range models {
		model.Close()
	}

	t.Logf("Successfully loaded model %d times sequentially, cache count: %d", len(models), cacheCount)

	llamacpp.ClearCache()
	llamacpp.Cleanup()
}

///////////////////////////////////////////////////////////////////////////////
// SEQUENTIAL SPECIAL TOKEN ACCESS

// TestSequentialSpecialTokens tests access to special tokens.
func TestSequentialSpecialTokens(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Access special tokens sequentially
	for i := 0; i < 20; i++ {
		_ = model.BOS()
		_ = model.EOS()
		_ = model.EOT()
		_ = model.NL()
		_ = model.PAD()

		bos := model.BOS()
		if bos >= 0 {
			_ = model.IsEOG(bos)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// SHARED MODEL WITH MULTIPLE CONTEXTS

// TestSharedModelMultipleContexts tests creating multiple contexts from one model.
// This tests whether multiple contexts can be created and used with a shared model.
func TestSharedModelMultipleContexts(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load one model
	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Create multiple contexts from the same model
	numContexts := 3
	contexts := make([]*llamacpp.Context, numContexts)

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512

	for i := 0; i < numContexts; i++ {
		ctx, err := llamacpp.NewContext(model, ctxParams)
		if err != nil {
			t.Fatalf("Failed to create context %d: %v", i, err)
		}
		defer ctx.Close()
		contexts[i] = ctx
	}

	// Use contexts sequentially (safe for basic operations)
	for i, ctx := range contexts {
		nEmbd := ctx.NEmbd()
		t.Logf("Context %d: NEmbd=%d", i, nEmbd)
	}
}

// TestSharedModelSequentialDecode tests sequential decode on multiple contexts from one model.
// According to llama.cpp docs, decode is NOT thread-safe even with separate contexts.
func TestSharedModelSequentialDecode(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load one model
	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Tokenize test text
	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("No tokens generated")
	}

	// Create two contexts
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512

	ctx1, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context 1: %v", err)
	}
	defer ctx1.Close()

	ctx2, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context 2: %v", err)
	}
	defer ctx2.Close()

	// Create batches and decode sequentially on each context
	batch1, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("Failed to create batch 1: %v", err)
	}
	defer batch1.Close()

	if err := batch1.Decode(ctx1); err != nil {
		t.Errorf("Context 1 decode failed: %v", err)
	}

	batch2, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("Failed to create batch 2: %v", err)
	}
	defer batch2.Close()

	if err := batch2.Decode(ctx2); err != nil {
		t.Errorf("Context 2 decode failed: %v", err)
	}

	t.Log("Sequential decode on separate contexts from same model completed")
}

// TestSharedModelConcurrentDecode demonstrates concurrent decode on multiple contexts.
// WARNING: This is expected to be UNSAFE according to llama.cpp documentation.
// We use sync.Mutex to serialize access and demonstrate the requirement for external locking.
func TestSharedModelConcurrentDecode(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Load one model
	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelConcurrency, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Tokenize test text
	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("No tokens generated")
	}

	// Create two contexts
	numContexts := 2
	contexts := make([]*llamacpp.Context, numContexts)

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512

	for i := 0; i < numContexts; i++ {
		ctx, err := llamacpp.NewContext(model, ctxParams)
		if err != nil {
			t.Fatalf("Failed to create context %d: %v", i, err)
		}
		defer ctx.Close()
		contexts[i] = ctx
	}

	// Use mutex to serialize decode operations (required for safety)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var decodeErrors []error
	var errMu sync.Mutex

	for i := 0; i < numContexts; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Use sequence ID 0 for all batches (each context has its own sequence space)
			batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
			if err != nil {
				errMu.Lock()
				decodeErrors = append(decodeErrors, err)
				errMu.Unlock()
				return
			}
			defer batch.Close()

			// CRITICAL: Must lock because llama.cpp decode is NOT thread-safe
			// even with separate contexts from the same model
			mu.Lock()
			err = batch.Decode(contexts[idx])
			mu.Unlock()

			if err != nil {
				errMu.Lock()
				decodeErrors = append(decodeErrors, err)
				errMu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	if len(decodeErrors) > 0 {
		t.Errorf("Decode errors: %v", decodeErrors)
	} else {
		t.Log("Concurrent decode with mutex protection completed successfully")
	}
}

///////////////////////////////////////////////////////////////////////////////
// CONCURRENT MODEL TESTS (with separate model instances)

// TestConcurrentSeparateModels tests operations on separate model instances.
// Each goroutine loads its own model instance.
func TestConcurrentSeparateModels(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	var wg sync.WaitGroup
	numGoroutines := 3 // Limited to avoid resource exhaustion

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Each goroutine loads its own model
			params := llamacpp.DefaultModelParams()
			model, err := llamacpp.LoadModel(testModelConcurrency, params)
			if err != nil {
				t.Errorf("Goroutine %d: Failed to load model: %v", id, err)
				return
			}
			defer model.Close()

			// Perform basic operations on the model
			_ = model.VocabSize()
			_ = model.BOS()
			_ = model.EOS()

			// Tokenize some text
			opts := llamacpp.DefaultTokenizeOptions()
			tokens, err := model.Tokenize("Hello world", opts)
			if err != nil {
				t.Errorf("Goroutine %d: Tokenize failed: %v", id, err)
				return
			}

			if len(tokens) == 0 {
				t.Errorf("Goroutine %d: Empty token result", id)
			}
		}(i)
	}

	wg.Wait()
	t.Logf("Successfully completed %d concurrent operations with separate models", numGoroutines)
}

///////////////////////////////////////////////////////////////////////////////
// PERFORMANCE BENCHMARKS

// BenchmarkBatchCreation benchmarks batch creation.
func BenchmarkBatchCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		batch, err := llamacpp.NewBatch(100, 1)
		if err != nil {
			b.Fatal(err)
		}
		batch.Close()
	}
}

// BenchmarkBatchOperations benchmarks batch add operations.
func BenchmarkBatchOperations(b *testing.B) {
	batch, err := llamacpp.NewBatch(1000, 1)
	if err != nil {
		b.Fatal(err)
	}
	defer batch.Close()

	tokens := make([]llamacpp.Token, 100)
	for i := range tokens {
		tokens[i] = llamacpp.Token(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch.Clear()
		batch.AddTokens(tokens, 0, 0, true)
	}
}

// BenchmarkConcurrentBatchCreation benchmarks concurrent batch creation.
func BenchmarkConcurrentBatchCreation(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			batch, err := llamacpp.NewBatch(100, 1)
			if err != nil {
				b.Error(err)
				return
			}
			batch.Close()
		}
	})
}
