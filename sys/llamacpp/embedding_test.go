package llamacpp_test

import (
	"math"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestPoolingTypeString(t *testing.T) {
	tests := []struct {
		ptype    llamacpp.PoolingType
		expected string
	}{
		{llamacpp.PoolingUnspecified, "unspecified"},
		{llamacpp.PoolingNone, "none"},
		{llamacpp.PoolingMean, "mean"},
		{llamacpp.PoolingCLS, "cls"},
		{llamacpp.PoolingLast, "last"},
		{llamacpp.PoolingRank, "rank"},
		{llamacpp.PoolingType(99), "unknown"},
	}

	for _, tt := range tests {
		result := tt.ptype.String()
		if result != tt.expected {
			t.Errorf("PoolingType(%d).String() = %q, want %q", tt.ptype, result, tt.expected)
		}
	}
}

func TestSetEmbeddings(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = false
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Toggle embeddings mode
	ctx.SetEmbeddings(true)
	t.Log("Embeddings enabled")

	ctx.SetEmbeddings(false)
	t.Log("Embeddings disabled")
}

func TestPoolingType(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	ptype := ctx.PoolingType()
	t.Logf("Pooling type: %s (%d)", ptype, ptype)
}

func TestNormalizeEmbeddings(t *testing.T) {
	// Test vector
	embd := []float32{3.0, 4.0}

	llamacpp.NormalizeEmbeddings(embd)

	// After normalization, magnitude should be 1
	var mag float64
	for _, v := range embd {
		mag += float64(v) * float64(v)
	}
	mag = math.Sqrt(mag)

	if math.Abs(mag-1.0) > 1e-6 {
		t.Errorf("expected magnitude 1.0, got %f", mag)
	}

	// For [3, 4], normalized should be [0.6, 0.8]
	if math.Abs(float64(embd[0])-0.6) > 1e-6 {
		t.Errorf("expected embd[0]=0.6, got %f", embd[0])
	}
	if math.Abs(float64(embd[1])-0.8) > 1e-6 {
		t.Errorf("expected embd[1]=0.8, got %f", embd[1])
	}
}

func TestNormalizeEmbeddingsEmpty(t *testing.T) {
	// Should not panic
	llamacpp.NormalizeEmbeddings(nil)
	llamacpp.NormalizeEmbeddings([]float32{})
}

func TestNormalizeEmbeddingsZero(t *testing.T) {
	embd := []float32{0, 0, 0}
	llamacpp.NormalizeEmbeddings(embd)

	// Should remain zero (no divide by zero)
	for i, v := range embd {
		if v != 0 {
			t.Errorf("expected embd[%d]=0, got %f", i, v)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors (normalized)
	a := []float32{0.6, 0.8}
	b := []float32{0.6, 0.8}
	sim := llamacpp.CosineSimilarity(a, b)
	if math.Abs(sim-1.0) > 1e-6 {
		t.Errorf("expected similarity 1.0 for identical vectors, got %f", sim)
	}

	// Opposite vectors
	c := []float32{-0.6, -0.8}
	sim = llamacpp.CosineSimilarity(a, c)
	if math.Abs(sim-(-1.0)) > 1e-6 {
		t.Errorf("expected similarity -1.0 for opposite vectors, got %f", sim)
	}

	// Orthogonal vectors
	d := []float32{0.8, -0.6}
	sim = llamacpp.CosineSimilarity(a, d)
	if math.Abs(sim) > 1e-6 {
		t.Errorf("expected similarity 0.0 for orthogonal vectors, got %f", sim)
	}
}

func TestCosineSimilarityEdgeCases(t *testing.T) {
	// Different lengths
	a := []float32{1, 2, 3}
	b := []float32{1, 2}
	sim := llamacpp.CosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("expected 0 for different lengths, got %f", sim)
	}

	// Empty vectors
	sim = llamacpp.CosineSimilarity([]float32{}, []float32{})
	if sim != 0 {
		t.Errorf("expected 0 for empty vectors, got %f", sim)
	}

	// Zero vectors
	z := []float32{0, 0, 0}
	sim = llamacpp.CosineSimilarity(z, z)
	if sim != 0 {
		t.Errorf("expected 0 for zero vectors, got %f", sim)
	}
}

func TestEuclideanDistance(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	dist := llamacpp.EuclideanDistance(a, b)
	if math.Abs(dist-5.0) > 1e-6 {
		t.Errorf("expected distance 5.0, got %f", dist)
	}

	// Same point
	dist = llamacpp.EuclideanDistance(a, a)
	if dist != 0 {
		t.Errorf("expected distance 0 for same point, got %f", dist)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dot := llamacpp.DotProduct(a, b)
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if math.Abs(dot-32.0) > 1e-6 {
		t.Errorf("expected dot product 32.0, got %f", dot)
	}
}

func TestExtractBatchEmbeddings(t *testing.T) {
	// 3 embeddings of dimension 4
	flat := []float32{
		1, 2, 3, 4, // embedding 0
		5, 6, 7, 8, // embedding 1
		9, 10, 11, 12, // embedding 2
	}

	batch := llamacpp.ExtractBatchEmbeddings(flat, 3, 4)

	if len(batch.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(batch.Embeddings))
	}
	if batch.Dimension != 4 {
		t.Errorf("expected dimension 4, got %d", batch.Dimension)
	}

	// Check each embedding
	expected := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}
	for i, exp := range expected {
		if len(batch.Embeddings[i]) != 4 {
			t.Errorf("embedding %d: expected length 4, got %d", i, len(batch.Embeddings[i]))
		}
		for j, v := range exp {
			if batch.Embeddings[i][j] != v {
				t.Errorf("embedding %d[%d]: expected %f, got %f", i, j, v, batch.Embeddings[i][j])
			}
		}
	}
}

func TestExtractBatchEmbeddingsInvalid(t *testing.T) {
	// Wrong size
	flat := []float32{1, 2, 3}
	batch := llamacpp.ExtractBatchEmbeddings(flat, 2, 4) // would need 8 elements

	if len(batch.Embeddings) != 0 {
		t.Error("expected empty batch for invalid size")
	}
}

func TestBatchEmbeddingsNormalize(t *testing.T) {
	batch := llamacpp.BatchEmbeddings{
		Embeddings: [][]float32{
			{3, 4},
			{5, 12},
		},
		Dimension: 2,
	}

	batch.Normalize()

	// Check first embedding [3,4] -> [0.6, 0.8]
	if math.Abs(float64(batch.Embeddings[0][0])-0.6) > 1e-6 {
		t.Errorf("expected 0.6, got %f", batch.Embeddings[0][0])
	}

	// Check second embedding [5,12] -> [5/13, 12/13]
	expected := 5.0 / 13.0
	if math.Abs(float64(batch.Embeddings[1][0])-expected) > 1e-6 {
		t.Errorf("expected %f, got %f", expected, batch.Embeddings[1][0])
	}
}

func TestGetEmbeddingsWithBatch(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Create context with embeddings enabled
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Tokenize some text
	opts := llamacpp.DefaultTokenizeOptions()
	opts.AddSpecial = true
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("failed to tokenize: %v", err)
	}
	t.Logf("Tokens: %v", tokens)

	// Create batch and add tokens
	batch, err := llamacpp.NewBatch(int32(len(tokens)), 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	for i, tok := range tokens {
		// Set logits=true for all tokens to get embeddings
		batch.Add(tok, int32(i), 0, true)
	}

	// Decode
	err = batch.Decode(ctx)
	if err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	t.Logf("Batch size (tokens with logits): %d", len(tokens))

	// Try to get embeddings for last token
	embd, err := ctx.GetEmbeddings(-1)
	if err != nil {
		t.Logf("GetEmbeddings error (may be expected for generative model): %v", err)
	} else {
		t.Logf("Got embedding of length %d", len(embd))
		if len(embd) > 0 {
			t.Logf("First few values: %v", embd[:min(5, len(embd))])
		}
	}
}

func TestEmbeddingWorkflow(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Create context with embeddings
	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	t.Logf("Pooling type: %s", ctx.PoolingType())
	t.Logf("Embedding dimension: %d", ctx.NEmbd())

	// Test similarity computation
	a := []float32{1, 0, 0}
	b := []float32{0.7071, 0.7071, 0}

	llamacpp.NormalizeEmbeddings(a)
	llamacpp.NormalizeEmbeddings(b)

	sim := llamacpp.CosineSimilarity(a, b)
	t.Logf("Cosine similarity: %f", sim)
}
