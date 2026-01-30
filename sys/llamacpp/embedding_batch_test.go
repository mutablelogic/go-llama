package llamacpp_test

import (
	"math"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestComputeEmbedding(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	opts := llamacpp.DefaultEmbeddingOptions()
	embd, err := ctx.ComputeEmbedding(model, "Hello world", opts)
	if err != nil {
		t.Fatalf("failed to compute embedding: %v", err)
	}

	t.Logf("Embedding dimension: %d", len(embd))

	if len(embd) != int(ctx.NEmbd()) {
		t.Errorf("expected embedding dimension %d, got %d", ctx.NEmbd(), len(embd))
	}

	var mag float64
	for _, v := range embd {
		mag += float64(v) * float64(v)
	}
	mag = math.Sqrt(mag)
	if math.Abs(mag-1.0) > 0.01 {
		t.Errorf("expected normalized embedding (mag ~= 1.0), got %f", mag)
	}
}

func TestComputeEmbeddings(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctxParams.NSeqMax = 8
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	texts := []string{
		"Hello world",
		"The cat sat on the mat",
		"Once upon a time",
	}

	opts := llamacpp.DefaultEmbeddingOptions()
	batch, err := ctx.ComputeEmbeddings(model, texts, opts)
	if err != nil {
		t.Fatalf("failed to compute embeddings: %v", err)
	}

	if len(batch.Embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(batch.Embeddings))
	}

	t.Logf("Computed %d embeddings of dimension %d", len(batch.Embeddings), batch.Dimension)
}

func TestComputeEmbeddingsEmpty(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	opts := llamacpp.DefaultEmbeddingOptions()
	batch, err := ctx.ComputeEmbeddings(model, []string{}, opts)
	if err != nil {
		t.Fatalf("failed with empty input: %v", err)
	}

	if len(batch.Embeddings) != 0 {
		t.Errorf("expected 0 embeddings for empty input, got %d", len(batch.Embeddings))
	}
}

func TestSimilarityMatrix(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctxParams.NSeqMax = 8
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	texts := []string{
		"The cat sat on the mat",
		"A feline rested on the rug",
		"The quick brown fox jumps",
	}

	opts := llamacpp.DefaultEmbeddingOptions()
	batch, err := ctx.ComputeEmbeddings(model, texts, opts)
	if err != nil {
		t.Fatalf("failed to compute embeddings: %v", err)
	}

	matrix := batch.SimilarityMatrix()
	if len(matrix) != len(texts) {
		t.Fatalf("expected %dx%d matrix", len(texts), len(texts))
	}

	for i := 0; i < len(matrix); i++ {
		if math.Abs(matrix[i][i]-1.0) > 0.01 {
			t.Errorf("expected self-similarity = 1.0, got %f", matrix[i][i])
		}
	}
}

func TestMostSimilar(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/all-MiniLM-L6-v2-Q4_K_M.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.Embeddings = true
	ctxParams.NSeqMax = 8
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	texts := []string{
		"Hello world",
		"Goodbye world",
		"The quick brown fox",
		"Hi there",
	}

	opts := llamacpp.DefaultEmbeddingOptions()
	batch, err := ctx.ComputeEmbeddings(model, texts, opts)
	if err != nil {
		t.Fatalf("failed to compute embeddings: %v", err)
	}

	similar := batch.MostSimilar(0, 2)
	if len(similar) != 2 {
		t.Errorf("expected 2 similar results, got %d", len(similar))
	}

	for _, idx := range similar {
		if idx == 0 {
			t.Error("query index should not appear in similar results")
		}
	}
}

func TestMostSimilarEdgeCases(t *testing.T) {
	batch := &llamacpp.BatchEmbeddings{
		Embeddings: [][]float32{
			{1, 0, 0},
			{0, 1, 0},
		},
		Dimension: 3,
	}

	if result := batch.MostSimilar(-1, 1); result != nil {
		t.Error("expected nil for negative query index")
	}

	if result := batch.MostSimilar(10, 1); result != nil {
		t.Error("expected nil for out of bounds query index")
	}

	if result := batch.MostSimilar(0, 0); result != nil {
		t.Error("expected nil for k=0")
	}

	result := batch.MostSimilar(0, 10)
	if len(result) != 1 {
		t.Errorf("expected 1 result, got %d", len(result))
	}
}

func TestSimilarityMatrixEmpty(t *testing.T) {
	batch := &llamacpp.BatchEmbeddings{
		Embeddings: [][]float32{},
		Dimension:  3,
	}

	matrix := batch.SimilarityMatrix()
	if matrix != nil {
		t.Error("expected nil matrix for empty batch")
	}
}

func TestDefaultEmbeddingOptions(t *testing.T) {
	opts := llamacpp.DefaultEmbeddingOptions()

	if !opts.Normalize {
		t.Error("expected Normalize to be true by default")
	}
	if !opts.AddBOS {
		t.Error("expected AddBOS to be true by default")
	}
	if opts.AddEOS {
		t.Error("expected AddEOS to be false by default")
	}
}
