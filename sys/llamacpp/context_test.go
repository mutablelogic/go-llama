package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModel = "../../testdata/stories260K.gguf"

func TestDefaultContextParams(t *testing.T) {
	params := llamacpp.DefaultContextParams()

	// Check that defaults are reasonable
	if params.NBatch == 0 {
		t.Error("Expected non-zero default batch size")
	}
	if params.NThreads < 0 {
		t.Error("Expected non-negative default thread count")
	}
	t.Logf("Default params: NCtx=%d, NBatch=%d, NUBatch=%d, NSeqMax=%d, NThreads=%d",
		params.NCtx, params.NBatch, params.NUBatch, params.NSeqMax, params.NThreads)
}

func TestNewContext(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512 // Use smaller context for testing

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	t.Log("Context created successfully")
}

func TestContextNilModel(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	ctxParams := llamacpp.DefaultContextParams()

	_, err := llamacpp.NewContext(nil, ctxParams)
	if err == nil {
		t.Error("Expected error when creating context with nil model")
	}
	if err != llamacpp.ErrInvalidModel {
		t.Errorf("Expected ErrInvalidModel, got %v", err)
	}
}

func TestContextSize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	ctxSize := ctx.ContextSize()
	if ctxSize == 0 {
		t.Error("Expected non-zero context size")
	}
	t.Logf("Context size: %d", ctxSize)
}

func TestContextBatchSize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NBatch = 128
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	batchSize := ctx.BatchSize()
	if batchSize == 0 {
		t.Error("Expected non-zero batch size")
	}
	t.Logf("Batch size: %d", batchSize)
}

func TestContextUBatchSize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	ubatchSize := ctx.UBatchSize()
	t.Logf("UBatch size: %d", ubatchSize)
}

func TestContextSeqMax(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	seqMax := ctx.SeqMax()
	if seqMax == 0 {
		t.Error("Expected non-zero seq max")
	}
	t.Logf("Seq max: %d", seqMax)
}

func TestContextModel(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	if ctx.Model() != model {
		t.Error("Expected context model to match original model")
	}
}

func TestContextClose(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	// Close explicitly
	err = ctx.Close()
	if err != nil {
		t.Errorf("Close returned error: %v", err)
	}

	// Context size should be 0 after close
	if ctx.ContextSize() != 0 {
		t.Error("Expected 0 context size after close")
	}
}

func TestContextDoubleClose(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	// Close twice should not panic
	ctx.Close()
	ctx.Close()
}

func TestContextMultiple(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256

	// Create multiple contexts from the same model
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

	// Both contexts should be valid
	if ctx1.ContextSize() == 0 {
		t.Error("Expected non-zero context size for ctx1")
	}
	if ctx2.ContextSize() == 0 {
		t.Error("Expected non-zero context size for ctx2")
	}
}

func TestContextWithEmbeddings(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, modelParams)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256
	ctxParams.Embeddings = true

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("Failed to create context with embeddings: %v", err)
	}
	defer ctx.Close()

	t.Log("Context created with embeddings enabled")
}
