package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestMemorySeqKeep(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NSeqMax = 4
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Tokenize and process some text
	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("failed to tokenize: %v", err)
	}

	batch, err := llamacpp.NewBatch(int32(len(tokens)), 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}

	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// Check that memory has content
	posMax := ctx.MemorySeqPosMax(0)
	if posMax < 0 {
		t.Fatal("expected memory to have content")
	}
	t.Logf("After decode, seq 0 pos range: [%d, %d]", ctx.MemorySeqPosMin(0), posMax)

	// MemorySeqKeep should keep only seq 0
	ctx.MemorySeqKeep(0)

	// Seq 0 should still have content
	posMax = ctx.MemorySeqPosMax(0)
	if posMax < 0 {
		t.Error("expected seq 0 to still have content after MemorySeqKeep")
	}
}

func TestMemorySeqCopy(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NSeqMax = 4
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Tokenize and process prefix
	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("failed to tokenize: %v", err)
	}

	batch, err := llamacpp.NewBatch(int32(len(tokens)), 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}

	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// Copy seq 0 to seq 1 (prefix sharing)
	ctx.MemorySeqCp(0, 1, -1, -1)

	// Both sequences should now have the same positions
	posMax0 := ctx.MemorySeqPosMax(0)
	posMax1 := ctx.MemorySeqPosMax(1)
	t.Logf("After copy: seq 0 posMax=%d, seq 1 posMax=%d", posMax0, posMax1)

	if posMax1 < 0 {
		t.Error("expected seq 1 to have content after copy")
	}
}

func TestStateSeqSaveLoad(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
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

	// Tokenize and process some text
	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("Hello world", opts)
	if err != nil {
		t.Fatalf("failed to tokenize: %v", err)
	}

	batch, err := llamacpp.NewBatch(int32(len(tokens)), 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}

	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// Save the sequence state
	stateSize := ctx.StateSeqGetSize(0)
	t.Logf("State size for seq 0: %d bytes", stateSize)
	if stateSize == 0 {
		t.Skip("StateSeqGetSize returned 0 - may not be supported for this model")
	}

	stateData, err := ctx.StateSeqGetData(0)
	if err != nil {
		t.Fatalf("failed to get state data: %v", err)
	}
	t.Logf("Saved %d bytes of state data", len(stateData))

	// Clear the memory
	ctx.MemoryClear(true)

	// Verify memory is cleared
	posMax := ctx.MemorySeqPosMax(0)
	if posMax >= 0 {
		t.Error("expected memory to be empty after clear")
	}

	// Restore the state
	read, err := ctx.StateSeqSetData(0, stateData)
	if err != nil {
		t.Fatalf("failed to restore state: %v", err)
	}
	t.Logf("Restored %d bytes of state data", read)

	// Verify memory is restored
	posMax = ctx.MemorySeqPosMax(0)
	if posMax < 0 {
		t.Error("expected memory to have content after restore")
	}
}

func TestPrefixCachingWorkflow(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NSeqMax = 4
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Step 1: Process a common prefix
	opts := llamacpp.DefaultTokenizeOptions()
	prefix := "Once upon a time"
	prefixTokens, err := model.Tokenize(prefix, opts)
	if err != nil {
		t.Fatalf("failed to tokenize prefix: %v", err)
	}

	batch, err := llamacpp.NewBatch(32, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Process prefix on seq 0
	for i, tok := range prefixTokens {
		batch.Add(tok, int32(i), 0, i == len(prefixTokens)-1)
	}

	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode prefix failed: %v", err)
	}
	t.Logf("Prefix processed: %d tokens, posMax=%d", len(prefixTokens), ctx.MemorySeqPosMax(0))

	// Step 2: Copy prefix to other sequences for parallel generation
	ctx.MemorySeqCp(0, 1, -1, -1)
	ctx.MemorySeqCp(0, 2, -1, -1)
	t.Logf("After copy - seq0=%d, seq1=%d, seq2=%d",
		ctx.MemorySeqPosMax(0), ctx.MemorySeqPosMax(1), ctx.MemorySeqPosMax(2))

	// Step 3: Add different suffixes to each sequence
	suffix1, _ := model.Tokenize(" there was", opts)
	suffix2, _ := model.Tokenize(" in a land", opts)

	// Add suffix1 to seq 1
	batch.Clear()
	for i, tok := range suffix1 {
		batch.Add(tok, int32(len(prefixTokens)+i), 1, i == len(suffix1)-1)
	}
	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode suffix1 failed: %v", err)
	}

	// Add suffix2 to seq 2
	batch.Clear()
	for i, tok := range suffix2 {
		batch.Add(tok, int32(len(prefixTokens)+i), 2, i == len(suffix2)-1)
	}
	if err := batch.Decode(ctx); err != nil {
		t.Fatalf("decode suffix2 failed: %v", err)
	}

	// Verify different sequences have different lengths
	posMax1 := ctx.MemorySeqPosMax(1)
	posMax2 := ctx.MemorySeqPosMax(2)
	t.Logf("Final positions - seq1=%d, seq2=%d", posMax1, posMax2)

	if posMax1 < int32(len(prefixTokens)) {
		t.Errorf("seq 1 should have at least prefix+suffix1 tokens, got posMax=%d", posMax1)
	}
	if posMax2 < int32(len(prefixTokens)) {
		t.Errorf("seq 2 should have at least prefix+suffix2 tokens, got posMax=%d", posMax2)
	}
}

func TestMemoryCanShift(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
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

	canShift := ctx.MemoryCanShift()
	t.Logf("Memory can shift: %v", canShift)
	// Just verify it doesn't crash - the result depends on the model
}
