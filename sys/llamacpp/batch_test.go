package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelBatch = "../../testdata/stories260K.gguf"

func TestBatchNew(t *testing.T) {
	batch, err := llamacpp.NewBatch(512, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	if batch.Capacity() != 512 {
		t.Errorf("expected capacity 512, got %d", batch.Capacity())
	}
	if batch.NumTokens() != 0 {
		t.Errorf("expected 0 tokens, got %d", batch.NumTokens())
	}
}

func TestBatchAdd(t *testing.T) {
	batch, err := llamacpp.NewBatch(10, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Add a token
	ok := batch.Add(1, 0, 0, true)
	if !ok {
		t.Fatal("failed to add token")
	}
	if batch.NumTokens() != 1 {
		t.Errorf("expected 1 token, got %d", batch.NumTokens())
	}

	// Add more tokens
	ok = batch.Add(2, 1, 0, false)
	if !ok {
		t.Fatal("failed to add second token")
	}
	ok = batch.Add(3, 2, 0, true)
	if !ok {
		t.Fatal("failed to add third token")
	}
	if batch.NumTokens() != 3 {
		t.Errorf("expected 3 tokens, got %d", batch.NumTokens())
	}
}

func TestBatchClear(t *testing.T) {
	batch, err := llamacpp.NewBatch(10, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Add some tokens
	batch.Add(1, 0, 0, true)
	batch.Add(2, 1, 0, true)
	if batch.NumTokens() != 2 {
		t.Errorf("expected 2 tokens, got %d", batch.NumTokens())
	}

	// Clear
	batch.Clear()
	if batch.NumTokens() != 0 {
		t.Errorf("expected 0 tokens after clear, got %d", batch.NumTokens())
	}

	// Should be able to add again
	ok := batch.Add(3, 0, 0, true)
	if !ok {
		t.Fatal("failed to add token after clear")
	}
}

func TestBatchFull(t *testing.T) {
	batch, err := llamacpp.NewBatch(3, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Fill the batch
	batch.Add(1, 0, 0, false)
	batch.Add(2, 1, 0, false)
	batch.Add(3, 2, 0, true)

	// Should fail to add more
	ok := batch.Add(4, 3, 0, true)
	if ok {
		t.Error("expected add to fail when batch is full")
	}
	if batch.NumTokens() != 3 {
		t.Errorf("expected 3 tokens, got %d", batch.NumTokens())
	}
}

func TestBatchAddTokens(t *testing.T) {
	batch, err := llamacpp.NewBatch(100, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	tokens := []llamacpp.Token{1, 2, 3, 4, 5}
	added := batch.AddTokens(tokens, 0, 0, true)

	if added != 5 {
		t.Errorf("expected 5 tokens added, got %d", added)
	}
	if batch.NumTokens() != 5 {
		t.Errorf("expected 5 tokens, got %d", batch.NumTokens())
	}
}

func TestBatchAddTokensPartial(t *testing.T) {
	batch, err := llamacpp.NewBatch(3, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	tokens := []llamacpp.Token{1, 2, 3, 4, 5}
	added := batch.AddTokens(tokens, 0, 0, true)

	if added != 3 {
		t.Errorf("expected 3 tokens added (partial), got %d", added)
	}
	if batch.NumTokens() != 3 {
		t.Errorf("expected 3 tokens, got %d", batch.NumTokens())
	}
}

func TestBatchAddSeq(t *testing.T) {
	batch, err := llamacpp.NewBatch(10, 4)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Add token with multiple sequence IDs
	seqIDs := []int32{0, 1, 2}
	ok := batch.AddSeq(42, 0, seqIDs, true)
	if !ok {
		t.Fatal("failed to add token with multiple sequences")
	}
	if batch.NumTokens() != 1 {
		t.Errorf("expected 1 token, got %d", batch.NumTokens())
	}
}

func TestBatchSetLogits(t *testing.T) {
	batch, err := llamacpp.NewBatch(10, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Add tokens without logits
	batch.Add(1, 0, 0, false)
	batch.Add(2, 1, 0, false)
	batch.Add(3, 2, 0, false)

	// Set logits for last token
	batch.SetLogits(2, true)
	// Should not panic
}

func TestBatchFromTokens(t *testing.T) {
	tokens := []llamacpp.Token{10, 20, 30, 40, 50}

	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch from tokens: %v", err)
	}
	defer batch.Close()

	if batch.NumTokens() != 5 {
		t.Errorf("expected 5 tokens, got %d", batch.NumTokens())
	}
	if batch.Capacity() != 5 {
		t.Errorf("expected capacity 5, got %d", batch.Capacity())
	}
}

func TestBatchDecode(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelBatch, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Create batch with a few tokens
	tokens := []llamacpp.Token{1, 2, 3} // BOS and a couple tokens
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Decode should succeed
	err = batch.Decode(ctx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
}

func TestBatchDecodeMultiple(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelBatch, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// First batch - initial prompt
	tokens := []llamacpp.Token{1, 100, 200}
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}

	err = batch.Decode(ctx)
	if err != nil {
		t.Fatalf("first decode failed: %v", err)
	}
	batch.Close()

	// Second batch - continue from position 3
	batch2, err := llamacpp.NewBatch(1, 1)
	if err != nil {
		t.Fatalf("failed to create second batch: %v", err)
	}
	defer batch2.Close()

	batch2.Add(300, 3, 0, true)
	err = batch2.Decode(ctx)
	if err != nil {
		t.Fatalf("second decode failed: %v", err)
	}
}

func TestBatchInvalidParams(t *testing.T) {
	// Zero capacity should fail
	_, err := llamacpp.NewBatch(0, 1)
	if err == nil {
		t.Error("expected error for zero capacity")
	}

	// Zero seq_max should fail
	_, err = llamacpp.NewBatch(10, 0)
	if err == nil {
		t.Error("expected error for zero seq_max")
	}
}

func TestBatchCapacityReuse(t *testing.T) {
	batch, err := llamacpp.NewBatch(100, 1)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	// Fill and clear multiple times
	for round := 0; round < 3; round++ {
		for i := int32(0); i < 50; i++ {
			ok := batch.Add(llamacpp.Token(i), i, 0, i == 49)
			if !ok {
				t.Fatalf("failed to add token at round %d, pos %d", round, i)
			}
		}
		if batch.NumTokens() != 50 {
			t.Errorf("round %d: expected 50 tokens, got %d", round, batch.NumTokens())
		}
		batch.Clear()
		if batch.NumTokens() != 0 {
			t.Errorf("round %d: expected 0 tokens after clear, got %d", round, batch.NumTokens())
		}
	}
}
