package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelDecode = "../../testdata/stories260K.gguf"

func TestDecodeGetLogits(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	// Create and decode a batch
	tokens := []llamacpp.Token{1, 100, 200} // BOS + tokens
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	defer batch.Close()

	err = batch.Decode(ctx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// Get logits for last token
	logits, err := ctx.GetLogits(-1)
	if err != nil {
		t.Fatalf("failed to get logits: %v", err)
	}

	nVocab := ctx.NVocab()
	if nVocab <= 0 {
		t.Fatal("expected positive vocabulary size")
	}

	if len(logits) != int(nVocab) {
		t.Errorf("expected %d logits, got %d", nVocab, len(logits))
	}

	t.Logf("Vocabulary size: %d, logits[0]=%f", nVocab, logits[0])
}

func TestDecodeNVocab(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	nVocab := ctx.NVocab()
	if nVocab <= 0 {
		t.Errorf("expected positive vocabulary size, got %d", nVocab)
	}
	t.Logf("Vocabulary size: %d", nVocab)
}

func TestDecodeNEmbd(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	nEmbd := ctx.NEmbd()
	if nEmbd <= 0 {
		t.Errorf("expected positive embedding dimension, got %d", nEmbd)
	}
	t.Logf("Embedding dimension: %d", nEmbd)
}

func TestDecodeMemoryClear(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	// Decode some tokens
	tokens := []llamacpp.Token{1, 100, 200}
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	err = batch.Decode(ctx)
	batch.Close()
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// Check memory has tokens
	posMax := ctx.MemorySeqPosMax(0)
	if posMax < 0 {
		t.Error("expected memory to have tokens after decode")
	}
	t.Logf("Memory pos max before clear: %d", posMax)

	// Clear memory
	ctx.MemoryClear(true)

	// Check memory is empty
	posMax = ctx.MemorySeqPosMax(0)
	if posMax >= 0 {
		t.Errorf("expected empty memory after clear, got pos_max %d", posMax)
	}
}

func TestDecodeMemorySeqRm(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	// Decode tokens
	tokens := []llamacpp.Token{1, 100, 200, 300, 400}
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	err = batch.Decode(ctx)
	batch.Close()
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	posMaxBefore := ctx.MemorySeqPosMax(0)
	t.Logf("Memory pos max before remove: %d", posMaxBefore)

	// Remove some tokens from sequence 0
	ok := ctx.MemorySeqRm(0, 2, 5)
	t.Logf("Memory seq rm returned: %v", ok)

	posMaxAfter := ctx.MemorySeqPosMax(0)
	t.Logf("Memory pos max after remove: %d", posMaxAfter)
}

func TestDecodeSynchronize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	// Should not panic
	ctx.Synchronize()
}

func TestDecodeMemoryCanShift(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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
}

func TestDecodeAndSample(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
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

	// Create sampler
	sampler, err := llamacpp.NewSampler(model, llamacpp.GreedySamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	defer sampler.Close()

	// Decode initial tokens
	tokens := []llamacpp.Token{1} // Just BOS
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}

	err = batch.Decode(ctx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	batch.Close()

	// Sample next token
	nextToken, err := sampler.Sample(ctx, -1)
	if err != nil {
		t.Fatalf("sample failed: %v", err)
	}

	if nextToken < 0 {
		t.Error("expected valid token")
	}
	t.Logf("Sampled token: %d", nextToken)

	// Accept the token
	sampler.Accept(nextToken)
}

func TestDecodeGenerateMultiple(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelDecode, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 256
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	// Greedy sampler for deterministic output
	sampler, err := llamacpp.NewSampler(model, llamacpp.GreedySamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	defer sampler.Close()

	// Start with BOS
	tokens := []llamacpp.Token{1}
	batch, err := llamacpp.BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		t.Fatalf("failed to create batch: %v", err)
	}
	err = batch.Decode(ctx)
	batch.Close()
	if err != nil {
		t.Fatalf("initial decode failed: %v", err)
	}

	pos := int32(1)
	generated := []llamacpp.Token{}

	// Generate 10 tokens
	for i := 0; i < 10; i++ {
		// Sample
		token, err := sampler.Sample(ctx, -1)
		if err != nil {
			t.Fatalf("sample failed at step %d: %v", i, err)
		}
		sampler.Accept(token)
		generated = append(generated, token)

		// Check for EOS
		if model.IsEOG(token) {
			t.Logf("Hit EOG at step %d", i)
			break
		}

		// Decode the new token
		b, err := llamacpp.NewBatch(1, 1)
		if err != nil {
			t.Fatalf("failed to create batch: %v", err)
		}
		b.Add(token, pos, 0, true)
		err = b.Decode(ctx)
		b.Close()
		if err != nil {
			t.Fatalf("decode failed at step %d: %v", i, err)
		}
		pos++
	}

	t.Logf("Generated %d tokens: %v", len(generated), generated)
	if len(generated) == 0 {
		t.Error("expected to generate at least one token")
	}
}
