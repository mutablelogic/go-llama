package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestModelInfo(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	info, err := model.Info()
	if err != nil {
		t.Fatalf("failed to get model info: %v", err)
	}

	t.Logf("Model Info:")
	t.Logf("  Layers: %d", info.NLayer)
	t.Logf("  Attention heads: %d", info.NHead)
	t.Logf("  KV heads: %d", info.NHeadKV)
	t.Logf("  Embedding dim: %d", info.NEmbd)
	t.Logf("  Training context: %d", info.NCtxTrain)
	t.Logf("  Parameters: %d", info.NParams)
	t.Logf("  Model size: %d bytes", info.ModelSize)

	if info.NLayer != 5 {
		t.Errorf("expected 5 layers, got %d", info.NLayer)
	}
	if info.NHead != 8 {
		t.Errorf("expected 8 heads, got %d", info.NHead)
	}
	if info.NEmbd != 64 {
		t.Errorf("expected 64 embd, got %d", info.NEmbd)
	}
	if info.NCtxTrain != 2048 {
		t.Errorf("expected 2048 ctx_train, got %d", info.NCtxTrain)
	}
}

func TestModelParamCount(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	params := model.ParamCount()
	t.Logf("Parameter count: %d", params)

	if params < 200000 || params > 350000 {
		t.Errorf("unexpected param count %d, expected ~260K", params)
	}
}

func TestModelSizeString(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	sizeBytes := model.SizeBytes()
	sizeStr := model.SizeString()
	t.Logf("Model size: %d bytes = %s", sizeBytes, sizeStr)

	if sizeBytes == 0 {
		t.Error("expected non-zero size")
	}
	if sizeStr == "" || sizeStr == "0 B" {
		t.Error("expected non-empty size string")
	}
}

func TestContextInfo(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.NCtx = 512
	ctxParams.NBatch = 256
	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()

	info, err := ctx.Info()
	if err != nil {
		t.Fatalf("failed to get context info: %v", err)
	}

	t.Logf("Context Info:")
	t.Logf("  Context size: %d", info.NCtx)
	t.Logf("  Batch size: %d", info.NBatch)
	t.Logf("  Micro-batch: %d", info.NUBatch)
	t.Logf("  Max sequences: %d", info.NSeqMax)
	t.Logf("  Threads: %d", info.NThreads)

	if info.NCtx != 512 {
		t.Errorf("expected n_ctx=512, got %d", info.NCtx)
	}
	if info.NBatch != 256 {
		t.Errorf("expected n_batch=256, got %d", info.NBatch)
	}
}

func TestContextPerf(t *testing.T) {
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

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, _ := model.Tokenize("Hello world", opts)
	batch, _ := llamacpp.NewBatch(int32(len(tokens)), 1)
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}
	batch.Decode(ctx)

	perf, err := ctx.Perf()
	if err != nil {
		t.Fatalf("failed to get perf data: %v", err)
	}

	t.Logf("Performance Data:")
	t.Logf("  Start time: %.2f ms", perf.StartMs)
	t.Logf("  Load time: %.2f ms", perf.LoadMs)
	t.Logf("  Prompt time: %.2f ms", perf.PromptMs)
	t.Logf("  Generate time: %.2f ms", perf.GenerateMs)
	t.Logf("  Prompt tokens: %d", perf.PromptCount)
	t.Logf("  Generated tokens: %d", perf.TokenCount)
	t.Logf("  Prompt speed: %.2f t/s", perf.PromptTokensPerSecond())
	t.Logf("  Generate speed: %.2f t/s", perf.TokensPerSecond())
}

func TestContextPerfReset(t *testing.T) {
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

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, _ := model.Tokenize("Hello world", opts)
	batch, _ := llamacpp.NewBatch(int32(len(tokens)), 1)
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, true)
	}
	batch.Decode(ctx)

	ctx.PerfReset()

	perf, _ := ctx.Perf()
	t.Logf("After reset - Prompt count: %d, Token count: %d", perf.PromptCount, perf.TokenCount)

	// Note: counters may not be exactly 0 due to internal operations
	// but should be much lower than before the reset
	if perf.PromptCount > 2 || perf.TokenCount > 2 {
		t.Errorf("expected counters to be low after reset, got prompt=%d, token=%d", perf.PromptCount, perf.TokenCount)
	}
}

func TestMemorySeqPos(t *testing.T) {
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

	posMin := ctx.MemorySeqPosMin(0)
	posMax := ctx.MemorySeqPosMax(0)
	length := ctx.MemorySeqLength(0)
	t.Logf("Before decode - min: %d, max: %d, length: %d", posMin, posMax, length)

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, _ := model.Tokenize("Hello world test", opts)
	batch, _ := llamacpp.NewBatch(int32(len(tokens)), 1)
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}
	batch.Decode(ctx)

	posMin = ctx.MemorySeqPosMin(0)
	posMax = ctx.MemorySeqPosMax(0)
	length = ctx.MemorySeqLength(0)
	t.Logf("After decode - min: %d, max: %d, length: %d", posMin, posMax, length)

	if length == 0 {
		t.Error("expected non-zero sequence length after decode")
	}
}

func TestPerfTokensPerSecond(t *testing.T) {
	perf := llamacpp.PerfData{
		GenerateMs:  1000.0,
		TokenCount:  50,
		PromptMs:    500.0,
		PromptCount: 100,
	}

	tps := perf.TokensPerSecond()
	if tps != 50.0 {
		t.Errorf("expected 50 t/s, got %.2f", tps)
	}

	ptps := perf.PromptTokensPerSecond()
	if ptps != 200.0 {
		t.Errorf("expected 200 t/s, got %.2f", ptps)
	}

	empty := llamacpp.PerfData{}
	if empty.TokensPerSecond() != 0 {
		t.Error("expected 0 t/s for empty data")
	}
	if empty.PromptTokensPerSecond() != 0 {
		t.Error("expected 0 prompt t/s for empty data")
	}
}

func TestModelInfoOnClosed(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, _ := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	model.Close()

	_, err := model.Info()
	if err == nil {
		t.Error("expected error on closed model")
	}

	desc, err := model.Description()
	if err == nil {
		t.Error("expected error on closed model for Description")
	}
	if desc != "" {
		t.Error("expected empty description on closed model")
	}

	if model.ParamCount() != 0 {
		t.Error("expected 0 params on closed model")
	}
}

func TestContextInfoOnClosed(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, _ := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctx, _ := llamacpp.NewContext(model, ctxParams)
	ctx.Close()

	_, err := ctx.Info()
	if err == nil {
		t.Error("expected error on closed context")
	}

	_, err = ctx.Perf()
	if err == nil {
		t.Error("expected error on closed context perf")
	}
}
