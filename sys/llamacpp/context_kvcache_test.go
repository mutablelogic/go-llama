package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestGGMLTypeString(t *testing.T) {
	tests := []struct {
		typ  llamacpp.GGMLType
		want string
	}{
		{llamacpp.GGMLTypeF32, "f32"},
		{llamacpp.GGMLTypeF16, "f16"},
		{llamacpp.GGMLTypeQ4_0, "q4_0"},
		{llamacpp.GGMLTypeQ4_1, "q4_1"},
		{llamacpp.GGMLTypeQ5_0, "q5_0"},
		{llamacpp.GGMLTypeQ5_1, "q5_1"},
		{llamacpp.GGMLTypeQ8_0, "q8_0"},
		{llamacpp.GGMLTypeQ8_1, "q8_1"},
		{llamacpp.GGMLTypeBF16, "bf16"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := tt.typ.String()
			if got != tt.want {
				t.Errorf("GGMLType(%d).String() = %q, want %q", tt.typ, got, tt.want)
			}
		})
	}
}

func TestDefaultKVCacheType(t *testing.T) {
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

	typeK := ctx.KVCacheTypeK()
	typeV := ctx.KVCacheTypeV()

	t.Logf("Default KV cache types: K=%s, V=%s", typeK, typeV)

	if typeK != llamacpp.GGMLTypeF16 {
		t.Errorf("expected default typeK=F16, got %s", typeK)
	}
	if typeV != llamacpp.GGMLTypeF16 {
		t.Errorf("expected default typeV=F16, got %s", typeV)
	}
}

func TestKVCacheQuantizationQ8(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.TypeK = llamacpp.GGMLTypeQ8_0
	ctxParams.TypeV = llamacpp.GGMLTypeQ8_0

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context with Q8 KV cache: %v", err)
	}
	defer ctx.Close()

	typeK := ctx.KVCacheTypeK()
	typeV := ctx.KVCacheTypeV()

	t.Logf("Q8 KV cache types: K=%s, V=%s", typeK, typeV)

	if typeK != llamacpp.GGMLTypeQ8_0 {
		t.Errorf("expected typeK=Q8_0, got %s", typeK)
	}
	if typeV != llamacpp.GGMLTypeQ8_0 {
		t.Errorf("expected typeV=Q8_0, got %s", typeV)
	}

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, _ := model.Tokenize("Hello world", opts)
	batch, _ := llamacpp.NewBatch(int32(len(tokens)), 1)
	defer batch.Close()

	for i, tok := range tokens {
		batch.Add(tok, int32(i), 0, i == len(tokens)-1)
	}

	if err := batch.Decode(ctx); err != nil {
		t.Errorf("decode failed with Q8 KV cache: %v", err)
	}
}

func TestKVCacheQuantizationQ4(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.TypeK = llamacpp.GGMLTypeQ4_0
	ctxParams.TypeV = llamacpp.GGMLTypeQ4_0

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context with Q4 KV cache: %v", err)
	}
	defer ctx.Close()

	typeK := ctx.KVCacheTypeK()
	typeV := ctx.KVCacheTypeV()

	t.Logf("Q4 KV cache types: K=%s, V=%s", typeK, typeV)

	if typeK != llamacpp.GGMLTypeQ4_0 {
		t.Errorf("expected typeK=Q4_0, got %s", typeK)
	}
	if typeV != llamacpp.GGMLTypeQ4_0 {
		t.Errorf("expected typeV=Q4_0, got %s", typeV)
	}
}

func TestKVCacheMixedTypes(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	ctxParams := llamacpp.DefaultContextParams()
	ctxParams.TypeK = llamacpp.GGMLTypeQ8_0
	ctxParams.TypeV = llamacpp.GGMLTypeQ4_0

	ctx, err := llamacpp.NewContext(model, ctxParams)
	if err != nil {
		t.Fatalf("failed to create context with mixed KV types: %v", err)
	}
	defer ctx.Close()

	t.Logf("Mixed KV cache types: K=%s, V=%s", ctx.KVCacheTypeK(), ctx.KVCacheTypeV())

	if ctx.KVCacheTypeK() != llamacpp.GGMLTypeQ8_0 {
		t.Errorf("expected typeK=Q8_0, got %s", ctx.KVCacheTypeK())
	}
	if ctx.KVCacheTypeV() != llamacpp.GGMLTypeQ4_0 {
		t.Errorf("expected typeV=Q4_0, got %s", ctx.KVCacheTypeV())
	}
}

func TestContextParamsDefaultTypes(t *testing.T) {
	params := llamacpp.DefaultContextParams()

	if params.TypeK != -1 {
		t.Errorf("expected default TypeK=-1, got %d", params.TypeK)
	}
	if params.TypeV != -1 {
		t.Errorf("expected default TypeV=-1, got %d", params.TypeV)
	}
}
