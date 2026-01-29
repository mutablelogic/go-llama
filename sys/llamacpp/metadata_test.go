package llamacpp_test

import (
	"errors"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelMeta = "../../testdata/stories260K.gguf"

func TestMetaCount(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	count := model.MetaCount()
	if count <= 0 {
		t.Errorf("expected positive metadata count, got %d", count)
	}
	t.Logf("Metadata count: %d", count)
}

func TestMetaKeyValue(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Get first key
	key, err := model.MetaKey(0)
	if err != nil {
		t.Fatalf("expected non-empty key at index 0: %v", err)
	}
	t.Logf("Key 0: %s", key)

	// Get value for that key
	value, err := model.MetaValue(key)
	if err != nil {
		t.Logf("Value not found for key %s: %v", key, err)
	} else {
		t.Logf("Value: %s", value)
	}

	// Invalid index should return error
	_, err = model.MetaKey(-1)
	if err == nil {
		t.Error("expected error for invalid index")
	}

	_, err = model.MetaKey(9999)
	if err == nil {
		t.Error("expected error for out-of-range index")
	}
}

func TestMetadata(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	meta, err := model.AllMetadata()
	if err != nil {
		t.Fatalf("failed to get metadata: %v", err)
	}
	if len(meta) == 0 {
		t.Fatal("expected non-empty metadata map")
	}

	t.Logf("Found %d metadata entries:", len(meta))
	for k, v := range meta {
		// Truncate long values for display
		if len(v) > 50 {
			v = v[:50] + "..."
		}
		t.Logf("  %s = %s", k, v)
	}
}

func TestModelName(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	name, err := model.Name()
	if err != nil && !errors.Is(err, llamacpp.ErrKeyNotFound) {
		t.Fatalf("unexpected error getting name: %v", err)
	}
	t.Logf("Model name: %q", name)
}

func TestModelArch(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	arch, err := model.Arch()
	if err != nil {
		t.Fatalf("failed to get architecture: %v", err)
	}
	if arch == "" {
		t.Error("expected non-empty architecture")
	}
	t.Logf("Model architecture: %s", arch)
}

func TestModelDimensions(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	t.Logf("Layers: %d", model.NLayer())
	t.Logf("Heads: %d", model.NHead())
	t.Logf("KV Heads: %d", model.NHeadKV())
	t.Logf("Embedding dim: %d", model.NEmbd())
	t.Logf("Training context: %d", model.NCtxTrain())

	if model.NLayer() <= 0 {
		t.Error("expected positive layer count")
	}
	if model.NHead() <= 0 {
		t.Error("expected positive head count")
	}
	if model.NEmbd() <= 0 {
		t.Error("expected positive embedding dimension")
	}
	if model.NCtxTrain() <= 0 {
		t.Error("expected positive training context length")
	}
}

func TestModelDescription(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelMeta, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	desc, err := model.Description()
	if err != nil && !errors.Is(err, llamacpp.ErrKeyNotFound) {
		t.Fatalf("unexpected error getting description: %v", err)
	}
	t.Logf("Model description: %q", desc)
	// Description may be empty for some models, so we just log it
}
