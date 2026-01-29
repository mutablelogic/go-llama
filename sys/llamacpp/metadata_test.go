package llamacpp_test

import (
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
	key := model.MetaKey(0)
	if key == "" {
		t.Fatal("expected non-empty key at index 0")
	}
	t.Logf("Key 0: %s", key)

	// Get value for that key
	value := model.MetaValue(key)
	t.Logf("Value: %s", value)

	// Invalid index should return empty string
	empty := model.MetaKey(-1)
	if empty != "" {
		t.Error("expected empty string for invalid index")
	}

	empty = model.MetaKey(9999)
	if empty != "" {
		t.Error("expected empty string for out-of-range index")
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

	meta := model.AllMetadata()
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

	name := model.Name()
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

	arch := model.Arch()
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

	desc := model.Description()
	t.Logf("Model description: %q", desc)
	// Description may be empty for some models, so we just log it
}
