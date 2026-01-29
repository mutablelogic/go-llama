package llamacpp_test

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

// testModelPath returns the path to the test model
func testModelPath(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to get test file path")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "testdata", "stories260K.gguf")
}

///////////////////////////////////////////////////////////////////////////////
// INITIALIZATION TESTS

func TestInit(t *testing.T) {
	// Test initialization
	if err := llamacpp.Init(); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	// Should be initialized
	if !llamacpp.IsInitialized() {
		t.Error("Expected backend to be initialized")
	}

	// Multiple init calls should be safe
	if err := llamacpp.Init(); err != nil {
		t.Fatalf("Second Init failed: %v", err)
	}
}

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMS TESTS

func TestDefaultParams(t *testing.T) {
	params := llamacpp.DefaultModelParams()

	// Check default values
	if params.NGPULayers != -1 {
		t.Errorf("Expected NGPULayers=-1, got %d", params.NGPULayers)
	}
	if params.MainGPU != 0 {
		t.Errorf("Expected MainGPU=0, got %d", params.MainGPU)
	}
	if !params.UseMmap {
		t.Error("Expected UseMmap=true")
	}
	if params.UseMlock {
		t.Error("Expected UseMlock=false")
	}
}

///////////////////////////////////////////////////////////////////////////////
// MODEL LOADING TESTS

func TestLoadModel(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	if model == nil {
		t.Fatal("LoadModel() returned nil model without error")
	}
}

func TestLoadModelInvalidPath(t *testing.T) {
	params := llamacpp.DefaultModelParams()

	// Try to load a non-existent model
	model, err := llamacpp.LoadModel("/nonexistent/model.gguf", params)
	if err == nil {
		model.Close()
		t.Fatal("Expected error loading non-existent model")
	}

	t.Logf("Got expected error: %v", err)
}

func TestLoadModelEmptyPath(t *testing.T) {
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel("", params)
	if err == nil {
		model.Close()
		t.Fatal("LoadModel() should fail for empty path")
	}
}

///////////////////////////////////////////////////////////////////////////////
// MODEL INFO TESTS

func TestModelPath(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	if model.Path() != modelPath {
		t.Errorf("Path() = %q, want %q", model.Path(), modelPath)
	}
}

func TestModelContextSize(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	ctxSize := model.ContextSize()
	if ctxSize <= 0 {
		t.Errorf("ContextSize() = %d, want > 0", ctxSize)
	}
	t.Logf("Model context size: %d", ctxSize)
}

func TestModelEmbeddingSize(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	embdSize := model.EmbeddingSize()
	if embdSize <= 0 {
		t.Errorf("EmbeddingSize() = %d, want > 0", embdSize)
	}
	t.Logf("Model embedding size: %d", embdSize)
}

func TestModelLayerCount(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	layers := model.LayerCount()
	if layers <= 0 {
		t.Errorf("LayerCount() = %d, want > 0", layers)
	}
	t.Logf("Model layer count: %d", layers)
}

func TestModelVocabSize(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	vocabSize := model.VocabSize()
	if vocabSize <= 0 {
		t.Errorf("VocabSize() = %d, want > 0", vocabSize)
	}
	t.Logf("Model vocab size: %d", vocabSize)
}

///////////////////////////////////////////////////////////////////////////////
// MODEL METADATA TESTS

func TestModelMetadata(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	// Try to get a common metadata key
	arch := model.Metadata("general.architecture")
	t.Logf("Model architecture: %q", arch)

	// Non-existent key should return empty string
	nonexistent := model.Metadata("nonexistent.key")
	if nonexistent != "" {
		t.Errorf("Metadata() for nonexistent key = %q, want empty string", nonexistent)
	}
}

// Note: TestModelChatTemplate is in chat_test.go

///////////////////////////////////////////////////////////////////////////////
// MODEL CLOSE TESTS

func TestModelClose(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}

	// Close should succeed
	err = model.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}

	// Close again should be safe (idempotent)
	err = model.Close()
	if err != nil {
		t.Errorf("Close() second call failed: %v", err)
	}
}

func TestModelMethodsAfterClose(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	// Methods should return zero/empty values after close, not panic
	if model.ContextSize() != 0 {
		t.Error("ContextSize() should return 0 after Close()")
	}
	if model.EmbeddingSize() != 0 {
		t.Error("EmbeddingSize() should return 0 after Close()")
	}
	if model.LayerCount() != 0 {
		t.Error("LayerCount() should return 0 after Close()")
	}
	if model.VocabSize() != 0 {
		t.Error("VocabSize() should return 0 after Close()")
	}
	if model.Metadata("test") != "" {
		t.Error("Metadata() should return empty string after Close()")
	}
	if model.ChatTemplate("") != "" {
		t.Error("ChatTemplate() should return empty string after Close()")
	}
}

///////////////////////////////////////////////////////////////////////////////
// CACHE TESTS

func TestCacheCount(t *testing.T) {
	// Clear cache first
	llamacpp.ClearCache()

	initialCount := llamacpp.CacheCount()
	if initialCount != 0 {
		t.Errorf("CacheCount() after ClearCache() = %d, want 0", initialCount)
	}

	// Load a model
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	defer model.Close()

	// Cache count should be 1
	count := llamacpp.CacheCount()
	if count != 1 {
		t.Errorf("CacheCount() = %d, want 1", count)
	}
}

func TestClearCache(t *testing.T) {
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	// Load and close a model
	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	// Clear cache
	llamacpp.ClearCache()

	count := llamacpp.CacheCount()
	if count != 0 {
		t.Errorf("CacheCount() after ClearCache() = %d, want 0", count)
	}
}

func TestCacheReuse(t *testing.T) {
	llamacpp.ClearCache()

	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	// Load model first time
	model1, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() first call failed: %v", err)
	}
	defer model1.Close()

	// Cache count should be 1
	if llamacpp.CacheCount() != 1 {
		t.Errorf("CacheCount() = %d, want 1", llamacpp.CacheCount())
	}

	// Load same model again - should reuse cache
	model2, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() second call failed: %v", err)
	}
	defer model2.Close()

	// Cache count should still be 1 (same model reused)
	if llamacpp.CacheCount() != 1 {
		t.Errorf("CacheCount() after second load = %d, want 1", llamacpp.CacheCount())
	}
}

///////////////////////////////////////////////////////////////////////////////
// CLEANUP TESTS

func TestCleanup(t *testing.T) {
	// Initialize if not already
	llamacpp.Init()

	// Clear cache
	llamacpp.ClearCache()

	// Cleanup
	llamacpp.Cleanup()
}

///////////////////////////////////////////////////////////////////////////////
// INTEGRATION TEST

func TestModelFullWorkflow(t *testing.T) {
	// Clear any previous state
	llamacpp.ClearCache()

	// Initialize
	err := llamacpp.Init()
	if err != nil {
		t.Fatalf("Init() failed: %v", err)
	}

	// Load model
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}

	// Verify model info
	t.Logf("Loaded model: %s", model.Path())
	t.Logf("  Context size: %d", model.ContextSize())
	t.Logf("  Embedding size: %d", model.EmbeddingSize())
	t.Logf("  Layer count: %d", model.LayerCount())
	t.Logf("  Vocab size: %d", model.VocabSize())
	t.Logf("  Architecture: %s", model.Metadata("general.architecture"))

	// Cleanup
	model.Close()
	llamacpp.ClearCache()

	if llamacpp.CacheCount() != 0 {
		t.Errorf("CacheCount() after cleanup = %d, want 0", llamacpp.CacheCount())
	}
}
