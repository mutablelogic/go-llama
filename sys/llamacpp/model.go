package llamacpp

/*
#include "model.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Model represents a loaded LLM model with caching support
type Model struct {
	handle unsafe.Pointer
	path   string
}

// ModelParams contains configuration for loading a model
type ModelParams struct {
	NGPULayers int32 // Number of layers to offload to GPU (-1 = all)
	MainGPU    int32 // Main GPU index
	UseMmap    bool  // Use memory mapping for model loading
	UseMlock   bool  // Lock model in memory
}

///////////////////////////////////////////////////////////////////////////////
// MODEL LOADING

// DefaultModelParams returns default model loading parameters
func DefaultModelParams() ModelParams {
	cParams := C.llama_go_model_default_params()
	return ModelParams{
		NGPULayers: int32(cParams.n_gpu_layers),
		MainGPU:    int32(cParams.main_gpu),
		UseMmap:    bool(cParams.use_mmap),
		UseMlock:   bool(cParams.use_mlock),
	}
}

// LoadModel loads a model from the given path
// Models are cached and reference-counted; loading the same path multiple
// times returns the same cached model with an incremented reference count
func LoadModel(path string, params ModelParams) (*Model, error) {
	// Ensure backend is initialized
	if err := Init(); err != nil {
		return nil, err
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cParams := C.llama_go_model_params{
		n_gpu_layers: C.int32_t(params.NGPULayers),
		main_gpu:     C.int32_t(params.MainGPU),
		use_mmap:     C.bool(params.UseMmap),
		use_mlock:    C.bool(params.UseMlock),
	}

	handle := C.llama_go_model_load(cPath, cParams)
	if handle == nil {
		return nil, getLastError()
	}

	model := &Model{
		handle: handle,
		path:   path,
	}

	// Set finalizer to release the model when garbage collected
	runtime.SetFinalizer(model, func(m *Model) {
		m.Close()
	})

	return model, nil
}

// Close releases the model reference
// The underlying model is freed when the last reference is released
func (m *Model) Close() error {
	if m.handle != nil {
		C.llama_go_model_release(m.handle)
		m.handle = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// MODEL INFO

// Path returns the file path of the model
func (m *Model) Path() string {
	return m.path
}

// ContextSize returns the model's training context size
func (m *Model) ContextSize() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_ctx_train(m.handle))
}

// EmbeddingSize returns the model's embedding dimension
func (m *Model) EmbeddingSize() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_embd(m.handle))
}

// LayerCount returns the number of layers in the model
func (m *Model) LayerCount() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_layer(m.handle))
}

// VocabSize returns the vocabulary size
func (m *Model) VocabSize() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_vocab(m.handle))
}

///////////////////////////////////////////////////////////////////////////////
// MODEL METADATA

// Metadata returns a metadata value by key
func (m *Model) Metadata(key string) string {
	if m.handle == nil {
		return ""
	}

	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	cVal := C.llama_go_model_meta_val_str(m.handle, cKey)
	if cVal == nil {
		return ""
	}

	return C.GoString(cVal)
}

///////////////////////////////////////////////////////////////////////////////
// CACHE MANAGEMENT

// CacheCount returns the number of cached models
func CacheCount() int32 {
	return int32(C.llama_go_model_cache_count())
}

// ClearCache clears all cached models
func ClearCache() {
	C.llama_go_model_cache_clear()
}
