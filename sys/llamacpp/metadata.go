package llamacpp

/*
#include "metadata.h"
#include "model.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

///////////////////////////////////////////////////////////////////////////////
// MODEL METADATA

// MetaCount returns the number of metadata key-value pairs in the model
func (m *Model) MetaCount() int {
	if m.handle == nil {
		return 0
	}
	return int(C.llama_go_model_meta_count(m.handle))
}

// MetaKey returns the metadata key at the given index.
// Returns ErrInvalidModel if model is closed, ErrIndexOutOfRange if index is invalid.
func (m *Model) MetaKey(index int) (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}
	key := C.llama_go_model_meta_key(m.handle, C.int32_t(index))
	if key == nil {
		return "", ErrIndexOutOfRange
	}
	return C.GoString(key), nil
}

// MetaValue returns the metadata value for the given key.
// Returns ErrInvalidModel if model is closed, ErrKeyNotFound if key doesn't exist.
func (m *Model) MetaValue(key string) (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	cValue := C.llama_go_model_meta_value(m.handle, cKey)
	if cValue == nil {
		return "", ErrKeyNotFound
	}
	defer C.llama_go_free_string(cValue)

	return C.GoString(cValue), nil
}

// AllMetadata returns all metadata as a map.
// Returns ErrInvalidModel if model is closed.
func (m *Model) AllMetadata() (map[string]string, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}
	count := m.MetaCount()
	if count == 0 {
		return make(map[string]string), nil
	}

	meta := make(map[string]string, count)
	for i := 0; i < count; i++ {
		key, err := m.MetaKey(i)
		if err != nil {
			continue // Skip invalid keys
		}
		value, err := m.MetaValue(key)
		if err != nil {
			continue // Skip keys with no value
		}
		meta[key] = value
	}
	return meta, nil
}

///////////////////////////////////////////////////////////////////////////////
// COMMON METADATA ACCESSORS

// Name returns the model name from metadata.
// Returns ErrInvalidModel if model is closed, ErrKeyNotFound if name not set.
func (m *Model) Name() (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}
	name := C.llama_go_model_name(m.handle)
	if name == nil {
		return "", ErrKeyNotFound
	}
	return C.GoString(name), nil
}

// Arch returns the model architecture (e.g., "llama", "mistral", "phi").
// Returns ErrInvalidModel if model is closed, ErrKeyNotFound if architecture not set.
func (m *Model) Arch() (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}
	arch := C.llama_go_model_arch(m.handle)
	if arch == nil {
		return "", ErrKeyNotFound
	}
	return C.GoString(arch), nil
}

// Description returns the model description from metadata.
// Returns ErrInvalidModel if model is closed, ErrKeyNotFound if description not set.
func (m *Model) Description() (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}
	desc := C.llama_go_model_description(m.handle)
	if desc == nil {
		return "", ErrKeyNotFound
	}
	return C.GoString(desc), nil
}

///////////////////////////////////////////////////////////////////////////////
// MODEL DIMENSIONS

// NLayer returns the number of layers in the model
func (m *Model) NLayer() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_layer(m.handle))
}

// NHead returns the number of attention heads
func (m *Model) NHead() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_head(m.handle))
}

// NHeadKV returns the number of key-value attention heads (for GQA/MQA)
func (m *Model) NHeadKV() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_head_kv(m.handle))
}

// NEmbd returns the embedding dimension
func (m *Model) NEmbd() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_embd(m.handle))
}

// NCtxTrain returns the training context length
func (m *Model) NCtxTrain() int32 {
	if m.handle == nil {
		return 0
	}
	return int32(C.llama_go_model_n_ctx_train(m.handle))
}
