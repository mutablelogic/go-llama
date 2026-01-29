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

// MetaKey returns the metadata key at the given index
// Returns empty string if index is out of range
func (m *Model) MetaKey(index int) string {
	if m.handle == nil {
		return ""
	}
	key := C.llama_go_model_meta_key(m.handle, C.int32_t(index))
	if key == nil {
		return ""
	}
	return C.GoString(key)
}

// MetaValue returns the metadata value for the given key
// Returns empty string if key not found
func (m *Model) MetaValue(key string) string {
	if m.handle == nil {
		return ""
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	cValue := C.llama_go_model_meta_value(m.handle, cKey)
	if cValue == nil {
		return ""
	}
	defer C.llama_go_free_string(cValue)

	return C.GoString(cValue)
}

// AllMetadata returns all metadata as a map
func (m *Model) AllMetadata() map[string]string {
	count := m.MetaCount()
	if count == 0 {
		return nil
	}

	meta := make(map[string]string, count)
	for i := 0; i < count; i++ {
		key := m.MetaKey(i)
		if key != "" {
			meta[key] = m.MetaValue(key)
		}
	}
	return meta
}

///////////////////////////////////////////////////////////////////////////////
// COMMON METADATA ACCESSORS

// Name returns the model name from metadata
func (m *Model) Name() string {
	if m.handle == nil {
		return ""
	}
	name := C.llama_go_model_name(m.handle)
	if name == nil {
		return ""
	}
	return C.GoString(name)
}

// Arch returns the model architecture (e.g., "llama", "mistral", "phi")
func (m *Model) Arch() string {
	if m.handle == nil {
		return ""
	}
	arch := C.llama_go_model_arch(m.handle)
	if arch == nil {
		return ""
	}
	return C.GoString(arch)
}

// Description returns the model description from metadata
func (m *Model) Description() string {
	if m.handle == nil {
		return ""
	}
	desc := C.llama_go_model_description(m.handle)
	if desc == nil {
		return ""
	}
	return C.GoString(desc)
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
