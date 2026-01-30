package gguf

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo pkg-config: libllama
#cgo linux pkg-config: libllama-linux
#cgo darwin pkg-config: libllama-darwin
#cgo windows pkg-config: libllama-windows
#include <gguf.h>
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// CONSTANTS

const (
	// FileExtension is the file extension for GGUF model files
	FileExtension = ".gguf"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Context represents a GGUF file context for reading metadata
type Context struct {
	ctx *C.struct_gguf_context
}

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// Open opens a GGUF file and returns a Context for reading metadata.
// The caller must call Close() when done.
func Open(path string) (*Context, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	params := C.struct_gguf_init_params{
		no_alloc: C.bool(true),
		ctx:      nil,
	}

	ctx := C.gguf_init_from_file(cPath, params)
	if ctx == nil {
		return nil, ErrOpenFailed
	}

	return &Context{ctx: ctx}, nil
}

// Close releases the GGUF context resources
func (c *Context) Close() error {
	if c.ctx != nil {
		C.gguf_free(c.ctx)
		c.ctx = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS - METADATA ACCESS

// MetaCount returns the number of key-value pairs in the GGUF metadata
func (c *Context) MetaCount() int {
	if c.ctx == nil {
		return 0
	}
	return int(C.gguf_get_n_kv(c.ctx))
}

// MetaKey returns the key at the given index
func (c *Context) MetaKey(index int) (string, error) {
	if c.ctx == nil {
		return "", ErrInvalidContext
	}
	n := int(C.gguf_get_n_kv(c.ctx))
	if index < 0 || index >= n {
		return "", ErrIndexOutOfRange
	}
	key := C.gguf_get_key(c.ctx, C.int64_t(index))
	return C.GoString(key), nil
}

// MetaValue returns the value for a given key as an interface{}
func (c *Context) MetaValue(key string) (any, error) {
	if c.ctx == nil {
		return nil, ErrInvalidContext
	}

	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	idx := C.gguf_find_key(c.ctx, cKey)
	if idx < 0 {
		return nil, ErrKeyNotFound
	}

	return c.valueAtIndex(int(idx))
}

// AllMetadata returns all key-value pairs as a map
func (c *Context) AllMetadata() (map[string]any, error) {
	if c.ctx == nil {
		return nil, ErrInvalidContext
	}

	n := c.MetaCount()
	result := make(map[string]any, n)

	for i := 0; i < n; i++ {
		key, err := c.MetaKey(i)
		if err != nil {
			return nil, err
		}
		value, err := c.valueAtIndex(i)
		if err != nil {
			return nil, err
		}
		result[key] = value
	}

	return result, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS - COMMON ACCESSORS

// Name returns the model name from metadata, or empty string if not found
func (c *Context) Name() string {
	if v, err := c.MetaValue("general.name"); err == nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Architecture returns the model architecture, or empty string if not found
func (c *Context) Architecture() string {
	if v, err := c.MetaValue("general.architecture"); err == nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Description returns the model description, or empty string if not found
func (c *Context) Description() string {
	if v, err := c.MetaValue("general.description"); err == nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// ChatTemplate returns the chat template, or empty string if not found
func (c *Context) ChatTemplate() string {
	if v, err := c.MetaValue("tokenizer.chat_template"); err == nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func (c *Context) valueAtIndex(idx int) (any, error) {
	valType := C.gguf_get_kv_type(c.ctx, C.int64_t(idx))

	switch valType {
	case C.GGUF_TYPE_UINT8:
		return uint8(C.gguf_get_val_u8(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_INT8:
		return int8(C.gguf_get_val_i8(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_UINT16:
		return uint16(C.gguf_get_val_u16(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_INT16:
		return int16(C.gguf_get_val_i16(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_UINT32:
		return uint32(C.gguf_get_val_u32(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_INT32:
		return int32(C.gguf_get_val_i32(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_UINT64:
		return uint64(C.gguf_get_val_u64(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_INT64:
		return int64(C.gguf_get_val_i64(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_FLOAT32:
		return float32(C.gguf_get_val_f32(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_FLOAT64:
		return float64(C.gguf_get_val_f64(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_BOOL:
		return bool(C.gguf_get_val_bool(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_STRING:
		return C.GoString(C.gguf_get_val_str(c.ctx, C.int64_t(idx))), nil
	case C.GGUF_TYPE_ARRAY:
		// Arrays are complex - return nil for now
		// Could be extended to handle typed arrays
		return nil, nil
	default:
		return nil, ErrTypeMismatch
	}
}
