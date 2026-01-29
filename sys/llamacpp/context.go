package llamacpp

/*
#include "context.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// GGML TYPES FOR KV CACHE QUANTIZATION

// GGMLType represents the data type for KV cache quantization
type GGMLType int32

const (
	GGMLTypeF32  GGMLType = 0  // 32-bit float (highest precision, most memory)
	GGMLTypeF16  GGMLType = 1  // 16-bit float (default, good balance)
	GGMLTypeQ4_0 GGMLType = 2  // 4-bit quantization
	GGMLTypeQ4_1 GGMLType = 3  // 4-bit quantization with offset
	GGMLTypeQ5_0 GGMLType = 6  // 5-bit quantization
	GGMLTypeQ5_1 GGMLType = 7  // 5-bit quantization with offset
	GGMLTypeQ8_0 GGMLType = 8  // 8-bit quantization (good quality, saves memory)
	GGMLTypeQ8_1 GGMLType = 9  // 8-bit quantization with offset
	GGMLTypeBF16 GGMLType = 30 // Brain float 16
)

// String returns the name of the GGML type
func (t GGMLType) String() string {
	name := C.llama_go_ggml_type_name(C.int32_t(t))
	if name == nil {
		return "unknown"
	}
	return C.GoString(name)
}

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Context represents an inference context for a model
type Context struct {
	handle unsafe.Pointer
	model  *Model   // Keep reference to prevent GC
	typeK  GGMLType // KV cache K type (for reference)
	typeV  GGMLType // KV cache V type (for reference)
}

// ContextParams contains configuration for creating a context
type ContextParams struct {
	NCtx          uint32   // Context size (0 = from model)
	NBatch        uint32   // Logical maximum batch size
	NUBatch       uint32   // Physical maximum batch size
	NSeqMax       uint32   // Max number of sequences
	NThreads      int32    // Threads for generation
	NThreadsBatch int32    // Threads for batch processing
	RopeFreqBase  float32  // RoPE base frequency (0 = from model)
	RopeFreqScale float32  // RoPE frequency scaling (0 = from model)
	TypeK         GGMLType // KV cache K type (-1 = default F16)
	TypeV         GGMLType // KV cache V type (-1 = default F16)
	Embeddings    bool     // Extract embeddings
	OffloadKQV    bool     // Offload KQV ops to GPU
	FlashAttn     bool     // Use flash attention
	NoPerf        bool     // Disable performance timings
}

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMS

// DefaultContextParams returns default context parameters
func DefaultContextParams() ContextParams {
	cParams := C.llama_go_context_default_params()
	return ContextParams{
		NCtx:          uint32(cParams.n_ctx),
		NBatch:        uint32(cParams.n_batch),
		NUBatch:       uint32(cParams.n_ubatch),
		NSeqMax:       uint32(cParams.n_seq_max),
		NThreads:      int32(cParams.n_threads),
		NThreadsBatch: int32(cParams.n_threads_batch),
		RopeFreqBase:  float32(cParams.rope_freq_base),
		RopeFreqScale: float32(cParams.rope_freq_scale),
		TypeK:         GGMLType(cParams.type_k),
		TypeV:         GGMLType(cParams.type_v),
		Embeddings:    bool(cParams.embeddings),
		OffloadKQV:    bool(cParams.offload_kqv),
		FlashAttn:     bool(cParams.flash_attn),
		NoPerf:        bool(cParams.no_perf),
	}
}

///////////////////////////////////////////////////////////////////////////////
// CONTEXT LIFECYCLE

// NewContext creates a new inference context from a model
func NewContext(model *Model, params ContextParams) (*Context, error) {
	if model == nil {
		return nil, ErrInvalidModel
	}
	if model.handle == nil {
		return nil, ErrInvalidModel
	}

	cParams := C.llama_go_context_params{
		n_ctx:           C.uint32_t(params.NCtx),
		n_batch:         C.uint32_t(params.NBatch),
		n_ubatch:        C.uint32_t(params.NUBatch),
		n_seq_max:       C.uint32_t(params.NSeqMax),
		n_threads:       C.int32_t(params.NThreads),
		n_threads_batch: C.int32_t(params.NThreadsBatch),
		rope_freq_base:  C.float(params.RopeFreqBase),
		rope_freq_scale: C.float(params.RopeFreqScale),
		type_k:          C.int32_t(params.TypeK),
		type_v:          C.int32_t(params.TypeV),
		embeddings:      C.bool(params.Embeddings),
		offload_kqv:     C.bool(params.OffloadKQV),
		flash_attn:      C.bool(params.FlashAttn),
		no_perf:         C.bool(params.NoPerf),
	}

	handle := C.llama_go_context_new(model.handle, cParams)
	if handle == nil {
		return nil, getLastError()
	}

	// Determine actual types used (defaults to F16 if -1)
	typeK := params.TypeK
	if typeK < 0 {
		typeK = GGMLTypeF16
	}
	typeV := params.TypeV
	if typeV < 0 {
		typeV = GGMLTypeF16
	}

	ctx := &Context{
		handle: handle,
		model:  model,
		typeK:  typeK,
		typeV:  typeV,
	}

	// Set finalizer to free context when garbage collected
	runtime.SetFinalizer(ctx, func(c *Context) {
		c.Close()
	})

	return ctx, nil
}

// Close frees the context resources
func (c *Context) Close() error {
	if c.handle != nil {
		C.llama_go_context_free(c.handle)
		c.handle = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// CONTEXT INFO

// ContextSize returns the actual context size
func (c *Context) ContextSize() uint32 {
	if c.handle == nil {
		return 0
	}
	return uint32(C.llama_go_context_n_ctx(c.handle))
}

// BatchSize returns the logical maximum batch size
func (c *Context) BatchSize() uint32 {
	if c.handle == nil {
		return 0
	}
	return uint32(C.llama_go_context_n_batch(c.handle))
}

// UBatchSize returns the physical maximum batch size
func (c *Context) UBatchSize() uint32 {
	if c.handle == nil {
		return 0
	}
	return uint32(C.llama_go_context_n_ubatch(c.handle))
}

// SeqMax returns the maximum number of sequences
func (c *Context) SeqMax() uint32 {
	if c.handle == nil {
		return 0
	}
	return uint32(C.llama_go_context_n_seq_max(c.handle))
}

// CtxSeq returns the context size per sequence
func (c *Context) CtxSeq() uint32 {
	if c.handle == nil {
		return 0
	}
	return uint32(C.llama_go_context_n_ctx_seq(c.handle))
}

// Model returns the model associated with this context
func (c *Context) Model() *Model {
	return c.model
}

// KVCacheTypeK returns the data type used for the K cache
func (c *Context) KVCacheTypeK() GGMLType {
	return c.typeK
}

// KVCacheTypeV returns the data type used for the V cache
func (c *Context) KVCacheTypeV() GGMLType {
	return c.typeV
}

///////////////////////////////////////////////////////////////////////////////
// Note: All memory/KV cache operations (MemoryClear, MemorySeq*) are in decode.go
// Note: Performance monitoring (Perf, PerfReset) is in runtime.go

///////////////////////////////////////////////////////////////////////////////
// STATE SAVE/LOAD

// StateGetSize returns the size in bytes needed to save full context state.
func (c *Context) StateGetSize() uint64 {
	if c.handle == nil {
		return 0
	}
	return uint64(C.llama_go_state_get_size(c.handle))
}

// StateGetData copies full context state into a byte slice.
// Returns the number of bytes written.
func (c *Context) StateGetData() ([]byte, error) {
	if c.handle == nil {
		return nil, ErrInvalidContext
	}

	size := c.StateGetSize()
	if size == 0 {
		return nil, nil
	}

	data := make([]byte, size)
	written := C.llama_go_state_get_data(
		c.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(size),
	)
	if written == 0 {
		return nil, getLastError()
	}

	return data[:written], nil
}

// StateSetData restores full context state from a byte slice.
// Returns the number of bytes read, or an error if restoration failed.
func (c *Context) StateSetData(data []byte) (uint64, error) {
	if c.handle == nil {
		return 0, ErrInvalidContext
	}
	if len(data) == 0 {
		return 0, nil
	}

	read := C.llama_go_state_set_data(
		c.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
	)
	if read == 0 {
		return 0, getLastError()
	}

	return uint64(read), nil
}

// StateSaveFile saves full context state to a file.
// tokens: optional slice of tokens to save alongside the state.
// Returns true on success.
func (c *Context) StateSaveFile(path string, tokens []Token) error {
	if c.handle == nil {
		return ErrInvalidContext
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokPtr *C.int32_t
	var nTokens C.size_t
	if len(tokens) > 0 {
		tokPtr = (*C.int32_t)(unsafe.Pointer(&tokens[0]))
		nTokens = C.size_t(len(tokens))
	}

	success := C.llama_go_state_save_file(c.handle, cPath, tokPtr, nTokens)
	if !success {
		return getLastError()
	}
	return nil
}

// StateLoadFile loads full context state from a file.
// If maxTokens > 0, also loads tokens into the returned slice.
// Returns tokens and error.
func (c *Context) StateLoadFile(path string, maxTokens int) ([]Token, error) {
	if c.handle == nil {
		return nil, ErrInvalidContext
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokens []Token
	var tokPtr *C.int32_t
	var nTokensOut C.size_t

	if maxTokens > 0 {
		tokens = make([]Token, maxTokens)
		tokPtr = (*C.int32_t)(unsafe.Pointer(&tokens[0]))
	}

	success := C.llama_go_state_load_file(c.handle, cPath, tokPtr, C.size_t(maxTokens), &nTokensOut)
	if !success {
		return nil, getLastError()
	}

	if maxTokens > 0 && nTokensOut > 0 {
		return tokens[:nTokensOut], nil
	}
	return nil, nil
}

// StateSeqGetSize returns the size in bytes needed to save a sequence's state.
func (c *Context) StateSeqGetSize(seqID int32) uint64 {
	if c.handle == nil {
		return 0
	}
	return uint64(C.llama_go_state_seq_get_size(c.handle, C.int32_t(seqID)))
}

// StateSeqGetData copies a sequence's state into a byte slice.
// Returns the number of bytes written.
func (c *Context) StateSeqGetData(seqID int32) ([]byte, error) {
	if c.handle == nil {
		return nil, ErrInvalidContext
	}

	size := c.StateSeqGetSize(seqID)
	if size == 0 {
		return nil, nil
	}

	data := make([]byte, size)
	written := C.llama_go_state_seq_get_data(
		c.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(size),
		C.int32_t(seqID),
	)
	if written == 0 {
		return nil, getLastError()
	}

	return data[:written], nil
}

// StateSeqSetData restores a sequence's state from a byte slice.
// Returns the number of bytes read, or an error if restoration failed.
func (c *Context) StateSeqSetData(seqID int32, data []byte) (uint64, error) {
	if c.handle == nil {
		return 0, ErrInvalidContext
	}
	if len(data) == 0 {
		return 0, nil
	}

	read := C.llama_go_state_seq_set_data(
		c.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		C.int32_t(seqID),
	)
	if read == 0 {
		return 0, getLastError()
	}

	return uint64(read), nil
}

// StateSeqSaveFile saves a sequence's state to a file.
// tokens: optional slice of tokens to save alongside the state.
// Returns the number of bytes written (0 on failure).
func (c *Context) StateSeqSaveFile(path string, seqID int32, tokens []Token) (uint64, error) {
	if c.handle == nil {
		return 0, ErrInvalidContext
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokPtr *C.int32_t
	var nTokens C.size_t
	if len(tokens) > 0 {
		tokPtr = (*C.int32_t)(unsafe.Pointer(&tokens[0]))
		nTokens = C.size_t(len(tokens))
	}

	written := C.llama_go_state_seq_save_file(c.handle, cPath, C.int32_t(seqID), tokPtr, nTokens)
	if written == 0 {
		return 0, getLastError()
	}
	return uint64(written), nil
}

// StateSeqLoadFile loads a sequence's state from a file.
// If maxTokens > 0, also loads tokens into the returned slice.
// Returns tokens read, bytes read, and error.
func (c *Context) StateSeqLoadFile(path string, seqID int32, maxTokens int) ([]Token, uint64, error) {
	if c.handle == nil {
		return nil, 0, ErrInvalidContext
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokens []Token
	var tokPtr *C.int32_t
	var nTokensOut C.size_t

	if maxTokens > 0 {
		tokens = make([]Token, maxTokens)
		tokPtr = (*C.int32_t)(unsafe.Pointer(&tokens[0]))
	}

	bytesRead := C.llama_go_state_seq_load_file(c.handle, cPath, C.int32_t(seqID), tokPtr, C.size_t(maxTokens), &nTokensOut)
	if bytesRead == 0 {
		return nil, 0, getLastError()
	}

	if maxTokens > 0 && nTokensOut > 0 {
		return tokens[:nTokensOut], uint64(bytesRead), nil
	}
	return nil, uint64(bytesRead), nil
}
