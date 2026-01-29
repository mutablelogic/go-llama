package llamacpp

/*
#include "decode.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// LOGITS

// GetLogits returns the logits for the token at the given index after a decode.
// Use idx=-1 to get logits for the last token that had logits=true in the batch.
// Returns a slice of n_vocab floats representing log-probabilities for each token.
func (ctx *Context) GetLogits(idx int32) ([]float32, error) {
	if ctx.handle == nil {
		return nil, ErrInvalidContext
	}

	logits := C.llama_go_get_logits(ctx.handle, C.int32_t(idx))
	if logits == nil {
		return nil, getLastError()
	}

	nVocab := int(C.llama_go_ctx_n_vocab(ctx.handle))
	if nVocab <= 0 {
		return nil, ErrInvalidContext
	}

	// Create a Go slice that references the C array (no copy)
	// This is safe as long as we don't use the slice after ctx is freed
	return unsafe.Slice((*float32)(unsafe.Pointer(logits)), nVocab), nil
}

// NVocab returns the vocabulary size
func (ctx *Context) NVocab() int32 {
	if ctx.handle == nil {
		return 0
	}
	return int32(C.llama_go_ctx_n_vocab(ctx.handle))
}

///////////////////////////////////////////////////////////////////////////////
// EMBEDDINGS

// GetEmbeddings returns the embeddings for the token at the given index.
// Use idx=-1 to get embeddings for the last token.
// Only available when context was created with embeddings=true.
func (ctx *Context) GetEmbeddings(idx int32) ([]float32, error) {
	if ctx.handle == nil {
		return nil, ErrInvalidContext
	}

	embd := C.llama_go_get_embeddings(ctx.handle, C.int32_t(idx))
	if embd == nil {
		return nil, getLastError()
	}

	nEmbd := int(C.llama_go_n_embd(ctx.handle))
	if nEmbd <= 0 {
		return nil, ErrInvalidContext
	}

	return unsafe.Slice((*float32)(unsafe.Pointer(embd)), nEmbd), nil
}

// NEmbd returns the embedding dimension
func (ctx *Context) NEmbd() int32 {
	if ctx.handle == nil {
		return 0
	}
	return int32(C.llama_go_n_embd(ctx.handle))
}

///////////////////////////////////////////////////////////////////////////////
// MEMORY (KV CACHE) MANAGEMENT

// ErrPartialRemovalNotSupported is returned when partial KV cache removal is not supported.
var ErrPartialRemovalNotSupported = errors.New("partial memory removal not supported")

// MemoryClear clears the memory (KV cache) contents.
// If clearData is true, the data buffers will also be cleared.
// Returns ErrInvalidContext if context is closed.
func (ctx *Context) MemoryClear(clearData bool) error {
	if ctx.handle == nil {
		return ErrInvalidContext
	}
	C.llama_go_memory_clear(ctx.handle, C.bool(clearData))
	return nil
}

// MemorySeqRm removes tokens from memory for a sequence.
// seqID: sequence ID (-1 for all sequences)
// p0: start position (inclusive, -1 for 0)
// p1: end position (exclusive, -1 for end)
// Returns ErrInvalidContext if context is closed, ErrPartialRemovalNotSupported if partial removal is not supported.
func (ctx *Context) MemorySeqRm(seqID, p0, p1 int32) error {
	if ctx.handle == nil {
		return ErrInvalidContext
	}
	if !bool(C.llama_go_memory_seq_rm(ctx.handle, C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1))) {
		return ErrPartialRemovalNotSupported
	}
	return nil
}

// MemorySeqCp copies a sequence in memory.
// Returns ErrInvalidContext if context is closed.
func (ctx *Context) MemorySeqCp(seqIDSrc, seqIDDst, p0, p1 int32) error {
	if ctx.handle == nil {
		return ErrInvalidContext
	}
	C.llama_go_memory_seq_cp(ctx.handle, C.int32_t(seqIDSrc), C.int32_t(seqIDDst), C.int32_t(p0), C.int32_t(p1))
	return nil
}

// MemorySeqAdd shifts positions in memory (for context shifting).
// Returns ErrInvalidContext if context is closed.
func (ctx *Context) MemorySeqAdd(seqID, p0, p1, delta int32) error {
	if ctx.handle == nil {
		return ErrInvalidContext
	}
	C.llama_go_memory_seq_add(ctx.handle, C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1), C.int32_t(delta))
	return nil
}

// MemorySeqPosMin returns the minimum position for a sequence (-1 if empty)
func (ctx *Context) MemorySeqPosMin(seqID int32) int32 {
	if ctx.handle == nil {
		return -1
	}
	return int32(C.llama_go_memory_seq_pos_min(ctx.handle, C.int32_t(seqID)))
}

// MemorySeqPosMax returns the maximum position for a sequence (-1 if empty)
func (ctx *Context) MemorySeqPosMax(seqID int32) int32 {
	if ctx.handle == nil {
		return -1
	}
	return int32(C.llama_go_memory_seq_pos_max(ctx.handle, C.int32_t(seqID)))
}

// MemorySeqKeep removes all tokens that do not belong to the specified sequence.
func (ctx *Context) MemorySeqKeep(seqID int32) {
	if ctx.handle == nil {
		return
	}
	C.llama_go_memory_seq_keep(ctx.handle, C.int32_t(seqID))
}

// MemorySeqDiv divides positions in a sequence range [p0, p1) by d (integer division).
func (ctx *Context) MemorySeqDiv(seqID, p0, p1, d int32) {
	if ctx.handle == nil {
		return
	}
	C.llama_go_memory_seq_div(ctx.handle, C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1), C.int32_t(d))
}

// MemoryCanShift returns whether the memory supports context shifting
func (ctx *Context) MemoryCanShift() bool {
	if ctx.handle == nil {
		return false
	}
	return bool(C.llama_go_memory_can_shift(ctx.handle))
}

// MemorySeqLength returns the number of tokens cached for a sequence.
// Returns 0 if the sequence is empty.
func (ctx *Context) MemorySeqLength(seqID int32) int32 {
	min := ctx.MemorySeqPosMin(seqID)
	max := ctx.MemorySeqPosMax(seqID)
	if min < 0 || max < 0 {
		return 0
	}
	return max - min + 1
}

///////////////////////////////////////////////////////////////////////////////
// SYNCHRONIZATION

// Synchronize waits for all GPU operations to complete.
// Returns ErrInvalidContext if context is closed.
func (ctx *Context) Synchronize() error {
	if ctx.handle == nil {
		return ErrInvalidContext
	}
	C.llama_go_synchronize(ctx.handle)
	return nil
}
