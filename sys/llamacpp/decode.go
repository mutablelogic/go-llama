package llamacpp

/*
#include "decode.h"
*/
import "C"
import "unsafe"

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

// MemoryClear clears the memory (KV cache) contents.
// If clearData is true, the data buffers will also be cleared.
func (ctx *Context) MemoryClear(clearData bool) {
	if ctx.handle != nil {
		C.llama_go_memory_clear(ctx.handle, C.bool(clearData))
	}
}

// MemorySeqRm removes tokens from memory for a sequence.
// seqID: sequence ID (-1 for all sequences)
// p0: start position (inclusive, -1 for 0)
// p1: end position (exclusive, -1 for end)
// Returns false if partial removal is not supported.
func (ctx *Context) MemorySeqRm(seqID, p0, p1 int32) bool {
	if ctx.handle == nil {
		return false
	}
	return bool(C.llama_go_memory_seq_rm(ctx.handle, C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1)))
}

// MemorySeqCp copies a sequence in memory
func (ctx *Context) MemorySeqCp(seqIDSrc, seqIDDst, p0, p1 int32) {
	if ctx.handle != nil {
		C.llama_go_memory_seq_cp(ctx.handle, C.int32_t(seqIDSrc), C.int32_t(seqIDDst), C.int32_t(p0), C.int32_t(p1))
	}
}

// MemorySeqAdd shifts positions in memory (for context shifting)
func (ctx *Context) MemorySeqAdd(seqID, p0, p1, delta int32) {
	if ctx.handle != nil {
		C.llama_go_memory_seq_add(ctx.handle, C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1), C.int32_t(delta))
	}
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

// MemoryCanShift returns whether the memory supports context shifting
func (ctx *Context) MemoryCanShift() bool {
	if ctx.handle == nil {
		return false
	}
	return bool(C.llama_go_memory_can_shift(ctx.handle))
}

///////////////////////////////////////////////////////////////////////////////
// SYNCHRONIZATION

// Synchronize waits for all GPU operations to complete
func (ctx *Context) Synchronize() {
	if ctx.handle != nil {
		C.llama_go_synchronize(ctx.handle)
	}
}
