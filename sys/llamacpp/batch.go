package llamacpp

/*
#include "batch.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Batch manages tokens to be processed by the model
type Batch struct {
	handle *C.llama_go_batch
}

///////////////////////////////////////////////////////////////////////////////
// BATCH LIFECYCLE

// NewBatch creates a new batch that can hold up to nTokens tokens.
// nSeqMax is the maximum number of sequence IDs per token (usually 1).
func NewBatch(nTokens, nSeqMax int32) (*Batch, error) {
	handle := C.llama_go_batch_init(C.int32_t(nTokens), C.int32_t(nSeqMax))
	if handle == nil {
		return nil, getLastError()
	}

	b := &Batch{handle: handle}
	runtime.SetFinalizer(b, func(b *Batch) {
		b.Close()
	})

	return b, nil
}

// Close frees the batch resources
func (b *Batch) Close() error {
	if b.handle != nil {
		C.llama_go_batch_free(b.handle)
		b.handle = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// BATCH OPERATIONS

// Clear resets the batch to empty (n_tokens = 0)
func (b *Batch) Clear() {
	if b.handle != nil {
		C.llama_go_batch_clear(b.handle)
	}
}

// NumTokens returns the current number of tokens in the batch
func (b *Batch) NumTokens() int32 {
	if b.handle == nil {
		return 0
	}
	return int32(C.llama_go_batch_n_tokens(b.handle))
}

// Capacity returns the maximum number of tokens this batch can hold
func (b *Batch) Capacity() int32 {
	if b.handle == nil {
		return 0
	}
	return int32(C.llama_go_batch_capacity(b.handle))
}

// Add adds a single token to the batch.
// pos is the position in the sequence.
// seqID is the sequence ID.
// logits indicates whether to compute logits for this token.
// Returns ErrInvalidBatch if batch is closed, ErrBatchFull if batch is full.
func (b *Batch) Add(token Token, pos int32, seqID int32, logits bool) error {
	if b.handle == nil {
		return ErrInvalidBatch
	}
	if !bool(C.llama_go_batch_add(b.handle, C.int32_t(token), C.int32_t(pos), C.int32_t(seqID), C.bool(logits))) {
		return ErrBatchFull
	}
	return nil
}

// AddSeq adds a single token with multiple sequence IDs.
// This is useful for shared prefixes across multiple sequences.
// Returns ErrInvalidBatch if batch is closed or seqIDs is empty, ErrBatchFull if batch is full.
func (b *Batch) AddSeq(token Token, pos int32, seqIDs []int32, logits bool) error {
	if b.handle == nil {
		return ErrInvalidBatch
	}
	if len(seqIDs) == 0 {
		return ErrInvalidBatch
	}
	if !bool(C.llama_go_batch_add_seq(
		b.handle,
		C.int32_t(token),
		C.int32_t(pos),
		(*C.int32_t)(unsafe.Pointer(&seqIDs[0])),
		C.int32_t(len(seqIDs)),
		C.bool(logits),
	)) {
		return ErrBatchFull
	}
	return nil
}

// AddTokens adds multiple tokens to the batch, all with the same sequence ID.
// posStart is the starting position for the first token.
// If logitsLast is true, only the last token will have logits computed.
// Returns the number of tokens actually added (may be less if batch fills up), or error.
// Returns ErrInvalidBatch if batch is closed.
func (b *Batch) AddTokens(tokens []Token, posStart int32, seqID int32, logitsLast bool) (int32, error) {
	if b.handle == nil {
		return 0, ErrInvalidBatch
	}
	if len(tokens) == 0 {
		return 0, nil
	}
	added := int32(C.llama_go_batch_add_tokens(
		b.handle,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens)),
		C.int32_t(posStart),
		C.int32_t(seqID),
		C.bool(logitsLast),
	))
	return added, nil
}

// SetLogits sets whether logits should be computed for the token at the given index.
// Returns ErrInvalidBatch if batch is closed.
func (b *Batch) SetLogits(idx int32, logits bool) error {
	if b.handle == nil {
		return ErrInvalidBatch
	}
	C.llama_go_batch_set_logits(b.handle, C.int32_t(idx), C.bool(logits))
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// DECODE/ENCODE

// Decode processes the batch using the given context.
// This runs the model forward pass and populates the KV cache.
// Returns nil on success.
// Returns ErrNoKVSlot if no KV cache slot is available (try smaller batch or larger context).
func (b *Batch) Decode(ctx *Context) error {
	if b.handle == nil {
		return ErrInvalidContext
	}
	if ctx == nil || ctx.handle == nil {
		return ErrInvalidContext
	}

	result := C.llama_go_batch_decode(ctx.handle, b.handle)
	switch result {
	case 0:
		return nil
	case 1:
		return ErrNoKVSlot
	default:
		return getLastError()
	}
}

// Encode processes the batch using the encoder (for encoder-decoder models).
// Returns nil on success.
func (b *Batch) Encode(ctx *Context) error {
	if b.handle == nil {
		return ErrInvalidContext
	}
	if ctx == nil || ctx.handle == nil {
		return ErrInvalidContext
	}

	result := C.llama_go_batch_encode(ctx.handle, b.handle)
	if result < 0 {
		return getLastError()
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS

// BatchFromTokens creates a batch pre-populated with the given tokens.
// All tokens are assigned to sequence 0 with sequential positions starting from posStart.
// If logitsLast is true, only the last token will have logits computed.
func BatchFromTokens(tokens []Token, posStart int32, seqID int32, logitsLast bool) (*Batch, error) {
	b, err := NewBatch(int32(len(tokens)), 1)
	if err != nil {
		return nil, err
	}

	added, err := b.AddTokens(tokens, posStart, seqID, logitsLast)
	if err != nil {
		b.Close()
		return nil, err
	}
	if added != int32(len(tokens)) {
		b.Close()
		return nil, ErrBatchFull
	}

	return b, nil
}
