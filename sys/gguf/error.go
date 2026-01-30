package gguf

import llama "github.com/mutablelogic/go-llama"

///////////////////////////////////////////////////////////////////////////////
// ERRORS

var (
	ErrInvalidContext  = llama.ErrInvalidContext
	ErrInvalidArgument = llama.ErrInvalidArgument
	ErrKeyNotFound     = llama.ErrKeyNotFound
	ErrIndexOutOfRange = llama.ErrIndexOutOfRange
	ErrTypeMismatch    = llama.ErrTypeMismatch
	ErrOpenFailed      = llama.ErrOpenFailed
)
