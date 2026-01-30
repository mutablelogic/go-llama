package llamacpp

/*
#include "error.h"
*/
import "C"
import (
	"errors"

	llama "github.com/mutablelogic/go-llama"
)

///////////////////////////////////////////////////////////////////////////////
// ERRORS

var (
	ErrInvalidModel    = llama.ErrInvalidModel
	ErrInvalidContext  = llama.ErrInvalidContext
	ErrInvalidArgument = llama.ErrInvalidArgument
	ErrNoKVSlot        = llama.ErrNoKVSlot
	ErrBatchFull       = llama.ErrBatchFull
	ErrInvalidBatch    = llama.ErrInvalidBatch
	ErrKeyNotFound     = llama.ErrKeyNotFound
	ErrIndexOutOfRange = llama.ErrIndexOutOfRange
	ErrInvalidToken    = llama.ErrInvalidToken
)

///////////////////////////////////////////////////////////////////////////////
// HELPERS

func getLastError() error {
	cErr := C.llama_go_last_error()
	if cErr == nil {
		return errors.New("unknown error")
	}
	err := errors.New(C.GoString(cErr))
	C.llama_go_clear_error()
	return err
}
