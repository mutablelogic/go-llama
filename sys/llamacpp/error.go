package llamacpp

/*
#include "error.h"
*/
import "C"
import "errors"

///////////////////////////////////////////////////////////////////////////////
// ERRORS

var (
	ErrInvalidModel    = errors.New("invalid model")
	ErrInvalidContext  = errors.New("invalid context")
	ErrInvalidArgument = errors.New("invalid argument")
	ErrNoKVSlot        = errors.New("no KV cache slot available")
	ErrBatchFull       = errors.New("batch is full")
	ErrInvalidBatch    = errors.New("invalid batch")
	ErrKeyNotFound     = errors.New("metadata key not found")
	ErrIndexOutOfRange = errors.New("index out of range")
	ErrInvalidToken    = errors.New("invalid token")
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
