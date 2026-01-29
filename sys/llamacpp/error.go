package llamacpp

/*
#include "error.h"
*/
import "C"
import "errors"

///////////////////////////////////////////////////////////////////////////////
// ERRORS

var (
	ErrInvalidModel   = errors.New("invalid model")
	ErrInvalidContext = errors.New("invalid context")
	ErrNoKVSlot       = errors.New("no KV cache slot available")
	ErrBatchFull      = errors.New("batch is full")
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
