package llama

import "fmt"

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Error represents an error code
type Error int

///////////////////////////////////////////////////////////////////////////////
// ERROR CODES

const (
	ErrSuccess Error = iota
	ErrInvalidContext
	ErrInvalidModel
	ErrInvalidArgument
	ErrIndexOutOfRange
	ErrKeyNotFound
	ErrTypeMismatch
	ErrInvalidToken
	ErrInvalidBatch
	ErrBatchFull
	ErrNoKVSlot
	ErrOpenFailed
	ErrNotFound
)

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (e Error) String() string {
	switch e {
	case ErrSuccess:
		return "success"
	case ErrInvalidContext:
		return "invalid context"
	case ErrInvalidModel:
		return "invalid model"
	case ErrInvalidArgument:
		return "invalid argument"
	case ErrIndexOutOfRange:
		return "index out of range"
	case ErrKeyNotFound:
		return "metadata key not found"
	case ErrTypeMismatch:
		return "metadata type mismatch"
	case ErrInvalidToken:
		return "invalid token"
	case ErrInvalidBatch:
		return "invalid batch"
	case ErrBatchFull:
		return "batch is full"
	case ErrNoKVSlot:
		return "no KV cache slot available"
	case ErrOpenFailed:
		return "failed to open file"
	case ErrNotFound:
		return "not found"
	default:
		return fmt.Sprintf("error(%d)", int(e))
	}
}

///////////////////////////////////////////////////////////////////////////////
// ERROR INTERFACE

func (e Error) Error() string {
	return e.String()
}

///////////////////////////////////////////////////////////////////////////////
// WRAPPING

// With returns a new error wrapping this error with additional context
func (e Error) With(msg string) error {
	return fmt.Errorf("%w: %s", e, msg)
}

// Withf returns a new error wrapping this error with formatted context
func (e Error) Withf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", e, fmt.Sprintf(format, args...))
}
