package httphandler

import (
	"errors"
	"net/http"

	// Packages
	llama "github.com/mutablelogic/go-llama"
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	httpresponse "github.com/mutablelogic/go-server/pkg/httpresponse"
	types "github.com/mutablelogic/go-server/pkg/types"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type HTTPMiddlewareFuncs []func(http.HandlerFunc) http.HandlerFunc

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// RegisterHandlers registers all llama HTTP handlers on the provided
// router with the given path prefix. The Llama instance must be non-nil.
func RegisterHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	RegisterModelHandlers(router, prefix, llamaInstance, middleware)
	RegisterCompletionHandlers(router, prefix, llamaInstance, middleware)
	RegisterEmbedHandlers(router, prefix, llamaInstance, middleware)
	RegisterTokenizerHandlers(router, prefix, llamaInstance, middleware)
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func (w HTTPMiddlewareFuncs) Wrap(handler http.HandlerFunc) http.HandlerFunc {
	if len(w) == 0 {
		return handler
	}
	for i := len(w) - 1; i >= 0; i-- {
		handler = w[i](handler)
	}
	return handler
}

func joinPath(prefix, path string) string {
	return types.JoinPath(prefix, path)
}

// httperr converts pkg errors to appropriate HTTP errors.
// Returns the original error if it's already an httpresponse.Err,
// otherwise maps pkg errors to their HTTP equivalents.
func httperr(err error) error {
	if err == nil {
		return nil
	}

	// If already an HTTP error, return as-is
	var httpErr httpresponse.Err
	if errors.As(err, &httpErr) {
		return err
	}

	// Map known errors to HTTP equivalents
	switch {
	case errors.Is(err, llama.ErrNotFound):
		return httpresponse.ErrNotFound.With(err.Error())
	case errors.Is(err, llama.ErrModelNotLoaded):
		// Model exists but not loaded - this is a conflict, not a bad request
		return httpresponse.ErrConflict.With(err.Error())
	case errors.Is(err, llama.ErrInvalidArgument),
		errors.Is(err, llama.ErrInvalidModel),
		errors.Is(err, llama.ErrInvalidContext),
		errors.Is(err, llama.ErrInvalidToken),
		errors.Is(err, llama.ErrInvalidBatch),
		errors.Is(err, llama.ErrBatchFull),
		errors.Is(err, llama.ErrNoKVSlot):
		return httpresponse.ErrBadRequest.With(err.Error())
	default:
		return httpresponse.ErrInternalError.With(err.Error())
	}
}
