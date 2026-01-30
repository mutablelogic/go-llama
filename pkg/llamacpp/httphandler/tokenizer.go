package httphandler

import (
	"net/http"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	httprequest "github.com/mutablelogic/go-server/pkg/httprequest"
	httpresponse "github.com/mutablelogic/go-server/pkg/httpresponse"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// RegisterTokenizerHandlers registers HTTP handlers for Tokenize/Detokenize operations
func RegisterTokenizerHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	router.HandleFunc(joinPath(prefix, "tokenize"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = tokenizeCreate(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))

	router.HandleFunc(joinPath(prefix, "detokenize"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = detokenizeCreate(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// tokenizeCreate handles POST /tokenize requests to tokenize text
func tokenizeCreate(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.TokenizeRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("failed to read request"), err.Error())
	}

	if req.Model == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model is required"))
	}

	result, err := llamaInstance.Tokenize(r.Context(), req)
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}

	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), result)
}

// detokenizeCreate handles POST /detokenize requests to convert tokens to text
func detokenizeCreate(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.DetokenizeRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("failed to read request"), err.Error())
	}

	if req.Model == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model is required"))
	}

	result, err := llamaInstance.Detokenize(r.Context(), req)
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}

	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), result)
}
