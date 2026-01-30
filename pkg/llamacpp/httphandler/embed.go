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

// RegisterEmbedHandlers registers HTTP handlers for Embed operations
func RegisterEmbedHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	router.HandleFunc(joinPath(prefix, "embed"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = embedCreate(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// embedCreate handles POST /embed requests to generate embeddings
func embedCreate(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.EmbedRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("failed to read request"), err.Error())
	}

	if req.Model == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model is required"))
	}

	result, err := llamaInstance.Embed(r.Context(), req)
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}

	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), result)
}
