package httphandler

import (
	"net/http"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	httprequest "github.com/mutablelogic/go-server/pkg/httprequest"
	httpresponse "github.com/mutablelogic/go-server/pkg/httpresponse"
	types "github.com/mutablelogic/go-server/pkg/types"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// RegisterCompletionHandlers registers HTTP handlers for Completion operations
func RegisterCompletionHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	router.HandleFunc(joinPath(prefix, "completion"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = completionCreate(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// completionCreate handles POST /completion requests to generate text
func completionCreate(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.CompletionRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("failed to read request"), err.Error())
	}

	if req.Model == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model is required"))
	}

	// Create text stream if requested
	var stream *httpresponse.TextStream
	if accept := r.Header.Get(types.ContentAcceptHeader); accept != "" {
		mimetype, err := types.ParseContentType(accept)
		if err != nil {
			return httpresponse.Error(w, httpresponse.ErrBadRequest.With("invalid Accept header"), err.Error())
		}
		if mimetype == types.ContentTypeTextStream {
			stream = httpresponse.NewTextStream(w)
			if stream == nil {
				return httpresponse.Error(w, httpresponse.ErrInternalError.With("cannot create text stream"))































}	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), result)	}		return nil		stream.Write(schema.CompletionStreamDoneType, result)	if stream != nil {	}		return httpresponse.Error(w, httperr(err))		}			return nil			stream.Write(schema.CompletionStreamErrorType, err.Error())		if stream != nil {	if err != nil {	}		result, err = llamaInstance.Complete(r.Context(), req, nil)	} else {		})			return nil			stream.Write(schema.CompletionStreamDeltaType, chunk)		result, err = llamaInstance.Complete(r.Context(), req, func(chunk schema.CompletionChunk) error {	if stream != nil {	var err error	var result *schema.CompletionResponse	}		}			defer stream.Close()			}