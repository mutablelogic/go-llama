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

// RegisterChatHandlers registers HTTP handlers for Chat operations
func RegisterChatHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	router.HandleFunc(joinPath(prefix, "chat"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = chatCreate(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// chatCreate handles POST /chat requests to generate chat responses
func chatCreate(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.ChatRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("failed to read request"), err.Error())
	}

	if req.Model == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model is required"))
	}

	if len(req.Messages) == 0 {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("messages are required"))
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
			}
			defer stream.Close()
		}
	}

	var err error
	var result *schema.ChatResponse

	// Execute chat
	if stream != nil {
		result, err = llamaInstance.Chat(r.Context(), req, func(chunk schema.ChatChunk) error {
			stream.Write(schema.CompletionStreamDeltaType, chunk)
			return nil
		})
	} else {
		result, err = llamaInstance.Chat(r.Context(), req, nil)
	}

	if err != nil {
		if stream != nil {
			stream.Write(schema.CompletionStreamErrorType, err.Error())
			return nil
		}
		return httpresponse.Error(w, httperr(err))
	}

	if stream != nil {
		stream.Write(schema.CompletionStreamDoneType, result)
		return nil
	}

	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), result)
}
