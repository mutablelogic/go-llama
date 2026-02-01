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

// RegisterModelHandlers registers HTTP handlers for Model operations
func RegisterModelHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	// GET /model - list all models
	// POST /model - pull (download) a model from URL
	router.HandleFunc(joinPath(prefix, "model"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			_ = modelList(w, r, llamaInstance)
		case http.MethodPost:
			_ = modelPull(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))

	// GET /model/{id} - get a specific model
	// POST /model/{id} - load/unload a model by id
	// DELETE /model/{id} - delete a specific model from disk
	router.HandleFunc(joinPath(prefix, "model/{id...}"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			_ = modelGet(w, r, llamaInstance)
		case http.MethodPost:
			_ = modelLoadUnload(w, r, llamaInstance)
		case http.MethodDelete:
			_ = modelDelete(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// modelList handles GET /model requests to list all available models
func modelList(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	models, err := llamaInstance.ListModels(r.Context())
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}
	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), models)
}

// modelGet handles GET /model/{id} requests to retrieve a specific model
func modelGet(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	id := r.PathValue("id")
	if id == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model id is required"))
	}

	model, err := llamaInstance.GetModel(r.Context(), id)
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}
	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), model)
}

// modelPull handles POST /model requests to download a model from URL
func modelPull(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.PullModelRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With(err.Error()))
	}

	if req.URL == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model URL is required"))
	}

	// Check Accept header for streaming progress
	var stream *httpresponse.TextStream
	if accept := r.Header.Get("Accept"); accept != "" {
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

	// Stream progress updates using TextStream
	var progressCallback llamacpp.PullCallback
	if stream != nil {
		progressCallback = func(filename string, bytesReceived, totalBytes uint64) error {
			var percentage float64
			if totalBytes > 0 {
				percentage = float64(bytesReceived) * 100.0 / float64(totalBytes)
			}
			stream.Write(schema.ModelPullProgressType, schema.ModelPullProgress{
				Filename:      filename,
				BytesReceived: bytesReceived,
				TotalBytes:    totalBytes,
				Percentage:    percentage,
			})
			return nil
		}
	}

	// Perform the model pull
	model, err := llamaInstance.PullModel(r.Context(), req, progressCallback)
	if err != nil {
		if stream != nil {
			// Send error via TextStream
			stream.Write(schema.ModelPullErrorType, map[string]string{"error": err.Error()})
			return nil
		}
		return httpresponse.Error(w, httperr(err))
	}

	if stream != nil {
		// Send completion event via TextStream
		stream.Write(schema.ModelPullCompleteType, model)
		return nil
	}

	// Return success
	return httpresponse.JSON(w, http.StatusCreated, httprequest.Indent(r), model)
}

// modelLoadUnload handles POST /model/{id} requests to load or unload a specific model by id
func modelLoadUnload(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.LoadModelRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With(err.Error()))
	}
	req.Name = r.PathValue("id")

	// Check if this is an unload request
	isUnload := req.Load != nil && !*req.Load
	if isUnload {
		if model, err := llamaInstance.UnloadModel(r.Context(), req.Name); err != nil {
			return httpresponse.Error(w, httperr(err))
		} else {
			return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), model)
		}
	} else if model, err := llamaInstance.LoadModel(r.Context(), req); err != nil {
		return httpresponse.Error(w, httperr(err))
	} else {
		return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), model)
	}
}

// modelDelete handles DELETE /model/{id} requests to delete a specific model from disk
func modelDelete(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	id := r.PathValue("id")
	if id == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model id is required"))
	}

	if err := llamaInstance.DeleteModel(r.Context(), id); err != nil {
		return httpresponse.Error(w, httperr(err))
	}
	return httpresponse.JSON(w, http.StatusNoContent, httprequest.Indent(r), nil)
}
