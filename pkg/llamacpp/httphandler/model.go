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

// RegisterModelHandlers registers HTTP handlers for Model operations
func RegisterModelHandlers(router *http.ServeMux, prefix string, llamaInstance *llamacpp.Llama, middleware HTTPMiddlewareFuncs) {
	// GET /model - list all models
	// POST /model - load a model
	router.HandleFunc(joinPath(prefix, "model"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			_ = modelList(w, r, llamaInstance)
		case http.MethodPost:
			_ = modelLoad(w, r, llamaInstance)
		default:
			_ = httpresponse.Error(w, httpresponse.Err(http.StatusMethodNotAllowed), r.Method)
		}
	}))

	// GET /model/{id} - get a specific model
	// DELETE /model/{id} - unload a specific model
	router.HandleFunc(joinPath(prefix, "model/{id}"), middleware.Wrap(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			_ = modelGet(w, r, llamaInstance)
		case http.MethodDelete:
			_ = modelUnload(w, r, llamaInstance)
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

// modelLoad handles POST /model requests to load a specific model
func modelLoad(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	var req schema.LoadModelRequest
	if err := httprequest.Read(r, &req); err != nil {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With(err.Error()))
	}

	if req.Name == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model name is required"))
	}

	model, err := llamaInstance.LoadModel(r.Context(), req)
	if err != nil {
		return httpresponse.Error(w, httperr(err))
	}
	return httpresponse.JSON(w, http.StatusOK, httprequest.Indent(r), model)
}

// modelUnload handles DELETE /model/{id} requests to unload a specific model
func modelUnload(w http.ResponseWriter, r *http.Request, llamaInstance *llamacpp.Llama) error {
	id := r.PathValue("id")
	if id == "" {
		return httpresponse.Error(w, httpresponse.ErrBadRequest.With("model id is required"))
	}

	if _, err := llamaInstance.UnloadModel(r.Context(), id); err != nil {
		return httpresponse.Error(w, httperr(err))
	}
	return httpresponse.JSON(w, http.StatusNoContent, httprequest.Indent(r), nil)
}