package llamacpp

import (
	"context"
	"path/filepath"
	"time"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	store "github.com/mutablelogic/go-llama/pkg/llamacpp/store"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	attribute "go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// PullCallback defines the callback function signature for progress updates during model downloads
type PullCallback func(filename string, bytes_received uint64, total_bytes uint64)

// LoadModel loads a model into memory with the given parameters.
// Returns a CachedModel with the model handle and load timestamp.
// If the model is already cached, returns the existing cached model.
func (l *Llama) LoadModel(ctx context.Context, req schema.LoadModelRequest) (result *schema.CachedModel, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("LoadModel"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	l.Lock()
	defer l.Unlock()

	// Get model from store
	model, err := l.Store.GetModel(ctx, req.Name)
	if err != nil {
		return nil, err
	}

	// Check if already cached
	if cached, ok := l.cached[model.Path]; ok {
		return cached, nil
	}

	// Build params with defaults for nil values
	params := llamacpp.DefaultModelParams()
	if req.Layers != nil {
		params.NGPULayers = *req.Layers
	}
	if req.Gpu != nil {
		params.MainGPU = *req.Gpu
	}
	if req.Mmap != nil {
		params.UseMmap = *req.Mmap
	}
	if req.Mlock != nil {
		params.UseMlock = *req.Mlock
	}

	// Load the model
	handle, err := llamacpp.LoadModel(filepath.Join(l.Store.Path(), model.Path), params)
	if err != nil {
		return nil, err
	}

	// Return the cached model
	cached := &schema.CachedModel{
		Model:    *model,
		LoadedAt: time.Now(),
		Handle:   handle,
	}
	l.cached[model.Path] = cached
	return cached, nil
}

// ListModels returns all models in the store as CachedModel structures.
// Models that are loaded have their LoadedAt timestamp and Handle set.
// Models that are not loaded have zero timestamp and nil Handle.
func (l *Llama) ListModels(ctx context.Context) (result []*schema.CachedModel, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("ListModels"))
	defer func() { endSpan(err) }()

	l.RLock()
	defer l.RUnlock()

	// Get all models from store
	models, err := l.Store.ListModels(ctx)
	if err != nil {
		return nil, err
	}

	// Transform to CachedModel, using cached version if available
	result = make([]*schema.CachedModel, 0, len(models))
	for _, m := range models {
		if cached, ok := l.cached[m.Path]; ok {
			result = append(result, cached)
		} else {
			result = append(result, &schema.CachedModel{
				Model: *m,
			})
		}
	}

	return result, nil
}

// GetModel returns a model by name as a CachedModel.
// If the model is loaded, returns the cached version with LoadedAt and Handle.
// If not loaded, returns a CachedModel with zero timestamp and nil Handle.
func (l *Llama) GetModel(ctx context.Context, name string) (result *schema.CachedModel, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("GetModel"),
		attribute.String("request", name),
	)
	defer func() { endSpan(err) }()

	l.RLock()
	defer l.RUnlock()

	// Get model from store
	model, err := l.Store.GetModel(ctx, name)
	if err != nil {
		return nil, err
	}

	// Return cached version if available
	if cached, ok := l.cached[model.Path]; ok {
		return cached, nil
	}

	// Return uncached model
	return &schema.CachedModel{
		Model: *model,
	}, nil
}

// UnloadModel unloads a model from memory and removes it from the cache.
// Returns the model (now uncached with zero timestamp) and any error.
func (l *Llama) UnloadModel(ctx context.Context, name string) (result *schema.CachedModel, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("UnloadModel"),
		attribute.String("request", name),
	)
	defer func() { endSpan(err) }()

	l.Lock()
	defer l.Unlock()

	// Get model from store
	model, err := l.Store.GetModel(ctx, name)
	if err != nil {
		return nil, err
	}

	// Check if cached
	cached, ok := l.cached[model.Path]
	if !ok {
		// Not loaded, just return uncached model
		return &schema.CachedModel{
			Model: *model,
		}, nil
	}

	// Close the handle
	if cached.Handle != nil {
		cached.Handle.Close()
	}

	// Remove from cache
	delete(l.cached, model.Path)

	// Return uncached model (zero timestamp, nil handle)
	return &schema.CachedModel{
		Model: *model,
	}, nil
}

// PullModel downloads a model from the given URL and returns the cached model.
func (l *Llama) PullModel(ctx context.Context, req schema.PullModelRequest, fn PullCallback) (result *schema.CachedModel, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("PullModel"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	// Download the model using the store
	model, err := l.Store.PullModel(ctx, req.URL, store.ClientCallback(fn))
	if err != nil {
		return nil, err
	}

	// Return the cached model
	return &schema.CachedModel{
		Model: *model,
	}, nil
}
