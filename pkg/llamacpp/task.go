package llamacpp

import (
	"context"

	// Packages

	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Task provides access to a loaded model and optionally a context for
// inference operations. Tasks are created via WithModel or WithContext
// and are valid only within the callback function.
type Task struct {
	model *schema.CachedModel
	ctx   *llamacpp.Context
}

// TaskFunc is a callback function that receives a context and Task.
type TaskFunc func(context.Context, *Task) error

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS - TASK ACCESSORS

// Model returns the underlying llamacpp Model handle.
func (t *Task) Model() *llamacpp.Model {
	if t.model == nil {
		return nil
	}
	return t.model.Handle
}

// Context returns the underlying llamacpp Context, or nil if this task
// was created with WithModel (no context).
func (t *Task) Context() *llamacpp.Context {
	return t.ctx
}

// CachedModel returns the full CachedModel metadata.
func (t *Task) CachedModel() *schema.CachedModel {
	return t.model
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS - WITH PATTERN

// WithModel loads a model (if not already cached) and calls the function
// with a Task containing the model. The model remains loaded after the
// callback returns. Use this for operations that only need the model
// (e.g., tokenization, metadata access).
//
// Thread-safety: The callback is responsible for acquiring the model's
// mutex if needed. Use task.CachedModel().Lock()/Unlock() for operations
// that are not thread-safe (most llama.cpp operations).
func (l *Llama) WithModel(ctx context.Context, req schema.LoadModelRequest, fn TaskFunc) (err error) {
	// Load or get cached model
	cached, err := l.LoadModel(ctx, req)
	if err != nil {
		return err
	}

	// Create task and call function
	task := &Task{
		model: cached,
	}
	return fn(ctx, task)
}

// WithContext loads a model (if not already cached), creates an inference
// context, and calls the function with a Task containing both. The context
// is freed after the callback returns, but the model remains loaded.
// Use this for operations that need a context (e.g., completion, embeddings).
//
// Thread-safety: The callback is responsible for acquiring the model's
// mutex if needed. Use task.CachedModel().Lock()/Unlock() for operations
// that are not thread-safe (most llama.cpp operations).
func (l *Llama) WithContext(ctx context.Context, req schema.ContextRequest, fn TaskFunc) (err error) {
	// Load or get cached model
	cached, err := l.LoadModel(ctx, req.LoadModelRequest)
	if err != nil {
		return err
	}

	// Build context params with defaults for nil values
	params := llamacpp.DefaultContextParams()
	if req.ContextSize != nil {
		params.NCtx = *req.ContextSize
	} else if cached.Model.ContextSize > 0 {
		params.NCtx = uint32(cached.Model.ContextSize)
	}
	if req.BatchSize != nil {
		params.NBatch = *req.BatchSize
	}
	if req.UBatchSize != nil {
		params.NUBatch = *req.UBatchSize
	}
	if req.Threads != nil {
		params.NThreads = *req.Threads
		params.NThreadsBatch = *req.Threads
	}
	if req.AttentionType != nil {
		params.AttentionType = llamacpp.AttentionType(*req.AttentionType)
	}
	if req.FlashAttn != nil {
		params.FlashAttn = llamacpp.FlashAttnType(*req.FlashAttn)
	}
	if req.Embeddings != nil {
		params.Embeddings = *req.Embeddings
	}
	if req.KVUnified != nil {
		params.KVUnified = *req.KVUnified
	}

	// Create context - this is expensive (allocates KV cache)
	llmCtx, err := llamacpp.NewContext(cached.Handle, params)
	if err != nil {
		return err
	}
	defer llmCtx.Close()

	// Create task and call function
	task := &Task{
		model: cached,
		ctx:   llmCtx,
	}
	return fn(ctx, task)
}
