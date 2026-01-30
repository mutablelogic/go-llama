package llamacpp

import (
	"context"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	llama "github.com/mutablelogic/go-llama"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	attribute "go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Embed generates embeddings for one or more texts.
// Loads the model if not already cached, creates a context with embeddings enabled,
// and computes embeddings for all input texts in a single batch.
func (l *Llama) Embed(ctx context.Context, req schema.EmbedRequest) (result *schema.EmbedResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Embed"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	// Build context request for embedding models:
	// - Embeddings enabled: required to extract embeddings
	embeddings := true
	contextReq := schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: req.Model,
		},
		Embeddings: &embeddings,
	}

	err = l.WithContext(ctx, contextReq, func(ctx context.Context, task *Task) error {
		// Lock the model - embedding computation is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		if task.Context().PoolingType() == llamacpp.PoolingNone {
			return llama.ErrNotEmbeddingModel
		}

		// Build embedding options
		opts := llamacpp.DefaultEmbeddingOptions()
		if req.Normalize != nil {
			opts.Normalize = *req.Normalize
		}

		// Compute embeddings for all inputs
		batch, err := task.Context().ComputeEmbeddings(task.Model(), req.Input, opts)
		if err != nil {
			return err
		}

		// Count total input tokens
		var inputTokens int
		for _, text := range req.Input {
			tokOpts := llamacpp.DefaultTokenizeOptions()
			tokens, err := task.Model().Tokenize(text, tokOpts)
			if err != nil {
				return err
			}
			inputTokens += len(tokens)
		}

		result = &schema.EmbedResponse{
			Model:      req.Model,
			Embeddings: batch.Embeddings,
			Dimension:  batch.Dimension,
			Usage: schema.Usage{
				InputTokens:  inputTokens,
				OutputTokens: 0,
			},
		}
		return nil
	})
	return
}
