package llamacpp

import (
	"context"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	"go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Tokenize converts text to tokens using the specified model.
// Loads the model if not already cached.
func (l *Llama) Tokenize(ctx context.Context, req schema.TokenizeRequest) (result *schema.TokenizeResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Tokenize"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	err = l.WithModel(ctx, schema.LoadModelRequest{Name: req.Model}, func(ctx context.Context, task *Task) error {
		// Lock the model - tokenization is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		// Build tokenize options
		opts := llamacpp.DefaultTokenizeOptions()
		if req.AddSpecial != nil {
			opts.AddSpecial = *req.AddSpecial
		}
		if req.ParseSpecial != nil {
			opts.ParseSpecial = *req.ParseSpecial
		}

		// Tokenize the text
		tokens, err := task.Model().Tokenize(req.Text, opts)
		if err != nil {
			return err
		}

		result = &schema.TokenizeResponse{
			Tokens: tokens,
		}
		return nil
	})
	return
}

// Detokenize converts tokens back to text using the specified model.
// Loads the model if not already cached.
func (l *Llama) Detokenize(ctx context.Context, req schema.DetokenizeRequest) (result *schema.DetokenizeResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Detokenize"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	err = l.WithModel(ctx, schema.LoadModelRequest{Name: req.Model}, func(ctx context.Context, task *Task) error {
		// Lock the model - detokenization is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		// Build detokenize options
		opts := llamacpp.DefaultDetokenizeOptions()
		if req.RemoveSpecial != nil {
			opts.RemoveSpecial = *req.RemoveSpecial
		}
		if req.UnparseSpecial != nil {
			opts.UnparseSpecial = *req.UnparseSpecial
		}

		// Detokenize the tokens
		text, err := task.Model().Detokenize(req.Tokens, opts)
		if err != nil {
			return err
		}

		result = &schema.DetokenizeResponse{
			Text: text,
		}
		return nil
	})
	return
}
