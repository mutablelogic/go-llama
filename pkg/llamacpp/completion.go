package llamacpp

import (
	"context"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	attribute "go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Complete generates a completion for the given prompt.
// If onChunk is provided, it will be called for each generated token.
// The callback can stop generation early by returning an error.
func (l *Llama) Complete(ctx context.Context, req schema.CompletionRequest, onChunk func(schema.CompletionChunk) error) (result *schema.CompletionResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Complete"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	// Create a context, and run the completion
	err = l.WithContext(ctx, schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: req.Model,
		},
	}, func(ctx context.Context, task *Task) error {
		// Lock the model - completion is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		opts := buildCompletionOptions(ctx, req)
		var callbackErr error
		if onChunk != nil {
			opts.OnToken = func(token string) bool {
				if callbackErr != nil {
					return false
				}
				if err := onChunk(schema.CompletionChunk{Text: token}); err != nil {
					callbackErr = err
					return false
				}
				return true
			}
		}

		text, err := task.Context().CompleteNative(req.Prompt, opts)
		if err != nil {
			return err
		}
		if callbackErr != nil {
			return callbackErr
		}

		usage, err := completionUsage(task.Model(), req.Prompt, text)
		if err != nil {
			return err
		}

		result = &schema.CompletionResponse{
			Model: req.Model,
			Text:  text,
			Usage: usage,
		}
		return nil
	})
	return
}

///////////////////////////////////////////////////////////////////////////////
// HELPERS

func buildCompletionOptions(ctx context.Context, req schema.CompletionRequest) llamacpp.CompletionOptions {
	opts := llamacpp.DefaultCompletionOptions()

	if req.MaxTokens != nil {
		opts.MaxTokens = int(*req.MaxTokens)
	}
	if req.Stop != nil {
		opts.StopWords = req.Stop
	}
	if req.PrefixCache != nil {
		opts.EnablePrefixCaching = *req.PrefixCache
	}

	params := opts.SamplerParams
	if req.Seed != nil {
		params.Seed = *req.Seed
	}
	if req.Temperature != nil {
		params.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		params.TopP = *req.TopP
	}
	if req.TopK != nil {
		params.TopK = *req.TopK
	}
	opts.SamplerParams = params
	if ctx != nil {
		opts.AbortContext = ctx
	}

	return opts
}

func completionUsage(model *llamacpp.Model, prompt, text string) (schema.Usage, error) {
	if model == nil {
		return schema.Usage{}, nil
	}

	promptOpts := llamacpp.DefaultTokenizeOptions()
	promptOpts.AddSpecial = true
	promptTokens, err := model.Tokenize(prompt, promptOpts)
	if err != nil {
		return schema.Usage{}, err
	}

	outputTokens := 0
	if text != "" {
		outOpts := llamacpp.DefaultTokenizeOptions()
		outOpts.AddSpecial = false
		outTokens, err := model.Tokenize(text, outOpts)
		if err != nil {
			return schema.Usage{}, err
		}
		outputTokens = len(outTokens)
	}

	return schema.Usage{
		InputTokens:  len(promptTokens),
		OutputTokens: outputTokens,
	}, nil
}
