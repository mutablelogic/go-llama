//go:build !client

package llamacpp

import (
	"context"
	"fmt"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	attribute "go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Chat generates a response for the given chat messages.
// If onChunk is provided, it will be called for each generated token.
// The callback can stop generation early by returning an error.
func (l *Llama) Chat(ctx context.Context, req schema.ChatRequest, onChunk func(schema.ChatChunk) error) (result *schema.ChatResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Chat"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	// Create a context, and run the chat completion
	err = l.WithContext(ctx, schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: req.Model,
		},
	}, func(ctx context.Context, task *Task) error {
		// Lock the model - chat is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		prompt, err := buildChatPrompt(task.Model(), req)
		if err != nil {
			return err
		}

		opts := buildCompletionOptions(ctx, req.CompletionRequest)
		var callbackErr error
		if onChunk != nil {
			opts.OnToken = func(token string) bool {
				if callbackErr != nil {
					return false
				}
				if err := onChunk(schema.ChatChunk{Message: schema.ChatMessage{Role: "assistant", Content: token}}); err != nil {
					callbackErr = err
					return false
				}
				return true
			}
		}

		text, stopWordHit, err := task.Context().CompleteNativeWithStopInfo(prompt, opts)
		if err != nil {
			return err
		}
		if callbackErr != nil {
			return callbackErr
		}

		usage, err := completionUsage(task.Model(), prompt, text)
		if err != nil {
			return err
		}
		finishReason := completionFinishReason(req.CompletionRequest, text, usage, stopWordHit)

		result = &schema.ChatResponse{
			Model: req.Model,
			Message: schema.ChatMessage{
				Role:    "assistant",
				Content: text,
			},
			Usage:        usage,
			FinishReason: finishReason,
		}
		return nil
	})
	return
}

///////////////////////////////////////////////////////////////////////////////
// HELPERS

func buildChatPrompt(model *llamacpp.Model, req schema.ChatRequest) (string, error) {
	if model == nil {
		return "", fmt.Errorf("model is required")
	}

	messages := make([]llamacpp.ChatMessage, 0, len(req.Messages)+1)
	if req.Prompt != "" {
		messages = append(messages, llamacpp.ChatMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}
	for _, msg := range req.Messages {
		messages = append(messages, llamacpp.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	if len(messages) == 0 {
		return "", fmt.Errorf("no chat messages provided")
	}
	if !model.HasChatTemplate() {
		return "", fmt.Errorf("model has no chat template")
	}

	return llamacpp.ApplyTemplateWithModel(model, "", messages, true)
}
