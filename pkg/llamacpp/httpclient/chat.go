package httpclient

import (
	"context"
	"fmt"

	// Packages
	client "github.com/mutablelogic/go-client"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Chat generates a response for the given chat messages.
// Use WithChunkCallback to receive streaming chunks as they are generated.
//
// Example:
//
//	result, err := client.Chat(ctx, "llama-7b",
//	    schema.ChatMessage{Role: "user", Content: "What is 2+2?"},
//	    httpclient.WithMaxTokens(100),
//	    httpclient.WithTemperature(0.7))
func (c *Client) Chat(ctx context.Context, model string, messages []schema.ChatMessage, opts ...Opt) (*schema.ChatResponse, error) {
	if model == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("messages cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	var systemPrompt string
	if o.System != nil {
		systemPrompt = *o.System
	}

	reqBody := schema.ChatRequest{
		CompletionRequest: schema.CompletionRequest{
			Model:         model,
			Prompt:        systemPrompt,
			MaxTokens:     o.MaxTokens,
			Temperature:   o.Temperature,
			TopP:          o.TopP,
			TopK:          o.TopK,
			RepeatPenalty: o.RepeatPenalty,
			RepeatLastN:   o.RepeatLastN,
			Seed:          o.Seed,
			Stop:          o.Stop,
			PrefixCache:   o.PrefixCache,
		},
		Messages: messages,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Set up request options
	reqOpts := []client.RequestOpt{client.OptPath("chat")}

	// If streaming callback provided, handle streaming
	var response schema.ChatResponse
	if o.chatChunkCallback != nil || o.chunkCallback != nil {
		reqOpts = append(reqOpts, client.OptReqHeader("Accept", "text/event-stream"))
		reqOpts = append(reqOpts, client.OptTextStreamCallback(func(evt client.TextStreamEvent) error {
			switch evt.Event {
			case schema.CompletionStreamDeltaType:
				var chunk schema.ChatChunk
				if err := evt.Json(&chunk); err != nil {
					return err
				}
				if o.chatChunkCallback != nil {
					return o.chatChunkCallback(&chunk)
				}
				if o.chunkCallback != nil {
					// Convert ChatChunk to CompletionChunk for callback compatibility
					completionChunk := schema.CompletionChunk{Text: chunk.Message.Content}
					return o.chunkCallback(&completionChunk)
				}
				return nil
			case schema.CompletionStreamDoneType:
				// Parse final response
				if err := evt.Json(&response); err != nil {
					return fmt.Errorf("failed to parse chat response: %w", err)
				}
			case schema.CompletionStreamErrorType:
				return fmt.Errorf("chat error: %s", evt.Data)
			}
			return nil
		}))
	}

	// Perform request
	if err := c.DoWithContext(ctx, req, &response, reqOpts...); err != nil {
		return nil, err
	}

	return &response, nil
}
