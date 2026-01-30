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

// Complete generates a text completion for the given prompt.
// Use WithChunkCallback to receive streaming chunks as they are generated.
//
// Example:
//
//	result, err := client.Complete(ctx, "llama-7b", "Once upon a time",
//	    httpclient.WithMaxTokens(100),
//	    httpclient.WithTemperature(0.7))
func (c *Client) Complete(ctx context.Context, model, prompt string, opts ...Opt) (*schema.CompletionResponse, error) {
	if model == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.CompletionRequest{
		Model:       model,
		Prompt:      prompt,
		MaxTokens:   o.MaxTokens,
		Temperature: o.Temperature,
		TopP:        o.TopP,
		TopK:        o.TopK,
		Seed:        o.Seed,
		Stop:        o.Stop,
		PrefixCache: o.PrefixCache,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Set up request options
	reqOpts := []client.RequestOpt{client.OptPath("completion")}

	// If streaming callback provided, handle streaming
	var response schema.CompletionResponse
	if o.chunkCallback != nil {
		reqOpts = append(reqOpts, client.OptReqHeader("Accept", "text/event-stream"))
		reqOpts = append(reqOpts, client.OptTextStreamCallback(func(evt client.TextStreamEvent) error {
			switch evt.Event {
			case schema.CompletionStreamDeltaType:
				var chunk schema.CompletionChunk
				if err := evt.Json(&chunk); err != nil {
					return err
				}
				return o.chunkCallback(&chunk)
			case schema.CompletionStreamDoneType:
				// Parse final response
				if err := evt.Json(&response); err != nil {
					return fmt.Errorf("failed to parse completion response: %w", err)
				}
			case schema.CompletionStreamErrorType:
				return fmt.Errorf("completion error: %s", evt.Data)
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
