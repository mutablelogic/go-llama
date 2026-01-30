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

// Embed generates embeddings for the given input texts.
//
// Example:
//
//	result, err := client.Embed(ctx, "embedding-model", []string{"Hello", "World"})
func (c *Client) Embed(ctx context.Context, model string, input []string, opts ...Opt) (*schema.EmbedResponse, error) {
	if model == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.EmbedRequest{
		Model:     model,
		Input:     input,
		Normalize: o.Normalize,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Perform request
	var response schema.EmbedResponse
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("embed")); err != nil {
		return nil, err
	}

	return &response, nil
}
