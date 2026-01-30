package httpclient

import (
	"context"
	"fmt"

	// Packages
	client "github.com/mutablelogic/go-client"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Tokenize converts text into tokens using the specified model.
//
// Example:
//
//	result, err := client.Tokenize(ctx, "llama-7b", "Hello, world!")
func (c *Client) Tokenize(ctx context.Context, model, text string, opts ...Opt) (*schema.TokenizeResponse, error) {
	if model == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.TokenizeRequest{
		Model:        model,
		Text:         text,
		AddSpecial:   o.AddSpecial,
		ParseSpecial: o.ParseSpecial,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Perform request
	var response schema.TokenizeResponse
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("tokenize")); err != nil {
		return nil, err
	}

	return &response, nil
}

// Detokenize converts tokens back into text using the specified model.
//
// Example:
//
//	result, err := client.Detokenize(ctx, "llama-7b", tokens)
func (c *Client) Detokenize(ctx context.Context, model string, tokens []llamacpp.Token, opts ...Opt) (*schema.DetokenizeResponse, error) {
	if model == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.DetokenizeRequest{
		Model:          model,
		Tokens:         tokens,
		RemoveSpecial:  o.RemoveSpecial,
		UnparseSpecial: o.UnparseSpecial,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Perform request
	var response schema.DetokenizeResponse
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("detokenize")); err != nil {
		return nil, err
	}

	return &response, nil
}
