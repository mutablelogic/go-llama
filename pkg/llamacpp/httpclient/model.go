package httpclient

import (
	"context"
	"fmt"
	"net/http"

	// Packages
	client "github.com/mutablelogic/go-client"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// ListModels returns a list of all available models from the llama API.
func (c *Client) ListModels(ctx context.Context) ([]*schema.CachedModel, error) {
	req := client.NewRequest()

	// Perform request
	var response []*schema.CachedModel
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("model")); err != nil {
		return nil, err
	}

	// Return the response
	return response, nil
}

// GetModel retrieves a specific model by its ID.
func (c *Client) GetModel(ctx context.Context, id string) (*schema.CachedModel, error) {
	if id == "" {
		return nil, fmt.Errorf("model id cannot be empty")
	}

	req := client.NewRequest()

	// Perform request
	var response schema.CachedModel
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("model", id)); err != nil {
		return nil, err
	}

	// Return the response
	return &response, nil
}

// LoadModel loads a model into memory with the given options.
func (c *Client) LoadModel(ctx context.Context, name string, opts ...Opt) (*schema.CachedModel, error) {
	if name == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.LoadModelRequest{
		Name:   name,
		Gpu:    o.Gpu,
		Layers: o.Layers,
		Mmap:   o.Mmap,
		Mlock:  o.Mlock,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Perform request
	var response schema.CachedModel
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("model")); err != nil {
		return nil, err
	}

	// Return the response
	return &response, nil
}

// UnloadModel unloads a model from memory.
func (c *Client) UnloadModel(ctx context.Context, id string) error {
	if id == "" {
		return fmt.Errorf("model id cannot be empty")
	}

	req := client.NewRequestEx(http.MethodDelete, "")

	// Perform request - expect 204 No Content
	if err := c.DoWithContext(ctx, req, nil, client.OptPath("model", id)); err != nil {
		return err
	}

	return nil
}
