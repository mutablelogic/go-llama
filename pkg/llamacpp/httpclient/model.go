package httpclient

import (
	"context"
	"encoding/json"
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

	// Perform request to the correct endpoint
	var response schema.CachedModel
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("model", name)); err != nil {
		return nil, err
	}

	// Return the response
	return &response, nil
}

// PullModel downloads and caches a model from a URL.
// Use WithProgressCallback to receive download progress updates.
//
// Example:
//
//	model, err := client.PullModel(ctx, "hf://microsoft/DialoGPT-medium",
//	    httpclient.WithProgressCallback(func(filename string, received, total uint64) error {
//	        if total > 0 {
//	            pct := float64(received) * 100.0 / float64(total)
//	            fmt.Printf("Downloading %s: %.1f%%\n", filename, pct)
//	        }
//	        return nil
//	    }))
func (c *Client) PullModel(ctx context.Context, url string, opts ...Opt) (*schema.CachedModel, error) {
	if url == "" {
		return nil, fmt.Errorf("model URL cannot be empty")
	}

	// Apply options
	o, err := applyOpts(opts...)
	if err != nil {
		return nil, err
	}

	// Build request body
	reqBody := schema.PullModelRequest{
		URL: url,
	}

	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Set up request options
	reqOpts := []client.RequestOpt{client.OptPath("model")}

	// If progress callback provided, handle streaming progress
	var response schema.CachedModel
	if o.progressCallback != nil {
		reqOpts = append(reqOpts, client.OptReqHeader("Accept", "text/event-stream"))
		reqOpts = append(reqOpts, client.OptTextStreamCallback(func(evt client.TextStreamEvent) error {
			switch evt.Event {
			case schema.ModelPullProgressType:
				var progress schema.ModelPullProgress
				if err := evt.Json(&progress); err != nil {
					return err
				}
				return o.progressCallback(progress.Filename, progress.BytesReceived, progress.TotalBytes)
			case schema.ModelPullCompleteType:
				// Parse final response
				if err := evt.Json(&response); err != nil {
					return fmt.Errorf("failed to parse model pull response: %w", err)
				}
			case schema.ModelPullErrorType:
				// Parse error JSON to extract clean error message
				var errResp struct {
					Error string `json:"error"`
				}
				if err := json.Unmarshal([]byte(evt.Data), &errResp); err == nil && errResp.Error != "" {
					return fmt.Errorf("%s", errResp.Error)
				}
				// Fall back to raw data if JSON parsing fails
				return fmt.Errorf("%s", evt.Data)
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

// UnloadModel unloads a model from memory and returns the unloaded model.
func (c *Client) UnloadModel(ctx context.Context, id string) (*schema.CachedModel, error) {
	if id == "" {
		return nil, fmt.Errorf("model id cannot be empty")
	}

	reqBody := map[string]any{"load": false}
	req, err := client.NewJSONRequest(reqBody)
	if err != nil {
		return nil, err
	}

	// Perform request - expect model response
	var response schema.CachedModel
	if err := c.DoWithContext(ctx, req, &response, client.OptPath("model", id)); err != nil {
		return nil, err
	}

	return &response, nil
}

// DeleteModel deletes a model from disk.
func (c *Client) DeleteModel(ctx context.Context, id string) error {
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
