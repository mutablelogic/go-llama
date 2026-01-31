package store

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	// Packages
	client "github.com/mutablelogic/go-client"
	llama "github.com/mutablelogic/go-llama"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	gguf "github.com/mutablelogic/go-llama/sys/gguf"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Store manages a collection of GGUF models in a directory.
type Store struct {
	sync.RWMutex
	path   string
	client *Client
}

type PullCallbackFunc func(filename string, bytes_received uint64, total_bytes uint64)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// New creates new model storage at the given path.
// The path must be an existing directory.
func New(path string, opts ...client.ClientOpt) (*Store, error) {
	// Check path exists and is a directory
	if info, err := os.Stat(path); err != nil {
		return nil, llama.ErrOpenFailed.Withf("%s: %v", path, err)
	} else if !info.IsDir() {
		return nil, llama.ErrInvalidArgument.Withf("not a directory: %s", path)
	} else {
		path = filepath.Clean(path)
	}

	// Create client for downloading models
	client, err := NewClient(opts...)
	if err != nil {
		return nil, llama.ErrOpenFailed.Withf("failed to create client: %v", err)
	}

	return &Store{path: path, client: client}, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Path returns the store's root directory path.
func (s *Store) Path() string {
	return s.path
}

// ListModels scans the store directory for GGUF models and returns their metadata.
// Hidden files and directories are skipped. Invalid GGUF files are silently skipped.
func (s *Store) ListModels(ctx context.Context) ([]*schema.Model, error) {
	s.RLock()
	defer s.RUnlock()
	return s.listModels(ctx)
}

// GetModel returns a model by name. It matches against the full relative path
// or the filename (last element of the path). Returns ErrNotFound if not found.
func (s *Store) GetModel(ctx context.Context, name string) (*schema.Model, error) {
	s.RLock()
	defer s.RUnlock()
	return s.getModel(ctx, name)
}

// DeleteModel deletes a model from the store by name. It matches against the full relative path,
// filename, or model name. Returns ErrNotFound if the model doesn't exist.
func (s *Store) DeleteModel(ctx context.Context, name string) error {
	s.Lock()
	defer s.Unlock()

	// Find the model first using internal unlocked method
	model, err := s.getModel(ctx, name)
	if err != nil {
		return err
	}

	// Delete the file
	if err := os.Remove(filepath.Join(s.path, model.Path)); err != nil {
		return llama.ErrOpenFailed.Withf("failed to delete model: %v", err)
	}

	// Return success
	return nil
}

// PullModel downloads a model from the given URL into the store and returns the loaded model.
// Supports HuggingFace URLs with hf:// scheme and regular HTTP(S) URLs.
// The callback receives progress updates during download.
func (s *Store) PullModel(ctx context.Context, url string, callback ClientCallback) (*schema.Model, error) {
	// Get the suggested destination path first
	destPath, err := s.client.GetDestPath(url)
	if err != nil {
		return nil, err
	}

	// Determine final file path in the store
	finalPath := filepath.Join(s.path, destPath)

	// Create a wrapper callback that checks if file exists with matching size
	var skipDownload bool
	wrappedCallback := func(filename string, bytesReceived, totalBytes uint64) error {
		// On first call (bytesReceived == 0), check if we can skip download
		if bytesReceived == 0 {
			if info, err := os.Stat(finalPath); err == nil {
				if uint64(info.Size()) == totalBytes {
					// File exists with matching size, skip download
					skipDownload = true
					return llama.ErrInvalidArgument.With("file already exists with correct size")
				}
			}
		}
		// Forward to user callback if provided
		if callback != nil {
			return callback(filename, bytesReceived, totalBytes)
		}
		return nil
	}

	// Create temporary file in the store directory for the download
	tempFile, err := os.CreateTemp(s.path, ".gguf-*.tmp")
	if err != nil {
		return nil, llama.ErrOpenFailed.Withf("failed to create temporary file: %v", err)
	}
	tempPath := tempFile.Name()
	defer func() {
		tempFile.Close()
		os.Remove(tempPath)
	}()

	// Download the model and get the suggested destination path
	destPath, err = s.client.PullModel(ctx, tempFile, url, wrappedCallback)
	if err != nil && skipDownload {
		// Download was skipped because file exists, load and return it
		return s.loadModel(finalPath)
	} else if err != nil {
		return nil, err // Return the original error from PullModel
	}

	// Recalculate final path (should be same as before)
	finalPath = filepath.Join(s.path, destPath)

	// Ensure the directory exists
	if err := os.MkdirAll(filepath.Dir(finalPath), 0755); err != nil {
		return nil, llama.ErrOpenFailed.Withf("failed to create directory: %v", err)
	}

	// Check if file already exists
	if _, err := os.Stat(finalPath); err == nil {
		return nil, llama.ErrInvalidArgument.Withf("model already exists at %s", destPath)
	}

	// Validate download
	if info, err := os.Stat(tempPath); err != nil {
		return nil, llama.ErrNotFound.Withf("temporary file not found after download: %v", err)
	} else if info.Size() == 0 {
		return nil, llama.ErrInvalidModel.With("downloaded file is empty")
	}

	// Move temporary file to final location (synchronized)
	s.Lock()
	defer s.Unlock()
	if _, err := os.Stat(finalPath); err == nil {
		return nil, llama.ErrInvalidArgument.Withf("model already exists at %s", destPath)
	}
	if err := os.Rename(tempPath, finalPath); err != nil {
		return nil, llama.ErrOpenFailed.Withf("failed to move file to final location: %v", err)
	}

	// Load the model and return it
	model, err := s.loadModel(finalPath)
	if err != nil {
		// If loadModel fails, clean up the downloaded file
		os.Remove(finalPath)
		return nil, err
	}

	return model, nil
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// listModels is the internal unlocked version of ListModels
func (s *Store) listModels(ctx context.Context) ([]*schema.Model, error) {
	// Walk the directory looking for .gguf files
	var models []*schema.Model
	err := filepath.Walk(s.path, func(path string, info os.FileInfo, err error) error {
		// Check for context cancellation, or other errors
		if ctx.Err() != nil {
			return ctx.Err()
		} else if err != nil {
			return err
		}

		// Skip hidden files and directories
		if strings.HasPrefix(info.Name(), ".") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		} else if info.IsDir() {
			return nil
		} else if filepath.Ext(path) != gguf.FileExtension {
			return nil
		}

		// Try to load model metadata, skip if invalid
		if model, err := s.loadModel(path); err == nil && model != nil {
			models = append(models, model)
		}

		// Return success
		return nil
	})

	// Sort models by path for predictable ordering
	sort.Slice(models, func(i, j int) bool {
		return models[i].Path < models[j].Path
	})

	// Return models and any error
	return models, err
}

// getModel is the internal unlocked version of GetModel
func (s *Store) getModel(ctx context.Context, name string) (*schema.Model, error) {
	models, err := s.listModels(ctx)
	if err != nil {
		return nil, err
	}
	for _, m := range models {
		if m.Path == name || filepath.Base(m.Path) == name || m.Name == name {
			return m, nil
		}
	}

	// If no match found and name has no extension, try with .gguf suffix
	if filepath.Ext(name) == "" {
		nameWithExt := name + ".gguf"
		for _, m := range models {
			if m.Path == nameWithExt || filepath.Base(m.Path) == nameWithExt || m.Name == nameWithExt {
				return m, nil
			}
		}
	}

	return nil, llama.ErrNotFound.Withf("%s", name)
}

func (s *Store) loadModel(path string) (*schema.Model, error) {
	// Open the model GGUF file
	ctx, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer ctx.Close()

	// Compute relative path from store root
	relPath, err := filepath.Rel(s.path, path)
	if err != nil {
		relPath = filepath.Base(path)
	}
	return schema.NewModelFromGGUF(s.path, relPath, ctx)
}
