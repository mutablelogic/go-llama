package store

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	// Packages
	llama "github.com/mutablelogic/go-llama"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	gguf "github.com/mutablelogic/go-llama/sys/gguf"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Store manages a collection of GGUF models in a directory.
type Store struct {
	sync.RWMutex
	path string
}

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// New creates new model storage at the given path.
// The path must be an existing directory.
func New(path string) (*Store, error) {
	// Check path exists and is a directory
	if info, err := os.Stat(path); err != nil {
		return nil, llama.ErrOpenFailed.Withf("%s: %v", path, err)
	} else if !info.IsDir() {
		return nil, llama.ErrInvalidArgument.Withf("not a directory: %s", path)
	} else {
		path = filepath.Clean(path)
	}

	return &Store{path: path}, nil
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

// GetModel returns a model by name. It matches against the full relative path
// or the filename (last element of the path). Returns ErrNotFound if not found.
func (s *Store) GetModel(ctx context.Context, name string) (*schema.Model, error) {
	models, err := s.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	for _, m := range models {
		if m.Path == name || filepath.Base(m.Path) == name {
			return m, nil
		}
	}
	return nil, llama.ErrNotFound.Withf("%s", name)
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func (s *Store) loadModel(path string) (*schema.Model, error) {
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
