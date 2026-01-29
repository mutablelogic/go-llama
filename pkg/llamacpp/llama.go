package llamacpp

import (
	"fmt"
	"path/filepath"
	"sync"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Llama is a singleton that manages the llama.cpp runtime and models
type Llama struct {
	mu         sync.RWMutex
	modelsPath string
	models     map[string]*llamacpp.Model
	opts       Options
}

// Options contains configuration for the Llama instance
type Options struct {
	// Add options here - to be discussed
}

///////////////////////////////////////////////////////////////////////////////
// GLOBALS

var (
	instance *Llama
	once     sync.Once
	initErr  error
)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// New creates or returns the singleton Llama instance
func New(modelsPath string, opts ...Option) (*Llama, error) {
	once.Do(func() {
		// Initialize llama.cpp runtime
		if err := llamacpp.Init(); err != nil {
			initErr = fmt.Errorf("failed to initialize llama.cpp: %w", err)
			return
		}

		// Create instance
		instance = &Llama{
			modelsPath: modelsPath,
			models:     make(map[string]*llamacpp.Model),
			opts:       Options{},
		}

		// Apply options
		for _, opt := range opts {
			if err := opt(instance); err != nil {
				initErr = err
				return
			}
		}
	})

	if initErr != nil {
		return nil, initErr
	}

	return instance, nil
}

// Close releases all resources and cleans up the llama.cpp runtime
func (l *Llama) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Close all loaded models
	for name, model := range l.models {
		if err := model.Close(); err != nil {
			// Log error but continue cleanup
			_ = fmt.Errorf("failed to close model %s: %w", name, err)
		}
	}
	l.models = make(map[string]*llamacpp.Model)

	// Cleanup llama.cpp runtime
	llamacpp.Cleanup()

	return nil
}

///////////////////////////////////////////////////////////////////////////////
// MODEL MANAGEMENT

// LoadModel loads a model by filename from the models directory
// Returns cached model if already loaded
func (l *Llama) LoadModel(filename string, opts ...ModelOption) (*llamacpp.Model, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Check cache
	if model, ok := l.models[filename]; ok {
		return model, nil
	}

	// Load model
	path := filepath.Join(l.modelsPath, filename)
	params := llamacpp.DefaultModelParams()
	
	// Apply model-specific options
	for _, opt := range opts {
		opt(&params)
	}

	model, err := llamacpp.LoadModel(path, params)
	if err != nil {
		return nil, fmt.Errorf("failed to load model %s: %w", filename, err)
	}

	// Cache model
	l.models[filename] = model

	return model, nil
}

// UnloadModel closes and removes a model from the cache
func (l *Llama) UnloadModel(filename string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	model, ok := l.models[filename]
	if !ok {
		return fmt.Errorf("model %s not loaded", filename)
	}

	if err := model.Close(); err != nil {
		return fmt.Errorf("failed to close model %s: %w", filename, err)
	}

	delete(l.models, filename)
	return nil
}

// Models returns a list of currently loaded model filenames
func (l *Llama) Models() []string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	names := make([]string, 0, len(l.models))
	for name := range l.models {
		names = append(names, name)
	}
	return names
}

///////////////////////////////////////////////////////////////////////////////
// OPTION PATTERN

// Option configures a Llama instance
type Option func(*Llama) error

// ModelOption configures model loading parameters
type ModelOption func(*llamacpp.ModelParams)

// WithGPULayers sets the number of layers to offload to GPU
func WithGPULayers(layers int) ModelOption {
	return func(p *llamacpp.ModelParams) {
		p.NGPULayers = int32(layers)
	}
}
