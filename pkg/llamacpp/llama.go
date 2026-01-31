package llamacpp

import (
	"context"
	"fmt"
	"sync"

	// Packages
	"github.com/mutablelogic/go-client"
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	store "github.com/mutablelogic/go-llama/pkg/llamacpp/store"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Llama is a singleton that manages the llama.cpp runtime and models
type Llama struct {
	sync.RWMutex
	opt
	*store.Store
	cached map[string]*schema.CachedModel
}

///////////////////////////////////////////////////////////////////////////////
// GLOBALS

var (
	instance *Llama
	once     sync.Once
)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// New creates or returns the singleton Llama instance
func New(path string, opts ...Opt) (*Llama, error) {
	// Initialize llama.cpp runtime
	var result error
	once.Do(func() {
		if err := llamacpp.Init(); err != nil {
			result = fmt.Errorf("failed to initialize llama.cpp: %w", err)
			return
		}
	})
	if result != nil {
		return nil, result
	} else {
		instance = &Llama{
			cached: make(map[string]*schema.CachedModel),
		}
	}

	// Create a model store
	if store, err := store.New(path, client.OptTracer(instance.tracer)); err != nil {
		return nil, err
	} else {
		instance.Store = store
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(&instance.opt); err != nil {
			return nil, err
		}
	}

	// Return success
	return instance, nil
}

// Close releases all resources and cleans up the llama.cpp runtime
func (l *Llama) Close() error {
	if l == nil {
		return nil
	}

	// Lock the instance
	l.Lock()
	defer l.Unlock()

	// Unload all cached models
	for path, cached := range l.cached {
		if cached.Handle != nil {
			cached.Handle.Close()
		}
		delete(l.cached, path)
	}

	// Cleanup llama.cpp runtime
	llamacpp.Cleanup()

	// Set instance to nil
	instance = nil

	// Return success
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// GPUInfo returns information about available GPU devices and the backend.
func (l *Llama) GPUInfo(ctx context.Context) *schema.GPUInfo {
	_, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("GPUInfo"))
	defer func() { endSpan(nil) }()

	// Get device list from low-level API
	sysDevices := llamacpp.GPUList()

	// Convert to schema types
	devices := make([]schema.GPUDevice, len(sysDevices))
	for i, d := range sysDevices {
		devices[i] = schema.GPUDevice{
			ID:               d.DeviceID,
			Name:             d.DeviceName,
			FreeMemoryBytes:  d.FreeMemoryBytes,
			TotalMemoryBytes: d.TotalMemoryBytes,
		}
	}

	return &schema.GPUInfo{
		Backend: llamacpp.GPUBackendName(),
		Devices: devices,
	}
}
