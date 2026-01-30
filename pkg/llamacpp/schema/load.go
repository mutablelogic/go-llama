package schema

import (
	"time"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// LoadModelRequest contains the parameters for loading a model into memory.
type LoadModelRequest struct {
	Name   string `json:"name"`                 // Model name or path to load
	Gpu    *int32 `json:"gpu,omitempty"`        // Main GPU index (nil = default)
	Layers *int32 `json:"gpu_layers,omitempty"` // Number of layers to offload to GPU (nil = default, -1 = all)
	Mmap   *bool  `json:"use_mmap,omitempty"`   // Use memory mapping for model loading (nil = default)
	Mlock  *bool  `json:"use_mlock,omitempty"`  // Lock model in memory (nil = default)
}

// CachedModel represents a model that has been loaded into memory.
type CachedModel struct {
	Model
	LoadedAt time.Time       `json:"loaded_at"`
	Handle   *llamacpp.Model `json:"-"`
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r LoadModelRequest) String() string {
	return stringify(r)
}

func (m CachedModel) String() string {
	return stringify(m)
}
