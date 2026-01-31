package schema

import (
	"sync"
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

// PullModelRequest contains the parameters for downloading a model from a URL.
type PullModelRequest struct {
	URL string `json:"url"` // URL to download the model from (supports hf:// and https://)
}

// CachedModel represents a model that has been loaded into memory.
// The embedded RWMutex provides thread-safety for operations on this model.
type CachedModel struct {
	sync.RWMutex
	Model
	LoadedAt time.Time       `json:"loaded_at"`
	Handle   *llamacpp.Model `json:"-"`
}

// ContextRequest contains parameters for creating an inference context.
type ContextRequest struct {
	LoadModelRequest
	ContextSize   *uint32 `json:"context_size,omitempty"`   // Context size (nil = from model)
	BatchSize     *uint32 `json:"batch_size,omitempty"`     // Logical batch size (nil = default)
	UBatchSize    *uint32 `json:"ubatch_size,omitempty"`    // Physical/micro batch size (nil = default, must equal batch_size for encoder models)
	Threads       *int32  `json:"threads,omitempty"`        // Number of threads (nil = default)
	AttentionType *int32  `json:"attention_type,omitempty"` // Attention type: -1=auto, 0=causal, 1=non-causal (nil = auto)
	FlashAttn     *int32  `json:"flash_attn,omitempty"`     // Flash attention: -1=auto, 0=disabled, 1=enabled (nil = auto)
	Embeddings    *bool   `json:"embeddings,omitempty"`     // Enable embeddings extraction (nil = false)
	KVUnified     *bool   `json:"kv_unified,omitempty"`     // Use unified KV cache (nil = default, required for BERT)
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r LoadModelRequest) String() string {
	return stringify(r)
}

func (r PullModelRequest) String() string {
	return stringify(r)
}

func (m CachedModel) String() string {
	return stringify(m)
}

func (r ContextRequest) String() string {
	return stringify(r)
}
