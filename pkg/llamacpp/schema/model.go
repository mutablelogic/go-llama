package schema

import (
	// Packges
	gguf "github.com/mutablelogic/go-llama/sys/gguf"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Model represents model metadata and capabilities (excluding load params).
type Model struct {
	// Identity
	Path         string `json:"path,omitempty"`
	Name         string `json:"name,omitempty"`
	Architecture string `json:"architecture,omitempty"`
	Description  string `json:"description,omitempty"`

	// Chat template
	ChatTemplate string `json:"chatTemplate,omitempty"`

	// Dimensions
	ContextSize   int32 `json:"contextSize,omitempty"`
	EmbeddingSize int32 `json:"embeddingSize,omitempty"`
	LayerCount    int32 `json:"layerCount,omitempty"`
	HeadCount     int32 `json:"headCount,omitempty"`
	HeadKVCount   int32 `json:"headKVCount,omitempty"`

	// Raw metadata key/value pairs from the model
	Meta map[string]any `json:"meta,omitempty"`
}

// ModelRuntime represents runtime statistics for a loaded model.
type ModelRuntime struct {
	NLayer    int32  `json:"layerCount,omitempty"`
	NHead     int32  `json:"headCount,omitempty"`
	NHeadKV   int32  `json:"headKVCount,omitempty"`
	NEmbd     int32  `json:"embeddingSize,omitempty"`
	NCtxTrain int32  `json:"contextSize,omitempty"`
	NParams   uint64 `json:"paramCount,omitempty"`
	ModelSize uint64 `json:"modelSizeBytes,omitempty"`
}

///////////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS

// NewModelFromGGUF builds a schema Model from a GGUF file context.
// The relPath is the relative path from basePath to the model file.
// This is a lightweight way to get model metadata without loading the full model.
func NewModelFromGGUF(basePath, relPath string, ctx *gguf.Context) (*Model, error) {
	if ctx == nil {
		return nil, gguf.ErrInvalidContext
	}

	arch := ctx.Architecture()
	meta, err := ctx.AllMetadata()
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:          relPath,
		Name:          ctx.Name(),
		Architecture:  arch,
		Description:   ctx.Description(),
		ChatTemplate:  ctx.ChatTemplate(),
		ContextSize:   getInt32(meta, arch+".context_length"),
		EmbeddingSize: getInt32(meta, arch+".embedding_length"),
		LayerCount:    getInt32(meta, arch+".block_count"),
		HeadCount:     getInt32(meta, arch+".attention.head_count"),
		HeadKVCount:   getInt32(meta, arch+".attention.head_count_kv"),
		Meta:          meta,
	}

	return model, nil
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE HELPERS

func getInt32(meta map[string]any, key string) int32 {
	if v, ok := meta[key]; ok {
		switch val := v.(type) {
		case int32:
			return val
		case uint32:
			return int32(val)
		case int64:
			return int32(val)
		case uint64:
			return int32(val)
		case int:
			return int32(val)
		}
	}
	return 0
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (m Model) String() string {
	return stringify(m)
}
