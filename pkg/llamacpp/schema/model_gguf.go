//go:build !client

package schema

import (
	// Packages
	gguf "github.com/mutablelogic/go-llama/sys/gguf"
)

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
