package schema

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
// STRINGIFY

func (m Model) String() string {
	return stringify(m)
}
