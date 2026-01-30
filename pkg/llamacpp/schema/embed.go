package schema

///////////////////////////////////////////////////////////////////////////////
// TYPES

// EmbedRequest contains parameters for generating embeddings.
type EmbedRequest struct {
	Model     string   `json:"model"`               // Model name
	Input     []string `json:"input"`               // Text(s) to embed
	Normalize *bool    `json:"normalize,omitempty"` // L2-normalize embeddings (default: true)
}

// EmbedResponse contains the generated embeddings.
type EmbedResponse struct {
	Model      string      `json:"model"`      // Model used
	Embeddings [][]float32 `json:"embeddings"` // One embedding vector per input
	Dimension  int         `json:"dimension"`  // Embedding dimension
	Usage      Usage       `json:"usage"`      // Token usage
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r EmbedRequest) String() string {
	return stringify(r)
}

func (r EmbedResponse) String() string {
	return stringify(r)
}
