package schema

///////////////////////////////////////////////////////////////////////////////
// TYPES

// CompletionRequest contains parameters for text completion.
type CompletionRequest struct {
	Model       string   `json:"model"`                  // Model name
	Prompt      string   `json:"prompt"`                 // Prompt to complete
	MaxTokens   *int32   `json:"max_tokens,omitempty"`   // Max tokens to generate
	Temperature *float32 `json:"temperature,omitempty"`  // Sampling temperature
	TopP        *float32 `json:"top_p,omitempty"`        // Nucleus sampling
	TopK        *int32   `json:"top_k,omitempty"`        // Top-k sampling
	Seed        *uint32  `json:"seed,omitempty"`         // RNG seed
	Stop        []string `json:"stop,omitempty"`         // Stop words
	PrefixCache *bool    `json:"prefix_cache,omitempty"` // Enable prefix caching
}

// CompletionResponse contains the generated completion.
type CompletionResponse struct {
	Model string `json:"model"` // Model used
	Text  string `json:"text"`  // Completion text
	Usage Usage  `json:"usage"` // Token usage
}

// CompletionChunk contains a streamed completion chunk.
type CompletionChunk struct {
	Text string `json:"text"` // Chunk text
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r CompletionRequest) String() string {
	return stringify(r)
}

func (r CompletionResponse) String() string {
	return stringify(r)
}

func (r CompletionChunk) String() string {
	return stringify(r)
}
