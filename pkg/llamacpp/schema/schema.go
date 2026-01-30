package schema

import (
	"encoding/json"
)

////////////////////////////////////////////////////////////////////////////////
// TYPES

// Usage tracks token usage for requests.
type Usage struct {
	InputTokens  int `json:"input_tokens"`  // Tokens in input (prompt/text to embed)
	OutputTokens int `json:"output_tokens"` // Tokens generated (0 for embeddings)
}

// TotalTokens returns the sum of input and output tokens.
func (u Usage) TotalTokens() int {
	return u.InputTokens + u.OutputTokens
}

////////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func stringify[T any](v T) string {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err.Error()
	}
	return string(data)
}
