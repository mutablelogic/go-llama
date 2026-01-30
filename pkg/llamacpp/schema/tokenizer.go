package schema

import (
	// Packages
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// TokenizeRequest contains parameters for tokenizing text.
type TokenizeRequest struct {
	Model        string `json:"model"`                   // Model name or path (must be loaded)
	Text         string `json:"text"`                    // Text to tokenize
	AddSpecial   *bool  `json:"add_special,omitempty"`   // Add BOS/EOS tokens (default: true)
	ParseSpecial *bool  `json:"parse_special,omitempty"` // Parse special tokens in text (default: false)
}

// TokenizeResponse contains the result of tokenization.
type TokenizeResponse struct {
	Tokens []llamacpp.Token `json:"tokens"`
}

// DetokenizeRequest contains parameters for detokenizing tokens.
type DetokenizeRequest struct {
	Model          string           `json:"model"`                     // Model name or path (must be loaded)
	Tokens         []llamacpp.Token `json:"tokens"`                    // Tokens to detokenize
	RemoveSpecial  *bool            `json:"remove_special,omitempty"`  // Remove BOS/EOS tokens (default: false)
	UnparseSpecial *bool            `json:"unparse_special,omitempty"` // Render special tokens as text (default: true)
}

// DetokenizeResponse contains the result of detokenization.
type DetokenizeResponse struct {
	Text string `json:"text"`
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r TokenizeRequest) String() string {
	return stringify(r)
}

func (r TokenizeResponse) String() string {
	return stringify(r)
}

func (r DetokenizeRequest) String() string {
	return stringify(r)
}

func (r DetokenizeResponse) String() string {
	return stringify(r)
}
