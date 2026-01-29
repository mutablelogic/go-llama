package llamacpp

/*
#include "tokenizer.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TOKEN TYPE

// Token represents a single token ID
type Token = int32

///////////////////////////////////////////////////////////////////////////////
// SPECIAL TOKENS

// BOS returns the beginning-of-sequence token
func (m *Model) BOS() Token {
	if m.handle == nil {
		return -1
	}
	return Token(C.llama_go_token_bos(m.handle))
}

// EOS returns the end-of-sequence token
func (m *Model) EOS() Token {
	if m.handle == nil {
		return -1
	}
	return Token(C.llama_go_token_eos(m.handle))
}

// EOT returns the end-of-turn token (for chat models)
func (m *Model) EOT() Token {
	if m.handle == nil {
		return -1
	}
	return Token(C.llama_go_token_eot(m.handle))
}

// NL returns the newline token
func (m *Model) NL() Token {
	if m.handle == nil {
		return -1
	}
	return Token(C.llama_go_token_nl(m.handle))
}

// PAD returns the padding token
func (m *Model) PAD() Token {
	if m.handle == nil {
		return -1
	}
	return Token(C.llama_go_token_pad(m.handle))
}

///////////////////////////////////////////////////////////////////////////////
// TOKEN CHECKING

// IsEOG checks if a token is end-of-generation
func (m *Model) IsEOG(token Token) bool {
	if m.handle == nil {
		return false
	}
	return bool(C.llama_go_token_is_eog(m.handle, C.int32_t(token)))
}

// IsControl checks if a token is a control token
func (m *Model) IsControl(token Token) bool {
	if m.handle == nil {
		return false
	}
	return bool(C.llama_go_token_is_control(m.handle, C.int32_t(token)))
}

///////////////////////////////////////////////////////////////////////////////
// TOKENIZATION

// TokenizeOptions configures tokenization behavior
type TokenizeOptions struct {
	AddSpecial   bool // Add BOS/EOS tokens
	ParseSpecial bool // Parse special tokens in text (like <|endoftext|>)
}

// DefaultTokenizeOptions returns default tokenization options
func DefaultTokenizeOptions() TokenizeOptions {
	return TokenizeOptions{
		AddSpecial:   true,
		ParseSpecial: false,
	}
}

// Tokenize converts text to tokens
func (m *Model) Tokenize(text string, opts TokenizeOptions) ([]Token, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}

	if len(text) == 0 {
		return []Token{}, nil
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// First call to get required size
	estimatedSize := len(text) + 16 // Add some padding for special tokens
	tokens := make([]Token, estimatedSize)

	n := C.llama_go_tokenize(
		m.handle,
		cText,
		C.int32_t(len(text)),
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(estimatedSize),
		C.bool(opts.AddSpecial),
		C.bool(opts.ParseSpecial),
	)

	if n < 0 {
		// Need larger buffer - n is negated required size
		requiredSize := int(-n)
		tokens = make([]Token, requiredSize)
		n = C.llama_go_tokenize(
			m.handle,
			cText,
			C.int32_t(len(text)),
			(*C.int32_t)(unsafe.Pointer(&tokens[0])),
			C.int32_t(requiredSize),
			C.bool(opts.AddSpecial),
			C.bool(opts.ParseSpecial),
		)
		if n < 0 {
			return nil, getLastError()
		}
	}

	return tokens[:n], nil
}

// TokenToString converts a single token to its string representation.
// Returns ErrInvalidModel if model is closed, ErrInvalidToken if token is invalid.
func (m *Model) TokenToString(token Token) (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}

	// Most tokens are short, start with small buffer
	buf := make([]byte, 32)
	n := C.llama_go_token_to_piece(
		m.handle,
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(len(buf)),
		C.bool(true), // render special tokens
	)

	if n < 0 {
		// Need larger buffer
		buf = make([]byte, -n)
		n = C.llama_go_token_to_piece(
			m.handle,
			C.int32_t(token),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int32_t(len(buf)),
			C.bool(true),
		)
		if n < 0 {
			return "", ErrInvalidToken
		}
	}

	return string(buf[:n]), nil
}

///////////////////////////////////////////////////////////////////////////////
// DETOKENIZATION

// DetokenizeOptions configures detokenization behavior
type DetokenizeOptions struct {
	RemoveSpecial  bool // Remove BOS/EOS tokens from output
	UnparseSpecial bool // Render special tokens as text
}

// DefaultDetokenizeOptions returns default detokenization options
func DefaultDetokenizeOptions() DetokenizeOptions {
	return DetokenizeOptions{
		RemoveSpecial:  false,
		UnparseSpecial: true,
	}
}

// Detokenize converts tokens back to text
func (m *Model) Detokenize(tokens []Token, opts DetokenizeOptions) (string, error) {
	if m.handle == nil {
		return "", ErrInvalidModel
	}

	if len(tokens) == 0 {
		return "", nil
	}

	// Estimate output size: average ~4 bytes per token
	estimatedSize := len(tokens) * 8
	if estimatedSize < 64 {
		estimatedSize = 64
	}

	buf := make([]byte, estimatedSize)

	n := C.llama_go_detokenize(
		m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens)),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(len(buf)),
		C.bool(opts.RemoveSpecial),
		C.bool(opts.UnparseSpecial),
	)

	if n < 0 {
		// Need larger buffer
		requiredSize := -int(n)
		buf = make([]byte, requiredSize)
		n = C.llama_go_detokenize(
			m.handle,
			(*C.int32_t)(unsafe.Pointer(&tokens[0])),
			C.int32_t(len(tokens)),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int32_t(len(buf)),
			C.bool(opts.RemoveSpecial),
			C.bool(opts.UnparseSpecial),
		)
		if n < 0 {
			return "", getLastError()
		}
	}

	return string(buf[:n]), nil
}
