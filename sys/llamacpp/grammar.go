package llamacpp

/*
#include "grammar.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// GRAMMAR SAMPLER

// GrammarSampler is a sampler that constrains generation using GBNF grammar
// It's a specialized Sampler that enforces grammar rules on token selection
type GrammarSampler struct {
	Sampler
}

// NewGrammarSampler creates a grammar-constrained sampler from GBNF grammar string
// The grammar uses GBNF (GGML BNF) format - see llama.cpp/grammars/README.md
// Example:
//
//	grammar := `root ::= "Yes" | "No"`
//	sampler, err := NewGrammarSampler(model, grammar, "root")
func NewGrammarSampler(model *Model, grammarStr, grammarRoot string) (*GrammarSampler, error) {
	if model == nil || model.handle == nil {
		return nil, ErrInvalidModel
	}
	if grammarStr == "" {
		return nil, ErrInvalidArgument
	}
	if grammarRoot == "" {
		grammarRoot = "root"
	}

	cGrammarStr := C.CString(grammarStr)
	defer C.free(unsafe.Pointer(cGrammarStr))

	cGrammarRoot := C.CString(grammarRoot)
	defer C.free(unsafe.Pointer(cGrammarRoot))

	handle := C.llama_go_grammar_sampler_new(
		model.handle,
		cGrammarStr,
		cGrammarRoot,
	)
	if handle == nil {
		return nil, getLastError()
	}

	gs := &GrammarSampler{
		Sampler: Sampler{
			handle: handle,
			model:  model,
		},
	}

	runtime.SetFinalizer(gs, func(gs *GrammarSampler) {
		gs.Close()
	})

	return gs, nil
}

///////////////////////////////////////////////////////////////////////////////
// LAZY GRAMMAR SAMPLER

// LazyGrammarSampler creates a grammar sampler that activates based on triggers
// Grammar is only applied after specific patterns or tokens appear in output
// Useful for conditional grammar application (e.g., only enforce JSON after seeing "{")
type LazyGrammarSampler struct {
	Sampler
}

// LazyGrammarOptions configures when grammar activation occurs
type LazyGrammarOptions struct {
	// GrammarStr is the GBNF grammar string
	GrammarStr string

	// GrammarRoot is the start symbol name (default: "root")
	GrammarRoot string

	// TriggerPatterns are regex patterns that activate the grammar
	// Pattern matches from start of generation, grammar gets content from first match group
	TriggerPatterns []string

	// TriggerTokens are token IDs that activate the grammar
	// Grammar gets content starting from the trigger token (included)
	TriggerTokens []Token
}

// NewLazyGrammarSampler creates a grammar sampler triggered by patterns or tokens
// Example:
//
//	opts := LazyGrammarOptions{
//	    GrammarStr: `root ::= object`,
//	    TriggerPatterns: []string{`\{`}, // Activate when "{" appears
//	}
//	sampler, err := NewLazyGrammarSampler(model, opts)
func NewLazyGrammarSampler(model *Model, opts LazyGrammarOptions) (*LazyGrammarSampler, error) {
	if model == nil || model.handle == nil {
		return nil, ErrInvalidModel
	}
	if opts.GrammarStr == "" {
		return nil, ErrInvalidArgument
	}
	if opts.GrammarRoot == "" {
		opts.GrammarRoot = "root"
	}

	cGrammarStr := C.CString(opts.GrammarStr)
	defer C.free(unsafe.Pointer(cGrammarStr))

	cGrammarRoot := C.CString(opts.GrammarRoot)
	defer C.free(unsafe.Pointer(cGrammarRoot))

	// Convert patterns to C strings
	var cPatterns **C.char
	var numPatterns C.size_t
	if len(opts.TriggerPatterns) > 0 {
		cPatternArray := make([]*C.char, len(opts.TriggerPatterns))
		for i, pattern := range opts.TriggerPatterns {
			cPatternArray[i] = C.CString(pattern)
		}
		defer func() {
			for _, cStr := range cPatternArray {
				C.free(unsafe.Pointer(cStr))
			}
		}()
		cPatterns = &cPatternArray[0]
		numPatterns = C.size_t(len(opts.TriggerPatterns))
	}

	// Convert tokens to C array
	var cTokens *C.int32_t
	var numTokens C.size_t
	if len(opts.TriggerTokens) > 0 {
		cTokenArray := make([]C.int32_t, len(opts.TriggerTokens))
		for i, token := range opts.TriggerTokens {
			cTokenArray[i] = C.int32_t(token)
		}
		cTokens = &cTokenArray[0]
		numTokens = C.size_t(len(opts.TriggerTokens))
	}

	handle := C.llama_go_grammar_sampler_new_lazy(
		model.handle,
		cGrammarStr,
		cGrammarRoot,
		cPatterns,
		numPatterns,
		cTokens,
		numTokens,
	)
	if handle == nil {
		return nil, getLastError()
	}

	lgs := &LazyGrammarSampler{
		Sampler: Sampler{
			handle: handle,
			model:  model,
		},
	}

	runtime.SetFinalizer(lgs, func(lgs *LazyGrammarSampler) {
		lgs.Close()
	})

	return lgs, nil
}
