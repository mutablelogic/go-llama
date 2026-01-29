# Grammar System Bindings

This document describes the Grammar system bindings added to `sys/llamacpp`.

## Overview

Grammar-based generation allows constraining model output to follow specific patterns using GBNF (GGML BNF) format. This is useful for:

- Forcing JSON/structured output
- Constraining responses to specific formats (Yes/No, multiple choice, etc.)
- Ensuring output follows specific schemas or templates

## Files Added

- `sys/llamacpp/grammar.h` - C wrapper declarations
- `sys/llamacpp/grammar.cpp` - C++ implementation using llama.cpp grammar API
- `sys/llamacpp/grammar.go` - Go bindings
- `sys/llamacpp/grammar_test.go` - 5 tests for grammar functionality

## API

### GrammarSampler

Creates a sampler that enforces grammar rules on all generated tokens.

```go
// Example: Only allow "Yes" or "No" responses
grammar := `root ::= "Yes" | "No"`
sampler, err := NewGrammarSampler(model, grammar, "root")
if err != nil {
    return err
}
defer sampler.Close()
```

### LazyGrammarSampler  

Creates a sampler that activates grammar rules only after specific patterns or tokens appear.

```go
// Example: Enforce JSON structure only after "{" appears
opts := LazyGrammarOptions{
    GrammarStr:      `root ::= object ...`,
    GrammarRoot:     "root",
    TriggerPatterns: []string{`\{`}, // Activate on "{"
}
sampler, err := NewLazyGrammarSampler(model, opts)
```

### Usage

Grammar samplers are specialized `Sampler` instances and can be used anywhere a regular sampler is used:

```go
// Use with Context.Sample
token, err := sampler.Sample(ctx, position)

// Accept token for repetition penalty tracking
sampler.Accept(token)
```

## GBNF Grammar Format

GBNF is a BNF-like grammar format. See `third_party/llama.cpp/grammars/README.md` for full syntax.

Basic example:

```gbnf
root ::= object
object ::= "{" ws "\"name\"" ws ":" ws string ws "}"
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
```

## Implementation Details

### C++ Layer

- `llama_go_grammar_sampler_new()` - Creates grammar sampler via `llama_sampler_init_grammar()`
- `llama_go_grammar_sampler_new_lazy()` - Creates lazy grammar sampler via `llama_sampler_init_grammar_lazy_patterns()`

Both functions:

1. Extract model vocab using `llama_model_get_vocab()`
2. Initialize grammar sampler with vocab and grammar string
3. Return sampler handle (same type as other samplers, so can use shared sampler functions)

### Go Layer

- `GrammarSampler` - Embeds `Sampler` struct, no additional fields needed
- `LazyGrammarSampler` - Embeds `Sampler` struct
- `LazyGrammarOptions` - Configuration for lazy grammar activation

Both types are just typed wrappers around the base `Sampler` type for type safety and documentation.

## Tests

5 tests covering:

1. `TestGrammarSampler` - Basic grammar (Yes/No)
2. `TestGrammarSampler_JSON` - JSON grammar
3. `TestLazyGrammarSampler` - Pattern-triggered grammar
4. `TestGrammarSampler_InvalidGrammar` - Error handling
5. `TestGrammarSampler_EmptyGrammar` - Error handling

Tests verify sampler creation succeeds/fails appropriately. Full generation testing is handled by existing sampler and completion tests.

## Error Handling

- Returns error if grammar string is empty
- Returns error if grammar has invalid syntax (llama.cpp parser error)
- Returns error if model is invalid

Added `ErrInvalidArgument` to error.go for empty grammar case.

## Integration

Grammar samplers integrate seamlessly with existing code:

- Work with `Context.Sample()` like any sampler
- Compatible with `Complete()` and `CompleteNative()` by passing as sampler
- Can be used in sampler chains if needed (though grammar typically used alone)

## Test Results

All 185 sys tests pass (added 5 new grammar tests).
All 9 pkg tests still pass (no changes needed).

## Next Steps

Grammar bindings complete and tested. Next binding categories to implement:

- Multimodal/MTMD (image/audio input)
- Speculative Decoding (draft models)
- Advanced Samplers (Mirostat, DRY, XTC)
