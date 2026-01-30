package httpclient

import (
	"fmt"

	// Packages
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type opt struct {
	// Model loading options
	Gpu    *int32
	Layers *int32
	Mmap   *bool
	Mlock  *bool

	// Completion options
	MaxTokens   *int32
	Temperature *float32
	TopP        *float32
	TopK        *int32
	Seed        *uint32
	Stop        []string
	PrefixCache *bool

	// Embedding options
	Normalize *bool

	// Tokenizer options
	AddSpecial     *bool
	ParseSpecial   *bool
	RemoveSpecial  *bool
	UnparseSpecial *bool

	// Streaming callback
	chunkCallback func(*schema.CompletionChunk) error
}

// Opt is an option to set on the client request.
type Opt func(*opt) error

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func applyOpts(opts ...Opt) (*opt, error) {
	o := new(opt)
	for _, opt := range opts {
		if err := opt(o); err != nil {
			return nil, err
		}
	}
	return o, nil
}

///////////////////////////////////////////////////////////////////////////////
// OPTIONS - MODEL LOADING

// WithGpu sets the main GPU index for model loading.
func WithGpu(gpu int32) Opt {
	return func(o *opt) error {
		o.Gpu = &gpu
		return nil
	}
}

// WithLayers sets the number of layers to offload to GPU.
// Use -1 to offload all layers.
func WithLayers(layers int32) Opt {
	return func(o *opt) error {
		o.Layers = &layers
		return nil
	}
}

// WithMmap enables or disables memory mapping for model loading.
func WithMmap(mmap bool) Opt {
	return func(o *opt) error {
		o.Mmap = &mmap
		return nil
	}
}

// WithMlock enables or disables locking the model in memory.
func WithMlock(mlock bool) Opt {
	return func(o *opt) error {
		o.Mlock = &mlock
		return nil
	}
}

///////////////////////////////////////////////////////////////////////////////
// OPTIONS - COMPLETION

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxTokens int32) Opt {
	return func(o *opt) error {
		if maxTokens < 1 {
			return fmt.Errorf("max_tokens must be at least 1")
		}
		o.MaxTokens = &maxTokens
		return nil
	}
}

// WithTemperature sets the sampling temperature.
// Valid range is [0, 2] inclusive.
func WithTemperature(temperature float32) Opt {
	return func(o *opt) error {
		if temperature < 0 || temperature > 2 {
			return fmt.Errorf("temperature must be between 0 and 2 (inclusive)")
		}
		o.Temperature = &temperature
		return nil
	}
}

// WithTopP sets the nucleus sampling parameter.
// Valid range is [0, 1] inclusive.
func WithTopP(topP float32) Opt {
	return func(o *opt) error {
		if topP < 0 || topP > 1 {
			return fmt.Errorf("top_p must be between 0 and 1 (inclusive)")
		}
		o.TopP = &topP
		return nil
	}
}

// WithTopK sets the top-k sampling parameter.
func WithTopK(topK int32) Opt {
	return func(o *opt) error {
		if topK < 1 {
			return fmt.Errorf("top_k must be at least 1")
		}
		o.TopK = &topK
		return nil
	}
}

// WithSeed sets the RNG seed for reproducible generation.
func WithSeed(seed uint32) Opt {
	return func(o *opt) error {
		o.Seed = &seed
		return nil
	}
}

// WithStop sets the stop sequences for generation.
func WithStop(stop ...string) Opt {
	return func(o *opt) error {
		o.Stop = stop
		return nil
	}
}

// WithPrefixCache enables or disables prefix caching optimization.
func WithPrefixCache(prefixCache bool) Opt {
	return func(o *opt) error {
		o.PrefixCache = &prefixCache
		return nil
	}
}

// WithChunkCallback sets a callback function to receive streaming chunks.
// This enables streaming support for text completion.
func WithChunkCallback(callback func(*schema.CompletionChunk) error) Opt {
	return func(o *opt) error {
		o.chunkCallback = callback
		return nil
	}
}

///////////////////////////////////////////////////////////////////////////////
// OPTIONS - EMBEDDING

// WithNormalize enables or disables L2 normalization of embeddings.
func WithNormalize(normalize bool) Opt {
	return func(o *opt) error {
		o.Normalize = &normalize
		return nil
	}
}

///////////////////////////////////////////////////////////////////////////////
// OPTIONS - TOKENIZER

// WithAddSpecial enables or disables adding BOS/EOS tokens during tokenization.
func WithAddSpecial(addSpecial bool) Opt {
	return func(o *opt) error {
		o.AddSpecial = &addSpecial
		return nil
	}
}

// WithParseSpecial enables or disables parsing special tokens in input text.
func WithParseSpecial(parseSpecial bool) Opt {
	return func(o *opt) error {
		o.ParseSpecial = &parseSpecial
		return nil
	}
}

// WithRemoveSpecial enables or disables removing BOS/EOS tokens during detokenization.
func WithRemoveSpecial(removeSpecial bool) Opt {
	return func(o *opt) error {
		o.RemoveSpecial = &removeSpecial
		return nil
	}
}

// WithUnparseSpecial enables or disables rendering special tokens as text during detokenization.
func WithUnparseSpecial(unparseSpecial bool) Opt {
	return func(o *opt) error {
		o.UnparseSpecial = &unparseSpecial
		return nil
	}
}
