package llamacpp

/*
#include "sampler.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// Sampler handles token selection from model output
type Sampler struct {
	handle unsafe.Pointer
	model  *Model // Keep reference to prevent GC
}

// SamplerParams configures sampling behavior
type SamplerParams struct {
	// Seed for random sampling (0 = random seed)
	Seed uint32

	// Temperature controls randomness (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
	// Set to 0 for greedy sampling
	Temperature float32

	// TopK limits sampling to top K tokens (0 = disabled)
	TopK int32

	// TopP (nucleus sampling) limits to tokens with cumulative probability <= P (1.0 = disabled)
	TopP float32

	// MinP filters tokens with probability < P * max_prob (0.0 = disabled)
	MinP float32

	// RepeatPenalty penalizes repeated tokens (1.0 = disabled)
	RepeatPenalty float32

	// RepeatLastN is the number of tokens to consider for repetition penalty
	RepeatLastN int32

	// FrequencyPenalty reduces probability of frequent tokens (0.0 = disabled)
	FrequencyPenalty float32

	// PresencePenalty reduces probability of tokens that appeared at all (0.0 = disabled)
	PresencePenalty float32
}

///////////////////////////////////////////////////////////////////////////////
// DEFAULT PARAMS

// DefaultSamplerParams returns sensible default sampling parameters
func DefaultSamplerParams() SamplerParams {
	cParams := C.llama_go_sampler_default_params()
	return SamplerParams{
		Seed:             uint32(cParams.seed),
		Temperature:      float32(cParams.temperature),
		TopK:             int32(cParams.top_k),
		TopP:             float32(cParams.top_p),
		MinP:             float32(cParams.min_p),
		RepeatPenalty:    float32(cParams.repeat_penalty),
		RepeatLastN:      int32(cParams.repeat_last_n),
		FrequencyPenalty: float32(cParams.frequency_penalty),
		PresencePenalty:  float32(cParams.presence_penalty),
	}
}

// GreedySamplerParams returns parameters for greedy (deterministic) sampling
func GreedySamplerParams() SamplerParams {
	return SamplerParams{
		Temperature:   0.0,
		TopK:          0,
		TopP:          1.0,
		MinP:          0.0,
		RepeatPenalty: 1.0,
	}
}

///////////////////////////////////////////////////////////////////////////////
// SAMPLER LIFECYCLE

// NewSampler creates a sampler chain with the given parameters
func NewSampler(model *Model, params SamplerParams) (*Sampler, error) {
	if model == nil || model.handle == nil {
		return nil, ErrInvalidModel
	}

	cParams := C.llama_go_sampler_params{
		seed:              C.uint32_t(params.Seed),
		temperature:       C.float(params.Temperature),
		top_k:             C.int32_t(params.TopK),
		top_p:             C.float(params.TopP),
		min_p:             C.float(params.MinP),
		repeat_penalty:    C.float(params.RepeatPenalty),
		repeat_last_n:     C.int32_t(params.RepeatLastN),
		frequency_penalty: C.float(params.FrequencyPenalty),
		presence_penalty:  C.float(params.PresencePenalty),
	}

	handle := C.llama_go_sampler_new(model.handle, cParams)
	if handle == nil {
		return nil, getLastError()
	}

	s := &Sampler{
		handle: handle,
		model:  model,
	}

	runtime.SetFinalizer(s, func(s *Sampler) {
		s.Close()
	})

	return s, nil
}

// Close frees the sampler resources
func (s *Sampler) Close() error {
	if s.handle != nil {
		C.llama_go_sampler_free(s.handle)
		s.handle = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// SAMPLING

// Sample selects the next token from the context at position idx
// idx is typically the last position where tokens were added
func (s *Sampler) Sample(ctx *Context, idx int32) (Token, error) {
	if s.handle == nil {
		return -1, ErrInvalidContext
	}
	if ctx == nil || ctx.handle == nil {
		return -1, ErrInvalidContext
	}

	token := C.llama_go_sampler_sample(s.handle, ctx.handle, C.int32_t(idx))
	if token < 0 {
		return -1, getLastError()
	}

	return Token(token), nil
}

// Accept records a token for repetition penalty tracking
func (s *Sampler) Accept(token Token) {
	if s.handle != nil {
		C.llama_go_sampler_accept(s.handle, C.int32_t(token))
	}
}

// Reset clears the sampler state
func (s *Sampler) Reset() {
	if s.handle != nil {
		C.llama_go_sampler_reset(s.handle)
	}
}

// ChainLength returns the number of samplers in the chain
func (s *Sampler) ChainLength() int32 {
	if s.handle == nil {
		return 0
	}
	return int32(C.llama_go_sampler_chain_n(s.handle))
}

///////////////////////////////////////////////////////////////////////////////
// CUSTOM SAMPLER CHAIN

// SamplerChain allows building custom sampler chains
type SamplerChain struct {
	handle unsafe.Pointer
}

// NewSamplerChain creates a new empty sampler chain
func NewSamplerChain() *SamplerChain {
	handle := C.llama_go_sampler_chain_init(C.bool(false))
	if handle == nil {
		return nil
	}

	sc := &SamplerChain{handle: handle}
	runtime.SetFinalizer(sc, func(sc *SamplerChain) {
		sc.Close()
	})
	return sc
}

// Close frees the sampler chain
func (sc *SamplerChain) Close() error {
	if sc.handle != nil {
		C.llama_go_sampler_free(sc.handle)
		sc.handle = nil
	}
	return nil
}

// AddGreedy adds a greedy sampler (always pick highest probability)
func (sc *SamplerChain) AddGreedy() {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_greedy())
	}
}

// AddDist adds a random distribution sampler
func (sc *SamplerChain) AddDist(seed uint32) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_dist(C.uint32_t(seed)))
	}
}

// AddTopK adds top-K filtering
func (sc *SamplerChain) AddTopK(k int32) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_top_k(C.int32_t(k)))
	}
}

// AddTopP adds top-P (nucleus) filtering
func (sc *SamplerChain) AddTopP(p float32, minKeep int) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_top_p(C.float(p), C.size_t(minKeep)))
	}
}

// AddMinP adds min-P filtering
func (sc *SamplerChain) AddMinP(p float32, minKeep int) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_min_p(C.float(p), C.size_t(minKeep)))
	}
}

// AddTemp adds temperature scaling
func (sc *SamplerChain) AddTemp(temp float32) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_temp(C.float(temp)))
	}
}

// AddPenalties adds repetition/frequency/presence penalties
func (sc *SamplerChain) AddPenalties(lastN int32, repeat, freq, presence float32) {
	if sc.handle != nil {
		C.llama_go_sampler_chain_add(sc.handle, C.llama_go_sampler_init_penalties(
			C.int32_t(lastN),
			C.float(repeat),
			C.float(freq),
			C.float(presence),
		))
	}
}

// Sample selects the next token
func (sc *SamplerChain) Sample(ctx *Context, idx int32) (Token, error) {
	if sc.handle == nil {
		return -1, ErrInvalidContext
	}
	if ctx == nil || ctx.handle == nil {
		return -1, ErrInvalidContext
	}

	token := C.llama_go_sampler_sample(sc.handle, ctx.handle, C.int32_t(idx))
	if token < 0 {
		return -1, getLastError()
	}

	return Token(token), nil
}

// Accept records a token for penalty tracking
func (sc *SamplerChain) Accept(token Token) {
	if sc.handle != nil {
		C.llama_go_sampler_accept(sc.handle, C.int32_t(token))
	}
}

// Reset clears sampler state
func (sc *SamplerChain) Reset() {
	if sc.handle != nil {
		C.llama_go_sampler_reset(sc.handle)
	}
}

// Length returns the number of samplers in the chain
func (sc *SamplerChain) Length() int32 {
	if sc.handle == nil {
		return 0
	}
	return int32(C.llama_go_sampler_chain_n(sc.handle))
}

///////////////////////////////////////////////////////////////////////////////
// ADVANCED SAMPLERS

// XTCSampler excludes top choices for diversity
type XTCSampler struct {
	Sampler
}

// XTCParams configures XTC (Exclude Top Choices) sampling
type XTCParams struct {
	Probability float32 // Probability threshold (0.0-1.0)
	Threshold   float32 // Logit threshold
	MinKeep     int     // Minimum tokens to keep
	Seed        uint32  // Random seed
}

// DefaultXTCParams returns sensible XTC defaults
func DefaultXTCParams() XTCParams {
	return XTCParams{
		Probability: 0.0,
		Threshold:   0.1,
		MinKeep:     1,
		Seed:        0,
	}
}

// NewXTCSampler creates an XTC sampler for diversity
func NewXTCSampler(params XTCParams) (*XTCSampler, error) {
	handle := C.llama_go_sampler_init_xtc(
		C.float(params.Probability),
		C.float(params.Threshold),
		C.size_t(params.MinKeep),
		C.uint32_t(params.Seed),
	)
	if handle == nil {
		return nil, getLastError()
	}

	s := &XTCSampler{
		Sampler: Sampler{handle: handle},
	}

	runtime.SetFinalizer(s, func(s *XTCSampler) {
		s.Close()
	})

	return s, nil
}

// MirostatV2Sampler uses simplified Mirostat algorithm
type MirostatV2Sampler struct {
	Sampler
}

// MirostatV2Params configures Mirostat v2 sampling
type MirostatV2Params struct {
	Seed uint32  // Random seed
	Tau  float32 // Target cross-entropy (higher = more random)
	Eta  float32 // Learning rate for mu updates
}

// DefaultMirostatV2Params returns sensible Mirostat v2 defaults
func DefaultMirostatV2Params() MirostatV2Params {
	return MirostatV2Params{
		Seed: 0,
		Tau:  5.0,
		Eta:  0.1,
	}
}

// NewMirostatV2Sampler creates a Mirostat v2 sampler
func NewMirostatV2Sampler(params MirostatV2Params) (*MirostatV2Sampler, error) {
	handle := C.llama_go_sampler_init_mirostat_v2(
		C.uint32_t(params.Seed),
		C.float(params.Tau),
		C.float(params.Eta),
	)
	if handle == nil {
		return nil, getLastError()
	}

	s := &MirostatV2Sampler{
		Sampler: Sampler{handle: handle},
	}

	runtime.SetFinalizer(s, func(s *MirostatV2Sampler) {
		s.Close()
	})

	return s, nil
}
