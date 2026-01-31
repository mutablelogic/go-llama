package llamacpp

/*
#include "completion.h"
#include "llama.h"
#include <stdbool.h>
#include <stdlib.h>

extern bool goAbortCallback(void* handle);
*/
import "C"
import (
	"context"
	"errors"
	"strings"
	"sync"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// CompletionOptions configures text generation
type CompletionOptions struct {
	// MaxTokens is the maximum number of tokens to generate (0 = default 128)
	MaxTokens int

	// StopWords are strings that stop generation when encountered
	StopWords []string

	// OnToken callback is called for each generated token
	// Return false to stop generation early
	OnToken func(token string) bool

	// Sampler parameters
	SamplerParams SamplerParams

	// EnablePrefixCaching reuses KV cache for matching prompt prefix
	EnablePrefixCaching bool

	// AbortContext cancels generation when done
	AbortContext context.Context
}

// DefaultCompletionOptions returns sensible defaults
func DefaultCompletionOptions() CompletionOptions {
	return CompletionOptions{
		MaxTokens:           2048,
		StopWords:           nil,
		OnToken:             nil,
		SamplerParams:       DefaultSamplerParams(),
		EnablePrefixCaching: true,
	}
}

///////////////////////////////////////////////////////////////////////////////
// C++ CALLBACK BRIDGE

// Global callback registry
var (
	callbackRegistry = make(map[uintptr]func(string) bool)
	callbackCounter  uintptr
	callbackMutex    sync.Mutex

	abortRegistry = make(map[uintptr]context.Context)
	abortCounter  uintptr
	abortMutex    sync.Mutex
)

//export goTokenCallback
func goTokenCallback(handle unsafe.Pointer, token *C.char) C.bool {
	if handle == nil {
		return C.bool(true)
	}

	callbackMutex.Lock()
	callback := callbackRegistry[uintptr(handle)]
	callbackMutex.Unlock()

	if callback == nil {
		return C.bool(true)
	}

	tokenStr := C.GoString(token)
	return C.bool(callback(tokenStr))
}

func registerCallback(cb func(string) bool) uintptr {
	callbackMutex.Lock()
	defer callbackMutex.Unlock()

	callbackCounter++
	handle := callbackCounter
	callbackRegistry[handle] = cb
	return handle
}

func unregisterCallback(handle uintptr) {
	callbackMutex.Lock()
	defer callbackMutex.Unlock()
	delete(callbackRegistry, handle)
}

//export goAbortCallback
func goAbortCallback(handle unsafe.Pointer) C.bool {
	if handle == nil {
		return C.bool(false)
	}

	abortMutex.Lock()
	ctx := abortRegistry[uintptr(handle)]
	abortMutex.Unlock()

	if ctx == nil {
		return C.bool(false)
	}

	select {
	case <-ctx.Done():
		return C.bool(true)
	default:
		return C.bool(false)
	}
}

func registerAbortContext(ctx context.Context) uintptr {
	abortMutex.Lock()
	defer abortMutex.Unlock()

	abortCounter++
	handle := abortCounter
	abortRegistry[handle] = ctx
	return handle
}

func unregisterAbortContext(handle uintptr) {
	abortMutex.Lock()
	defer abortMutex.Unlock()
	delete(abortRegistry, handle)
}

///////////////////////////////////////////////////////////////////////////////
// TEXT GENERATION

// CompleteNative generates text completion using the C++ generation loop
// This minimizes CGO overhead by keeping the entire generation loop in C++
func (ctx *Context) CompleteNative(prompt string, opts CompletionOptions) (string, error) {
	if ctx.handle == nil {
		return "", ErrInvalidContext
	}

	if ctx.model == nil || ctx.model.handle == nil {
		return "", ErrInvalidModel
	}

	// Set up default options
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = DefaultCompletionOptions().MaxTokens
	}

	// Register callback if provided
	var callbackHandle uintptr
	if opts.OnToken != nil {
		callbackHandle = registerCallback(opts.OnToken)
		defer unregisterCallback(callbackHandle)
	}

	// Register abort callback if provided
	var abortHandle uintptr
	if opts.AbortContext != nil {
		abortHandle = registerAbortContext(opts.AbortContext)
		cCtx := (*C.struct_llama_context)(ctx.handle)
		C.llama_set_abort_callback(cCtx, (C.ggml_abort_callback)(C.goAbortCallback), unsafe.Pointer(abortHandle))
		defer func() {
			C.llama_set_abort_callback(cCtx, nil, nil)
			unregisterAbortContext(abortHandle)
		}()
	}

	// Convert SamplerParams to C struct
	cParams := C.llama_go_completion_default_params()
	cParams.seed = C.uint32_t(opts.SamplerParams.Seed)
	cParams.temperature = C.float(opts.SamplerParams.Temperature)
	cParams.top_k = C.int32_t(opts.SamplerParams.TopK)
	cParams.top_p = C.float(opts.SamplerParams.TopP)
	cParams.min_p = C.float(opts.SamplerParams.MinP)
	cParams.repeat_penalty = C.float(opts.SamplerParams.RepeatPenalty)
	cParams.repeat_last_n = C.int32_t(opts.SamplerParams.RepeatLastN)
	cParams.frequency_penalty = C.float(opts.SamplerParams.FrequencyPenalty)
	cParams.presence_penalty = C.float(opts.SamplerParams.PresencePenalty)
	cParams.max_tokens = C.int32_t(opts.MaxTokens)
	cParams.enable_prefix_caching = C.bool(opts.EnablePrefixCaching)

	// Convert stop words to C array
	var cStopWords **C.char
	if len(opts.StopWords) > 0 {
		cStopWords = (**C.char)(C.malloc(C.size_t(len(opts.StopWords)+1) * C.size_t(unsafe.Sizeof(uintptr(0)))))
		defer C.free(unsafe.Pointer(cStopWords))

		stopWordsSlice := (*[1 << 30]*C.char)(unsafe.Pointer(cStopWords))[: len(opts.StopWords)+1 : len(opts.StopWords)+1]
		for i, word := range opts.StopWords {
			stopWordsSlice[i] = C.CString(word)
			defer C.free(unsafe.Pointer(stopWordsSlice[i]))
		}
		stopWordsSlice[len(opts.StopWords)] = nil
		cParams.stop_words = cStopWords
		cParams.stop_words_count = C.int32_t(len(opts.StopWords))
	}

	// Set callback handle if provided
	if callbackHandle != 0 {
		cParams.callback_handle = unsafe.Pointer(callbackHandle)
	}

	// Convert prompt to C string
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Call C++ generation function
	cResult := C.llama_go_completion_generate(ctx.handle, ctx.model.handle, cPrompt, &cParams)
	if cResult == nil {
		return "", getLastError()
	}
	defer C.llama_go_completion_free_result(cResult)

	return C.GoString(cResult.text), nil
}

// CompleteNativeWithStopInfo generates text completion and returns whether a stop sequence was hit
func (ctx *Context) CompleteNativeWithStopInfo(prompt string, opts CompletionOptions) (string, bool, error) {
	if ctx.handle == nil {
		return "", false, ErrInvalidContext
	}

	if ctx.model == nil || ctx.model.handle == nil {
		return "", false, ErrInvalidModel
	}

	// Set up default options
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = DefaultCompletionOptions().MaxTokens
	}

	// Register callback if provided
	var callbackHandle uintptr
	if opts.OnToken != nil {
		callbackHandle = registerCallback(opts.OnToken)
		defer unregisterCallback(callbackHandle)
	}

	// Register abort callback if provided
	var abortHandle uintptr
	if opts.AbortContext != nil {
		abortHandle = registerAbortContext(opts.AbortContext)
		cCtx := (*C.struct_llama_context)(ctx.handle)
		C.llama_set_abort_callback(cCtx, (C.ggml_abort_callback)(C.goAbortCallback), unsafe.Pointer(abortHandle))
		defer func() {
			C.llama_set_abort_callback(cCtx, nil, nil)
			unregisterAbortContext(abortHandle)
		}()
	}

	// Convert SamplerParams to C struct
	cParams := C.llama_go_completion_default_params()
	cParams.seed = C.uint32_t(opts.SamplerParams.Seed)
	cParams.temperature = C.float(opts.SamplerParams.Temperature)
	cParams.top_k = C.int32_t(opts.SamplerParams.TopK)
	cParams.top_p = C.float(opts.SamplerParams.TopP)
	cParams.min_p = C.float(opts.SamplerParams.MinP)
	cParams.repeat_penalty = C.float(opts.SamplerParams.RepeatPenalty)
	cParams.repeat_last_n = C.int32_t(opts.SamplerParams.RepeatLastN)
	cParams.frequency_penalty = C.float(opts.SamplerParams.FrequencyPenalty)
	cParams.presence_penalty = C.float(opts.SamplerParams.PresencePenalty)
	cParams.max_tokens = C.int32_t(opts.MaxTokens)
	cParams.enable_prefix_caching = C.bool(opts.EnablePrefixCaching)

	// Convert stop words to C array
	var cStopWords **C.char
	if len(opts.StopWords) > 0 {
		cStopWords = (**C.char)(C.malloc(C.size_t(len(opts.StopWords)+1) * C.size_t(unsafe.Sizeof(uintptr(0)))))
		defer C.free(unsafe.Pointer(cStopWords))

		stopWordsSlice := (*[1 << 30]*C.char)(unsafe.Pointer(cStopWords))[: len(opts.StopWords)+1 : len(opts.StopWords)+1]
		for i, word := range opts.StopWords {
			stopWordsSlice[i] = C.CString(word)
			defer C.free(unsafe.Pointer(stopWordsSlice[i]))
		}
		stopWordsSlice[len(opts.StopWords)] = nil
		cParams.stop_words = cStopWords
		cParams.stop_words_count = C.int32_t(len(opts.StopWords))
	}

	// Set callback handle if provided
	if callbackHandle != 0 {
		cParams.callback_handle = unsafe.Pointer(callbackHandle)
	}

	// Convert prompt to C string
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Call C++ generation function
	cResult := C.llama_go_completion_generate(ctx.handle, ctx.model.handle, cPrompt, &cParams)
	if cResult == nil {
		return "", false, getLastError()
	}
	defer C.llama_go_completion_free_result(cResult)

	text := C.GoString(cResult.text)
	stopWordHit := bool(cResult.stop_word_hit)
	return text, stopWordHit, nil
}

// Complete generates text completion for the given prompt
// This Go implementation makes multiple CGO calls per token.
// For better performance, use CompleteNative which keeps the generation loop in C++
func (ctx *Context) Complete(prompt string, opts CompletionOptions) (string, error) {
	if ctx.handle == nil {
		return "", ErrInvalidContext
	}

	if ctx.model == nil || ctx.model.handle == nil {
		return "", ErrInvalidModel
	}

	// Tokenize prompt
	tokenizeOpts := DefaultTokenizeOptions()
	tokenizeOpts.AddSpecial = true
	tokens, err := ctx.model.Tokenize(prompt, tokenizeOpts)
	if err != nil {
		return "", err
	}

	if len(tokens) == 0 {
		return "", errors.New("empty token sequence")
	}

	// Check context size
	nCtx := int(ctx.ContextSize())
	if len(tokens) >= nCtx {
		return "", errors.New("prompt too long for context size")
	}

	// Set max tokens
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}

	// Create sampler
	sampler, err := NewSampler(ctx.model, opts.SamplerParams)
	if err != nil {
		return "", err
	}
	defer sampler.Close()

	// Clear KV cache if not using prefix caching
	if !opts.EnablePrefixCaching {
		ctx.MemoryClear(true)
	}

	// Create batch and decode prompt
	batch, err := BatchFromTokens(tokens, 0, 0, true)
	if err != nil {
		return "", err
	}
	defer batch.Close()

	if err := batch.Decode(ctx); err != nil {
		return "", err
	}

	// Generation loop
	var result strings.Builder
	nPast := len(tokens)

	for i := 0; i < maxTokens; i++ {
		// Sample next token
		newToken, err := sampler.Sample(ctx, -1)
		if err != nil {
			break
		}

		// Check for EOS
		if ctx.model.IsEOG(newToken) {
			break
		}

		// Detokenize
		detokenizeOpts := DefaultDetokenizeOptions()
		tokenStr, err := ctx.model.Detokenize([]Token{newToken}, detokenizeOpts)
		if err != nil {
			break
		}
		result.WriteString(tokenStr)

		// Call callback if provided
		if opts.OnToken != nil {
			if !opts.OnToken(tokenStr) {
				break // User requested stop
			}
		}

		// Check stop words
		if len(opts.StopWords) > 0 {
			currentText := result.String()
			shouldStop := false
			for _, stopWord := range opts.StopWords {
				if strings.Contains(currentText, stopWord) {
					shouldStop = true
					break
				}
			}
			if shouldStop {
				break
			}
		}

		// Decode the sampled token for next iteration
		genBatch, err := BatchFromTokens([]Token{newToken}, int32(nPast), 0, true)
		if err != nil {
			break
		}

		if err := genBatch.Decode(ctx); err != nil {
			genBatch.Close()
			break
		}
		genBatch.Close()

		nPast++
	}

	return result.String(), nil
}

// CompleteWithMessages generates a completion using chat messages
// The messages are formatted using the model's chat template
func (ctx *Context) CompleteWithMessages(messages []ChatMessage, opts CompletionOptions) (string, error) {
	if ctx.model == nil {
		return "", ErrInvalidModel
	}

	// Apply chat template
	prompt, err := ctx.model.ApplyTemplate(messages, true)
	if err != nil {
		return "", err
	}

	return ctx.Complete(prompt, opts)
}
