package llamacpp

/*
#include "runtime.h"
*/
import "C"
import (
	"fmt"
)

///////////////////////////////////////////////////////////////////////////////
// MODEL INFO

// ModelInfo contains runtime information about a loaded model.
type ModelInfo struct {
	NLayer     int32  // Total number of layers
	NHead      int32  // Number of attention heads
	NHeadKV    int32  // Number of KV heads (for GQA/MQA)
	NEmbd      int32  // Embedding dimension
	NCtxTrain  int32  // Training context length
	NParams    uint64 // Total parameter count
	ModelSize  uint64 // Model size in bytes
}

// Info returns runtime information about the model.
func (m *Model) Info() (ModelInfo, error) {
	if m.handle == nil {
		return ModelInfo{}, ErrInvalidModel
	}

	var cInfo C.llama_go_model_info
	if !C.llama_go_get_model_info(m.handle, &cInfo) {
		return ModelInfo{}, getLastError()
	}

	return ModelInfo{
		NLayer:    int32(cInfo.n_layer),
		NHead:     int32(cInfo.n_head),
		NHeadKV:   int32(cInfo.n_head_kv),
		NEmbd:     int32(cInfo.n_embd),
		NCtxTrain: int32(cInfo.n_ctx_train),
		NParams:   uint64(cInfo.n_params),
		ModelSize: uint64(cInfo.model_size),
	}, nil
}

// ParamCount returns the total number of parameters in the model.
func (m *Model) ParamCount() uint64 {
	if m.handle == nil {
		return 0
	}

	info, err := m.Info()
	if err != nil {
		return 0
	}
	return info.NParams
}

// SizeBytes returns the model size in bytes.
func (m *Model) SizeBytes() uint64 {
	if m.handle == nil {
		return 0
	}

	info, err := m.Info()
	if err != nil {
		return 0
	}
	return info.ModelSize
}

// SizeString returns a human-readable model size (e.g., "4.2 GB").
func (m *Model) SizeString() string {
	size := m.SizeBytes()
	if size == 0 {
		return "0 B"
	}

	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)

	switch {
	case size >= GB:
		return fmt.Sprintf("%.2f GB", float64(size)/float64(GB))
	case size >= MB:
		return fmt.Sprintf("%.2f MB", float64(size)/float64(MB))
	case size >= KB:
		return fmt.Sprintf("%.2f KB", float64(size)/float64(KB))
	default:
		return fmt.Sprintf("%d B", size)
	}
}

///////////////////////////////////////////////////////////////////////////////
// CONTEXT INFO

// ContextInfo contains runtime information about a context.
type ContextInfo struct {
	NCtx     uint32 // Context size
	NBatch   uint32 // Batch size
	NUBatch  uint32 // Micro-batch size
	NSeqMax  uint32 // Max sequences
	NThreads int32  // Thread count
}

// Info returns runtime information about the context.
func (ctx *Context) Info() (ContextInfo, error) {
	if ctx.handle == nil {
		return ContextInfo{}, ErrInvalidContext
	}

	var cInfo C.llama_go_context_info
	if !C.llama_go_get_context_info(ctx.handle, &cInfo) {
		return ContextInfo{}, getLastError()
	}

	return ContextInfo{
		NCtx:     uint32(cInfo.n_ctx),
		NBatch:   uint32(cInfo.n_batch),
		NUBatch:  uint32(cInfo.n_ubatch),
		NSeqMax:  uint32(cInfo.n_seq_max),
		NThreads: int32(cInfo.n_threads),
	}, nil
}

///////////////////////////////////////////////////////////////////////////////
// PERFORMANCE DATA

// PerfData contains performance timing data.
type PerfData struct {
	StartMs     float64 // Absolute start time (ms)
	LoadMs      float64 // Model loading time (ms)
	PromptMs    float64 // Prompt processing time (ms)
	GenerateMs  float64 // Token generation time (ms)
	PromptCount int32   // Prompt tokens processed
	TokenCount  int32   // Tokens generated
}

// TokensPerSecond returns the token generation rate.
func (p PerfData) TokensPerSecond() float64 {
	if p.GenerateMs <= 0 || p.TokenCount <= 0 {
		return 0
	}
	return float64(p.TokenCount) / (p.GenerateMs / 1000.0)
}

// PromptTokensPerSecond returns the prompt processing rate.
func (p PerfData) PromptTokensPerSecond() float64 {
	if p.PromptMs <= 0 || p.PromptCount <= 0 {
		return 0
	}
	return float64(p.PromptCount) / (p.PromptMs / 1000.0)
}

// Perf returns performance timing data for this context.
func (ctx *Context) Perf() (PerfData, error) {
	if ctx.handle == nil {
		return PerfData{}, ErrInvalidContext
	}

	var cData C.llama_go_perf_data
	if !C.llama_go_get_perf_data(ctx.handle, &cData) {
		return PerfData{}, getLastError()
	}

	return PerfData{
		StartMs:     float64(cData.t_start_ms),
		LoadMs:      float64(cData.t_load_ms),
		PromptMs:    float64(cData.t_p_eval_ms),
		GenerateMs:  float64(cData.t_eval_ms),
		PromptCount: int32(cData.n_p_eval),
		TokenCount:  int32(cData.n_eval),
	}, nil
}

// PerfReset resets the performance counters.
func (ctx *Context) PerfReset() {
	if ctx.handle != nil {
		C.llama_go_perf_reset(ctx.handle)
	}
}

///////////////////////////////////////////////////////////////////////////////
// KV CACHE INFO

// MemorySeqLength returns the number of tokens cached for a sequence.
// Returns 0 if the sequence is empty.
func (ctx *Context) MemorySeqLength(seqID int32) int32 {
	min := ctx.MemorySeqPosMin(seqID)
	max := ctx.MemorySeqPosMax(seqID)
	if min < 0 || max < 0 {
		return 0
	}
	return max - min + 1
}
