package llamacpp

/*
#include "embedding.h"
#include "decode.h"
*/
import "C"
import (
	"math"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// PoolingType represents the pooling strategy for embeddings
type PoolingType int32

const (
	PoolingUnspecified PoolingType = -1
	PoolingNone        PoolingType = 0
	PoolingMean        PoolingType = 1
	PoolingCLS         PoolingType = 2
	PoolingLast        PoolingType = 3
	PoolingRank        PoolingType = 4
)

// String returns a human-readable name for the pooling type
func (p PoolingType) String() string {
	switch p {
	case PoolingUnspecified:
		return "unspecified"
	case PoolingNone:
		return "none"
	case PoolingMean:
		return "mean"
	case PoolingCLS:
		return "cls"
	case PoolingLast:
		return "last"
	case PoolingRank:
		return "rank"
	default:
		return "unknown"
	}
}

///////////////////////////////////////////////////////////////////////////////
// EMBEDDING MODE

// SetEmbeddings enables or disables embedding extraction mode.
// When enabled, embeddings will be computed during decode.
func (ctx *Context) SetEmbeddings(enabled bool) {
	if ctx.handle != nil {
		C.llama_go_set_embeddings(ctx.handle, C.bool(enabled))
	}
}

// PoolingType returns the pooling type configured for this context.
func (ctx *Context) PoolingType() PoolingType {
	if ctx.handle == nil {
		return PoolingUnspecified
	}
	return PoolingType(C.llama_go_get_pooling_type(ctx.handle))
}

///////////////////////////////////////////////////////////////////////////////
// BATCH EMBEDDINGS

// GetAllEmbeddings returns all embeddings from the last decode.
// Use nOutputs to specify how many embeddings to extract (number of tokens with logits=true).
// The returned slice has size nOutputs * nEmbd.
func (ctx *Context) GetAllEmbeddings(nOutputs int) ([]float32, error) {
	if ctx.handle == nil {
		return nil, ErrInvalidContext
	}

	embd := C.llama_go_get_all_embeddings(ctx.handle)
	if embd == nil {
		return nil, getLastError()
	}

	nEmbd := int(C.llama_go_n_embd(ctx.handle))
	if nOutputs <= 0 || nEmbd <= 0 {
		return nil, ErrInvalidContext
	}

	total := nOutputs * nEmbd
	return unsafe.Slice((*float32)(unsafe.Pointer(embd)), total), nil
}

// GetEmbeddingsBySeq returns the pooled embeddings for a specific sequence.
// This only works when pooling_type is not NONE.
// The sequence must have been processed in a batch.
func (ctx *Context) GetEmbeddingsBySeq(seqID int32) ([]float32, error) {
	if ctx.handle == nil {
		return nil, ErrInvalidContext
	}

	embd := C.llama_go_get_embeddings_seq(ctx.handle, C.int32_t(seqID))
	if embd == nil {
		return nil, getLastError()
	}

	nEmbd := int(C.llama_go_n_embd(ctx.handle))
	if nEmbd <= 0 {
		return nil, ErrInvalidContext
	}

	return unsafe.Slice((*float32)(unsafe.Pointer(embd)), nEmbd), nil
}

///////////////////////////////////////////////////////////////////////////////
// EMBEDDING UTILITIES

// NormalizeEmbeddings performs L2 normalization on an embedding vector in-place.
// This is commonly needed before computing cosine similarity.
func NormalizeEmbeddings(embd []float32) {
	if len(embd) == 0 {
		return
	}

	// Compute L2 norm
	var sum float64
	for _, v := range embd {
		sum += float64(v) * float64(v)
	}

	if sum > 0 {
		norm := math.Sqrt(sum)
		for i := range embd {
			embd[i] = float32(float64(embd[i]) / norm)
		}
	}
}

// CosineSimilarity computes the cosine similarity between two embedding vectors.
// Both vectors should be normalized for best results.
// Returns a value between -1 and 1.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EuclideanDistance computes the Euclidean distance between two embedding vectors.
func EuclideanDistance(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float64
	for i := range a {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// DotProduct computes the dot product between two embedding vectors.
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}

	return dot
}

// BatchEmbeddings holds multiple embeddings with their sequence information.
type BatchEmbeddings struct {
	Embeddings [][]float32 // One embedding vector per sequence/token
	Dimension  int         // Embedding dimension
}

// ExtractBatchEmbeddings extracts individual embeddings from a flattened buffer.
// nOutputs is the number of embeddings, nEmbd is the dimension.
func ExtractBatchEmbeddings(flat []float32, nOutputs, nEmbd int) BatchEmbeddings {
	if len(flat) != nOutputs*nEmbd {
		return BatchEmbeddings{}
	}

	result := BatchEmbeddings{
		Embeddings: make([][]float32, nOutputs),
		Dimension:  nEmbd,
	}

	for i := 0; i < nOutputs; i++ {
		start := i * nEmbd
		end := start + nEmbd
		// Make a copy to avoid referencing the original buffer
		embd := make([]float32, nEmbd)
		copy(embd, flat[start:end])
		result.Embeddings[i] = embd
	}

	return result
}

// Normalize normalizes all embeddings in the batch.
func (be *BatchEmbeddings) Normalize() {
	for i := range be.Embeddings {
		NormalizeEmbeddings(be.Embeddings[i])
	}
}

///////////////////////////////////////////////////////////////////////////////
// BATCH EMBEDDING COMPUTATION

// EmbeddingOptions configures how embeddings are computed.
type EmbeddingOptions struct {
	// Normalize whether to L2-normalize embeddings (recommended for cosine similarity)
	Normalize bool
	// AddBOS whether to add BOS token at the start
	AddBOS bool
	// AddEOS whether to add EOS token at the end
	AddEOS bool
}

// DefaultEmbeddingOptions returns default options for embedding computation.
func DefaultEmbeddingOptions() EmbeddingOptions {
	return EmbeddingOptions{
		Normalize: true,
		AddBOS:    true,
		AddEOS:    false,
	}
}

// ComputeEmbeddings computes embeddings for multiple texts in a single batch.
// This is the high-level API for batch embedding computation.
// Returns one embedding vector per input text.
func (ctx *Context) ComputeEmbeddings(model *Model, texts []string, opts EmbeddingOptions) (*BatchEmbeddings, error) {
	if ctx.handle == nil {
		return nil, ErrInvalidContext
	}
	if model == nil || model.handle == nil {
		return nil, ErrInvalidModel
	}
	if len(texts) == 0 {
		return &BatchEmbeddings{Dimension: int(ctx.NEmbd())}, nil
	}

	// Enable embeddings mode
	ctx.SetEmbeddings(true)

	// Get embedding dimension
	nEmbd := ctx.NEmbd()
	poolingType := ctx.PoolingType()

	// Tokenize all texts
	tokOpts := DefaultTokenizeOptions()
	tokOpts.AddSpecial = opts.AddBOS
	tokOpts.ParseSpecial = false

	allTokens := make([][]Token, len(texts))
	totalTokens := int32(0)

	for i, text := range texts {
		tokens, err := model.Tokenize(text, tokOpts)
		if err != nil {
			return nil, err
		}
		// Add EOS if requested
		if opts.AddEOS {
			eos := model.EOS()
			if eos != -1 {
				tokens = append(tokens, eos)
			}
		}
		allTokens[i] = tokens
		totalTokens += int32(len(tokens))
	}

	// Create batch large enough for all tokens
	batch, err := NewBatch(totalTokens, int32(len(texts)))
	if err != nil {
		return nil, err
	}
	defer batch.Close()

	// Clear KV cache before processing
	if err := ctx.MemoryClear(true); err != nil {
		return nil, err
	}

	// Process based on pooling type
	if poolingType == PoolingNone {
		// No pooling: get embedding from last token of each sequence
		return ctx.computeEmbeddingsNoPooling(batch, allTokens, int(nEmbd), opts.Normalize)
	}

	// With pooling: each text gets its own sequence ID
	return ctx.computeEmbeddingsWithPooling(batch, allTokens, int(nEmbd), opts.Normalize)
}

// computeEmbeddingsNoPooling handles the case when pooling is disabled.
// Gets the embedding from the last token of each text.
func (ctx *Context) computeEmbeddingsNoPooling(batch *Batch, allTokens [][]Token, nEmbd int, normalize bool) (*BatchEmbeddings, error) {
	result := &BatchEmbeddings{
		Embeddings: make([][]float32, len(allTokens)),
		Dimension:  nEmbd,
	}

	// Process each text separately since we need per-token embeddings
	for i, tokens := range allTokens {
		if len(tokens) == 0 {
			result.Embeddings[i] = make([]float32, nEmbd)
			continue
		}

		batch.Clear()
		if err := ctx.MemoryClear(true); err != nil {
			return nil, err
		}

		// Add all tokens, mark only the last one for logits
		for j, tok := range tokens {
			if err := batch.Add(tok, int32(j), 0, j == len(tokens)-1); err != nil {
				return nil, err
			}
		}

		if err := batch.Decode(ctx); err != nil {
			return nil, err
		}

		// Get embedding from the last token (index len(tokens)-1)
		embd, err := ctx.GetEmbeddings(int32(len(tokens) - 1))
		if err != nil {
			return nil, err
		}

		// Copy the embedding
		result.Embeddings[i] = make([]float32, nEmbd)
		copy(result.Embeddings[i], embd)

		if normalize {
			NormalizeEmbeddings(result.Embeddings[i])
		}
	}

	return result, nil
}

// computeEmbeddingsWithPooling handles the case when pooling is enabled.
// Each text gets its own sequence ID and embeddings are pooled automatically.
func (ctx *Context) computeEmbeddingsWithPooling(batch *Batch, allTokens [][]Token, nEmbd int, normalize bool) (*BatchEmbeddings, error) {
	result := &BatchEmbeddings{
		Embeddings: make([][]float32, len(allTokens)),
		Dimension:  nEmbd,
	}

	// Add all tokens to the batch with different sequence IDs
	for seqID, tokens := range allTokens {
		for j, tok := range tokens {
			// Mark the last token of each sequence for output
			isLast := j == len(tokens)-1
			if err := batch.Add(tok, int32(j), int32(seqID), isLast); err != nil {
				return nil, err
			}
		}
	}

	// Decode the entire batch at once
	if err := batch.Decode(ctx); err != nil {
		return nil, err
	}

	// Extract embeddings for each sequence
	for seqID := range allTokens {
		if len(allTokens[seqID]) == 0 {
			result.Embeddings[seqID] = make([]float32, nEmbd)
			continue
		}

		embd, err := ctx.GetEmbeddingsBySeq(int32(seqID))
		if err != nil {
			return nil, err
		}

		// Copy the embedding
		result.Embeddings[seqID] = make([]float32, nEmbd)
		copy(result.Embeddings[seqID], embd)

		if normalize {
			NormalizeEmbeddings(result.Embeddings[seqID])
		}
	}

	return result, nil
}

// ComputeEmbedding computes embedding for a single text.
// Convenience wrapper around ComputeEmbeddings.
func (ctx *Context) ComputeEmbedding(model *Model, text string, opts EmbeddingOptions) ([]float32, error) {
	batch, err := ctx.ComputeEmbeddings(model, []string{text}, opts)
	if err != nil {
		return nil, err
	}
	if len(batch.Embeddings) == 0 {
		return nil, ErrInvalidContext
	}
	return batch.Embeddings[0], nil
}

// SimilarityMatrix computes pairwise cosine similarity between all embeddings.
// Returns an NxN matrix where result[i][j] is the similarity between embeddings i and j.
func (be *BatchEmbeddings) SimilarityMatrix() [][]float64 {
	n := len(be.Embeddings)
	if n == 0 {
		return nil
	}

	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				matrix[i][j] = 1.0 // Self-similarity
			} else if j < i {
				matrix[i][j] = matrix[j][i] // Symmetric
			} else {
				matrix[i][j] = CosineSimilarity(be.Embeddings[i], be.Embeddings[j])
			}
		}
	}

	return matrix
}

// MostSimilar returns indices of the k most similar embeddings to the given query index.
// Excludes the query itself from results.
func (be *BatchEmbeddings) MostSimilar(queryIdx int, k int) []int {
	n := len(be.Embeddings)
	if queryIdx < 0 || queryIdx >= n || k <= 0 {
		return nil
	}

	// Compute similarities
	type scoredIdx struct {
		idx   int
		score float64
	}

	scores := make([]scoredIdx, 0, n-1)
	query := be.Embeddings[queryIdx]
	for i, embd := range be.Embeddings {
		if i != queryIdx {
			scores = append(scores, scoredIdx{i, CosineSimilarity(query, embd)})
		}
	}

	// Sort by score descending
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Return top-k indices
	if k > len(scores) {
		k = len(scores)
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = scores[i].idx
	}

	return result
}
