//go:build !client

package llamacpp

import (
	"context"
	"fmt"
	"strings"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
	attribute "go.opentelemetry.io/otel/attribute"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Chat generates a response for the given chat messages.
// If onChunk is provided, it will be called for each generated token.
// The callback can stop generation early by returning an error.
func (l *Llama) Chat(ctx context.Context, req schema.ChatRequest, onChunk func(schema.ChatChunk) error) (result *schema.ChatResponse, err error) {
	ctx, endSpan := otel.StartSpan(l.tracer, ctx, schema.SpanName("Chat"),
		attribute.String("request", req.String()),
	)
	defer func() { endSpan(err) }()

	if len(req.Stop) == 0 {
		req.Stop = defaultStopSequences
	}

	// Create a context, and run the chat completion
	err = l.WithContext(ctx, schema.ContextRequest{
		LoadModelRequest: schema.LoadModelRequest{
			Name: req.Model,
		},
	}, func(ctx context.Context, task *Task) error {
		// Lock the model - chat is not thread-safe
		task.CachedModel().Lock()
		defer task.CachedModel().Unlock()

		prompt, err := buildChatPrompt(task.Model(), req)
		if err != nil {
			return err
		}

		opts := buildCompletionOptions(ctx, req.CompletionRequest)
		var callbackErr error
		var splitter *thinkingStreamSplitter
		var stopFilter *stopMarkerFilter
		if onChunk != nil {
			splitter = newThinkingStreamSplitter()
			stopFilter = newStopMarkerFilter(req.Stop)
			opts.OnToken = func(token string) bool {
				if callbackErr != nil {
					return false
				}
				trimmed, stopped := stopFilter.Process(token)
				if trimmed != "" {
					for _, chunk := range splitter.Process(trimmed) {
						if err := onChunk(chunk); err != nil {
							callbackErr = err
							return false
						}
					}
				}
				if stopped {
					return false
				}
				return true
			}
		}

		text, stopWordHit, err := task.Context().CompleteNativeWithStopInfo(prompt, opts)
		if err != nil {
			return err
		}
		if callbackErr != nil {
			return callbackErr
		}

		if onChunk != nil && splitter != nil && stopFilter != nil && !stopFilter.Stopped() {
			if tail := stopFilter.Flush(); tail != "" {
				for _, chunk := range splitter.Process(tail) {
					if err := onChunk(chunk); err != nil {
						return err
					}
				}
			}
		}

		text, _ = trimAtStop(text, req.Stop)

		usage, err := completionUsage(task.Model(), prompt, text)
		if err != nil {
			return err
		}
		finishReason := completionFinishReason(req.CompletionRequest, text, usage, stopWordHit)
		parsed := ParseReasoning(text)
		cleanText := parsed.Content
		var thinkingMsg *schema.ChatMessage
		if parsed.HasThinking {
			thinkingMsg = &schema.ChatMessage{Role: "thinking", Content: parsed.Thinking}
		}

		result = &schema.ChatResponse{
			Model:    req.Model,
			Thinking: thinkingMsg,
			Message: schema.ChatMessage{
				Role:    "assistant",
				Content: cleanText,
			},
			Usage:        usage,
			FinishReason: finishReason,
		}

		if onChunk != nil && splitter != nil {
			for _, chunk := range splitter.Flush() {
				if err := onChunk(chunk); err != nil {
					return err
				}
			}
		}
		return nil
	})
	return
}

///////////////////////////////////////////////////////////////////////////////
// HELPERS

func buildChatPrompt(model *llamacpp.Model, req schema.ChatRequest) (string, error) {
	if model == nil {
		return "", fmt.Errorf("model is required")
	}

	messages := make([]llamacpp.ChatMessage, 0, len(req.Messages)+1)
	if req.Prompt != "" {
		messages = append(messages, llamacpp.ChatMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}
	for _, msg := range req.Messages {
		messages = append(messages, llamacpp.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	if len(messages) == 0 {
		return "", fmt.Errorf("no chat messages provided")
	}
	if !model.HasChatTemplate() {
		return "", fmt.Errorf("model has no chat template")
	}

	return llamacpp.ApplyTemplateWithModel(model, "", messages, true)
}

type thinkingStreamSplitter struct {
	inThinking bool
	buffer     string
}

var (
	thinkingOpenTags  = []string{"<think>", "<reasoning>", "<scratchpad>", "<thought>", "<internal>"}
	thinkingCloseTags = []string{"</think>", "</reasoning>", "</scratchpad>", "</thought>", "</internal>"}
	maxThinkingTagLen = maxTagLen(append(thinkingOpenTags, thinkingCloseTags...))
)

func newThinkingStreamSplitter() *thinkingStreamSplitter {
	return &thinkingStreamSplitter{}
}

func (f *thinkingStreamSplitter) Process(chunk string) []schema.ChatChunk {
	return f.process(chunk, false)
}

func (f *thinkingStreamSplitter) Flush() []schema.ChatChunk {
	return f.process("", true)
}

func (f *thinkingStreamSplitter) process(chunk string, flush bool) []schema.ChatChunk {
	if chunk == "" && !flush {
		return nil
	}

	s := f.buffer + chunk
	if !flush {
		limit := len(s) - (maxThinkingTagLen - 1)
		if limit < 0 {
			f.buffer = s
			return nil
		}
		f.buffer = s[limit:]
		s = s[:limit]
	} else {
		f.buffer = ""
	}

	var out []schema.ChatChunk
	var current strings.Builder
	currentRole := "assistant"
	if f.inThinking {
		currentRole = "thinking"
	}

	flushCurrent := func() {
		if current.Len() == 0 {
			return
		}
		out = append(out, schema.ChatChunk{Message: schema.ChatMessage{Role: currentRole, Content: current.String()}})
		current.Reset()
	}

	for i := 0; i < len(s); {
		if f.inThinking {
			if tagLen := matchTag(s, i, thinkingCloseTags); tagLen > 0 {
				flushCurrent()
				f.inThinking = false
				currentRole = "assistant"
				i += tagLen
				continue
			}
			current.WriteByte(s[i])
			i++
			continue
		}

		if tagLen := matchTag(s, i, thinkingOpenTags); tagLen > 0 {
			flushCurrent()
			f.inThinking = true
			currentRole = "thinking"
			i += tagLen
			continue
		}

		current.WriteByte(s[i])
		i++
	}

	flushCurrent()
	return out
}

func matchTag(s string, i int, tags []string) int {
	for _, tag := range tags {
		if len(s)-i >= len(tag) && s[i:i+len(tag)] == tag {
			return len(tag)
		}
	}
	return 0
}

func maxTagLen(tags []string) int {
	max := 0
	for _, tag := range tags {
		if len(tag) > max {
			max = len(tag)
		}
	}
	if max < 1 {
		return 1
	}
	return max
}

var defaultStopSequences = []string{
	"<|end|>",
	"<|end|}",
	"<|",
	"</s>",
	"<|eos|>",
	"<|endoftext|>",
	"<|user|>",
	"<|assistant|>",
	"<|system|>",
}

type stopMarkerFilter struct {
	stops   []string
	buffer  string
	stopped bool
}

func newStopMarkerFilter(stops []string) *stopMarkerFilter {
	return &stopMarkerFilter{stops: stops}
}

func (f *stopMarkerFilter) Stopped() bool {
	return f.stopped
}

// Process handles incoming tokens using llama.cpp's two-phase approach:
// 1. Check for full stop sequences
// 2. If no full stop, check if text ends with a prefix of any stop word (partial match)
//
// Returns the text safe to send and whether generation should stop.
func (f *stopMarkerFilter) Process(text string) (string, bool) {
	if f.stopped {
		return "", true
	}
	if text == "" {
		return "", false
	}

	combined := f.buffer + text

	// Phase 1: Check for full stop sequence
	if idx, found := indexAnyStop(combined, f.stops); found {
		f.stopped = true
		f.buffer = ""
		return combined[:idx], true
	}

	// Phase 2: Check if text ends with a partial stop sequence
	// If so, withhold the partial match in the buffer
	if partialPos := findPartialStop(combined, f.stops); partialPos != -1 {
		// Text ends with a prefix of a stop word - withhold from partialPos onward
		f.buffer = combined[partialPos:]
		if partialPos == 0 {
			return "", false
		}
		return combined[:partialPos], false
	}

	// No full or partial match - safe to send everything
	f.buffer = ""
	return combined, false
}

func (f *stopMarkerFilter) Flush() string {
	if f.stopped || f.buffer == "" {
		return ""
	}
	out := f.buffer
	f.buffer = ""
	return out
}

// findPartialStop checks if s ends with a prefix of any stop word.
// Returns the position where the partial match starts, or -1 if no partial match.
// This mirrors llama.cpp's string_find_partial_stop function.
func findPartialStop(s string, stops []string) int {
	if len(s) == 0 {
		return -1
	}

	lastChar := s[len(s)-1]

	for _, stop := range stops {
		if len(stop) == 0 {
			continue
		}

		// Check each possible prefix of the stop word, starting from longest.
		// We only need to consider prefixes up to the length of s, since longer
		// prefixes cannot be a suffix of s.
		maxPrefixLen := len(stop)
		if maxPrefixLen > len(s) {
			maxPrefixLen = len(s)
		}

		// We look for prefixes that end with the last character of s.
		for prefixLen := maxPrefixLen; prefixLen >= 1; prefixLen-- {
			if stop[prefixLen-1] != lastChar {
				continue
			}

			prefix := stop[:prefixLen]
			if strings.HasSuffix(s, prefix) {
				return len(s) - prefixLen
			}
		}
	}

	return -1
}

func indexAnyStop(s string, stops []string) (int, bool) {
	idx := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if i := strings.Index(s, stop); i >= 0 {
			if idx == -1 || i < idx {
				idx = i
			}
		}
	}
	return idx, idx >= 0
}

func trimAtStop(content string, stops []string) (string, bool) {
	idx, found := indexAnyStop(content, stops)
	if !found {
		return content, false
	}
	return content[:idx], true
}
