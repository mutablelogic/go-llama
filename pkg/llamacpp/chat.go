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
		if onChunk != nil {
			splitter = newThinkingStreamSplitter()
			opts.OnToken = func(token string) bool {
				if callbackErr != nil {
					return false
				}
				for _, chunk := range splitter.Process(token) {
					if err := onChunk(chunk); err != nil {
						callbackErr = err
						return false
					}
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
