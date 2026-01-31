package schema

///////////////////////////////////////////////////////////////////////////////
// TYPES

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", "assistant", or "tool"
	Content string `json:"content"` // The message content
}

// ChatRequest contains parameters for chat completion.
// It embeds CompletionRequest to reuse sampling and model options.
type ChatRequest struct {
	CompletionRequest
	Messages []ChatMessage `json:"messages"`
}

// ChatResponse contains the generated assistant message.
type ChatResponse struct {
	Model        string      `json:"model"`                   // Model used
	Message      ChatMessage `json:"message"`                 // Assistant message
	Usage        Usage       `json:"usage"`                   // Token usage
	FinishReason string      `json:"finish_reason,omitempty"` // Reason generation ended
}

// ChatChunk contains a streamed chat chunk.
type ChatChunk struct {
	Message ChatMessage `json:"message"`
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (r ChatRequest) String() string {
	return stringify(r)
}

func (r ChatResponse) String() string {
	return stringify(r)
}

func (r ChatChunk) String() string {
	return stringify(r)
}
