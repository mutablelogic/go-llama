package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	// Packages
	readline "github.com/chzyer/readline"
	otel "github.com/mutablelogic/go-client/pkg/otel"
	httpclient "github.com/mutablelogic/go-llama/pkg/llamacpp/httpclient"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type ChatCommands struct {
	Chat ChatCommand `cmd:"" name:"chat" help:"Chat with a model." group:"CHAT"`
}

type ChatCommand struct {
	Model         string   `arg:"" name:"model" help:"Model name or path"`
	System        string   `arg:"" name:"system" optional:"" help:"System prompt (or use stdin if empty)"`
	MaxTokens     *int32   `name:"max-tokens" help:"Maximum tokens to generate"`
	Temperature   *float32 `name:"temperature" help:"Sampling temperature (0-2)"`
	TopP          *float32 `name:"top-p" help:"Nucleus sampling parameter (0-1)"`
	TopK          *int32   `name:"top-k" help:"Top-k sampling parameter"`
	RepeatPenalty *float32 `name:"repeat-penalty" help:"Penalize repeated tokens (1.0 = disabled)"`
	RepeatLastN   *int32   `name:"repeat-last-n" help:"Repeat penalty window size"`
	Seed          *uint32  `name:"seed" help:"RNG seed for reproducibility"`
	Stop          []string `name:"stop" help:"Stop sequences"`
	PrefixCache   *bool    `name:"prefix-cache" help:"Enable prefix caching"`
	Stream        bool     `name:"stream" help:"Stream output tokens" default:"true"`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *ChatCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "ChatCommand")
	defer func() { endSpan(err) }()

	// Get system prompt from argument or stdin (only if stdin isn't used for messages)
	stdinHasData := stdinHasData()
	system := cmd.System
	if system == "" {
		system, err = readStdin()
		if err != nil {
			return err
		}
	}

	// Interactive mode if stdin is a terminal
	if !stdinHasData {
		return runInteractiveChat(parent, ctx, client, cmd, system)
	}

	// Build messages from stdin (one per line)
	messages, err := readChatMessages()
	if err != nil {
		return err
	}
	if len(messages) == 0 {
		return fmt.Errorf("at least one message is required")
	}

	// Build options
	opts, err := buildChatOpts(cmd, system)
	if err != nil {
		return err
	}

	// Add streaming callback if requested
	printer := newRolePrinter(isTerminal(os.Stdout))
	if cmd.Stream {
		splitter := newThinkingStreamSplitter()
		opts = append(opts, httpclient.WithChatChunkCallback(func(chunk *schema.ChatChunk) error {
			if printWithFallback(printer, splitter, chunk) {
				return nil
			}
			printer.Print(chunk.Message.Role, chunk.Message.Content)
			return nil
		}))
	}

	// Chat
	result, err := client.Chat(parent, cmd.Model, messages, opts...)
	if err != nil {
		return err
	}
	if ctx.Debug {
		if b, err := json.MarshalIndent(result, "", "  "); err == nil {
			fmt.Fprintln(os.Stderr, string(b))
		}
	}

	// Print result if not streaming (streaming already printed chunks)
	if !cmd.Stream {
		printNonStreaming(printer, result)
	}
	if cmd.Stream {
		fmt.Print("\n") // Ensure newline at end of stream
	} else {
		fmt.Println() // Ensure newline at end
	}
	if result.FinishReason != "" {
		fmt.Printf("[finish_reason=%s]\n", result.FinishReason)
	}

	return nil
}

func runInteractiveChat(parentCtx context.Context, ctx *Globals, client *httpclient.Client, cmd *ChatCommand, system string) (err error) {
	// Build base options
	baseOpts, err := buildChatOpts(cmd, system)
	if err != nil {
		return err
	}

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          "you> ",
		HistoryLimit:    200,
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	})
	if err != nil {
		return err
	}
	defer rl.Close()

	var messages []schema.ChatMessage
	for {
		line, err := rl.Readline()
		if err == readline.ErrInterrupt {
			if len(line) == 0 {
				break
			}
			continue
		}
		if err == io.EOF {
			break
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "/exit" || line == "/quit" {
			break
		}

		messages = append(messages, schema.ChatMessage{Role: "user", Content: line})

		// Build request options for this turn
		opts := append([]httpclient.Opt{}, baseOpts...)
		var assistant strings.Builder
		printer := newRolePrinter(isTerminal(os.Stdout))
		if cmd.Stream {
			splitter := newThinkingStreamSplitter()
			opts = append(opts, httpclient.WithChatChunkCallback(func(chunk *schema.ChatChunk) error {
				if printWithFallback(printer, splitter, chunk) {
					if chunk.Message.Role == "assistant" {
						assistant.WriteString(chunk.Message.Content)
					}
					return nil
				}
				printer.Print(chunk.Message.Role, chunk.Message.Content)
				if chunk.Message.Role == "assistant" {
					assistant.WriteString(chunk.Message.Content)
				}
				return nil
			}))
		}

		result, err := client.Chat(parentCtx, cmd.Model, messages, opts...)
		if err != nil {
			return err
		}
		if ctx.Debug {
			if b, err := json.MarshalIndent(result, "", "  "); err == nil {
				fmt.Fprintln(os.Stderr, string(b))
			}
		}

		if !cmd.Stream {
			assistant.WriteString(printNonStreaming(printer, result))
		}
		if cmd.Stream {
			fmt.Print("\n")
		} else {
			fmt.Println()
		}
		if result.FinishReason != "" {
			fmt.Printf("[finish_reason=%s]\n", result.FinishReason)
		}

		if assistant.Len() > 0 {
			messages = append(messages, schema.ChatMessage{Role: "assistant", Content: assistant.String()})
		} else if result.Message.Content != "" {
			messages = append(messages, schema.ChatMessage{Role: "assistant", Content: result.Message.Content})
		}
	}

	return nil
}

// readChatMessages reads messages from stdin in format: "role: content"
// Example: "user: What is 2+2?" or "user\nWhat is 2+2?"
func readChatMessages() ([]schema.ChatMessage, error) {
	info, err := os.Stdin.Stat()
	if err != nil {
		return nil, err
	}

	// Check if stdin has data (pipe or redirect)
	if (info.Mode() & os.ModeCharDevice) != 0 {
		// stdin is a terminal, no piped input
		return nil, nil
	}

	reader := bufio.NewReader(os.Stdin)
	var messages []schema.ChatMessage
	var currentRole string
	var currentContent strings.Builder

	for {
		line, err := reader.ReadString('\n')
		line = strings.TrimSpace(line)

		if line == "" {
			// Empty line might indicate end of message or continuation
			if err != nil || reader.Buffered() == 0 {
				// End of input
				if currentRole != "" {
					messages = append(messages, schema.ChatMessage{
						Role:    currentRole,
						Content: strings.TrimSpace(currentContent.String()),
					})
				}
				break
			}
			continue
		}

		// Check if this line has "role:" prefix
		if idx := strings.Index(line, ":"); idx > 0 {
			potentialRole := strings.TrimSpace(line[:idx])
			// Common chat roles
			if potentialRole == "user" || potentialRole == "assistant" || potentialRole == "system" {
				// Save previous message if any
				if currentRole != "" {
					messages = append(messages, schema.ChatMessage{
						Role:    currentRole,
						Content: strings.TrimSpace(currentContent.String()),
					})
					currentContent.Reset()
				}
				currentRole = potentialRole
				currentContent.WriteString(strings.TrimSpace(line[idx+1:]))
				continue
			}
		}

		// If no role found in current line, append to current content
		if currentRole != "" {
			if currentContent.Len() > 0 {
				currentContent.WriteString("\n")
			}
			currentContent.WriteString(line)
		}

		if err != nil {
			if currentRole != "" {
				messages = append(messages, schema.ChatMessage{
					Role:    currentRole,
					Content: strings.TrimSpace(currentContent.String()),
				})
			}
			break
		}
	}

	return messages, nil
}

///////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS

// stdinHasData checks if stdin has pipe or redirect data
func stdinHasData() bool {
	info, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return (info.Mode() & os.ModeCharDevice) == 0
}

// buildChatOpts builds httpclient options from command flags
func buildChatOpts(cmd *ChatCommand, system string) ([]httpclient.Opt, error) {
	var opts []httpclient.Opt

	if system != "" {
		opts = append(opts, httpclient.WithSystem(system))
	}
	if cmd.MaxTokens != nil {
		opts = append(opts, httpclient.WithMaxTokens(*cmd.MaxTokens))
	}
	if cmd.Temperature != nil {
		opts = append(opts, httpclient.WithTemperature(*cmd.Temperature))
	}
	if cmd.TopP != nil {
		opts = append(opts, httpclient.WithTopP(*cmd.TopP))
	}
	if cmd.TopK != nil {
		opts = append(opts, httpclient.WithTopK(*cmd.TopK))
	}
	if cmd.RepeatPenalty != nil {
		opts = append(opts, httpclient.WithRepeatPenalty(*cmd.RepeatPenalty))
	}
	if cmd.RepeatLastN != nil {
		opts = append(opts, httpclient.WithRepeatLastN(*cmd.RepeatLastN))
	}
	if cmd.Seed != nil {
		opts = append(opts, httpclient.WithSeed(*cmd.Seed))
	}
	if len(cmd.Stop) > 0 {
		opts = append(opts, httpclient.WithStop(cmd.Stop...))
	}
	if cmd.PrefixCache != nil {
		opts = append(opts, httpclient.WithPrefixCache(*cmd.PrefixCache))
	}

	return opts, nil
}

// rolePrinter prints content with role-specific formatting
type rolePrinter struct {
	isTerminal bool
}

func newRolePrinter(isTerminal bool) *rolePrinter {
	return &rolePrinter{isTerminal: isTerminal}
}

func (p *rolePrinter) Print(role, content string) {
	if !p.isTerminal {
		fmt.Print(content)
		return
	}

	switch role {
	case "assistant":
		// Bold white for assistant
		fmt.Printf("\033[1;37m%s\033[0m", content)
	case "thinking":
		// Dark grey for thinking/reasoning
		fmt.Printf("\033[0;90m%s\033[0m", content)
	default:
		fmt.Print(content)
	}
}

// thinkingStreamSplitter processes chat chunks and splits thinking from assistant
type thinkingStreamSplitter struct {
	buffer string
}

func newThinkingStreamSplitter() *thinkingStreamSplitter {
	return &thinkingStreamSplitter{}
}

// Process returns true if the chunk was handled (thinking block)
func (s *thinkingStreamSplitter) Process(chunk *schema.ChatChunk) (*schema.ChatChunk, bool) {
	if chunk == nil {
		return nil, false
	}

	s.buffer += chunk.Message.Content

	// Look for thinking end tags
	thinkTags := []string{"</think>", "</reasoning>", "</scratchpad>", "</thought>", "</internal>"}
	for _, tag := range thinkTags {
		if idx := strings.Index(s.buffer, tag); idx >= 0 {
			// Found end of thinking block
			return nil, true
		}
	}

	// Not yet complete
	return nil, false
}

// Flush returns any remaining content
func (s *thinkingStreamSplitter) Flush() string {
	result := s.buffer
	s.buffer = ""
	return result
}

// printWithFallback prints content with fallback thinking parsing if needed
func printWithFallback(printer *rolePrinter, splitter *thinkingStreamSplitter, chunk *schema.ChatChunk) bool {
	if chunk == nil {
		return false
	}

	// If chunk already has a thinking role, print it
	if chunk.Message.Role == "thinking" {
		printer.Print("thinking", chunk.Message.Content)
		return true
	}

	// If chunk has assistant role, check if content contains thinking tags
	if chunk.Message.Role == "assistant" {
		content := chunk.Message.Content

		// Look for thinking start tags
		thinkStartTags := []string{"<think>", "<reasoning>", "<scratchpad>", "<thought>", "<internal>"}
		hasThinkingStart := false
		for _, tag := range thinkStartTags {
			if strings.Contains(content, tag) {
				hasThinkingStart = true
				break
			}
		}

		if hasThinkingStart {
			// Parse thinking blocks and print them separately
			return parseAndPrintThinkingBlocks(printer, content)
		}
	}

	return false
}

// parseAndPrintThinkingBlocks finds and prints thinking blocks separately
func parseAndPrintThinkingBlocks(printer *rolePrinter, content string) bool {
	thinkTags := map[string]string{
		"<think>":      "</think>",
		"<reasoning>":  "</reasoning>",
		"<scratchpad>": "</scratchpad>",
		"<thought>":    "</thought>",
		"<internal>":   "</internal>",
	}

	hasThinking := false
	remaining := content

	for startTag, endTag := range thinkTags {
		for {
			startIdx := strings.Index(remaining, startTag)
			if startIdx == -1 {
				break
			}

			endIdx := strings.Index(remaining[startIdx:], endTag)
			if endIdx == -1 {
				break
			}

			endIdx += startIdx

			// Print text before thinking block as assistant
			if startIdx > 0 {
				printer.Print("assistant", remaining[:startIdx])
			}

			// Extract and print thinking content
			thinkContent := remaining[startIdx+len(startTag) : endIdx]
			printer.Print("thinking", thinkContent)

			remaining = remaining[endIdx+len(endTag):]
			hasThinking = true
		}
	}

	// Print any remaining content as assistant
	if remaining != "" {
		printer.Print("assistant", remaining)
	}

	return hasThinking
}

// printNonStreaming prints non-streamed response with thinking handling
func printNonStreaming(printer *rolePrinter, result *schema.ChatResponse) string {
	if result == nil {
		return ""
	}

	var output strings.Builder

	// Print thinking if present
	if result.Thinking != nil && result.Thinking.Content != "" {
		printer.Print("thinking", result.Thinking.Content)
		output.WriteString(result.Thinking.Content)
	}

	// Print main message, checking for embedded thinking tags
	content := result.Message.Content
	if !parseAndPrintThinkingBlocks(printer, content) {
		// No thinking tags found, print as-is
		printer.Print("assistant", content)
	}
	output.WriteString(content)

	return output.String()
}
