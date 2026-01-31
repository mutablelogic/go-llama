package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	// Packages
	"github.com/chzyer/readline"
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
	if system == "" && !stdinHasData {
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

func buildChatOpts(cmd *ChatCommand, system string) ([]httpclient.Opt, error) {
	opts := []httpclient.Opt{}
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
		stopSeqs := unescapeStopSequences(cmd.Stop)
		opts = append(opts, httpclient.WithStop(stopSeqs...))
	}
	if cmd.PrefixCache != nil {
		opts = append(opts, httpclient.WithPrefixCache(*cmd.PrefixCache))
	}

	return opts, nil
}

type rolePrinter struct {
	isTerminal bool
	current    string
}

func newRolePrinter(isTerminal bool) *rolePrinter {
	return &rolePrinter{isTerminal: isTerminal}
}

func (p *rolePrinter) Print(role, content string) {
	if content == "" {
		return
	}
	if role == "" {
		role = "assistant"
	}

	if p.current != role {
		if p.current != "" {
			fmt.Print("\n")
		}
		p.printPrefix(role)
		p.current = role
	}

	if p.isTerminal {
		fmt.Print(roleColor(role))
		fmt.Print(content)
		fmt.Print(colorReset)
	} else {
		fmt.Print(content)
	}
}

func (p *rolePrinter) printPrefix(role string) {
	if p.isTerminal {
		fmt.Print(roleColor(role))
		fmt.Printf("%s: ", role)
		fmt.Print(colorReset)
	} else {
		fmt.Printf("%s: ", role)
	}
}

const (
	colorReset     = "\033[0m"
	colorAssistant = "\033[1;37m"
	colorThinking  = "\033[0;90m"
)

func roleColor(role string) string {
	switch role {
	case "thinking":
		return colorThinking
	default:
		return colorAssistant
	}
}

func printWithFallback(printer *rolePrinter, splitter *thinkingStreamSplitter, chunk *schema.ChatChunk) bool {
	role := strings.TrimSpace(chunk.Message.Role)
	content := chunk.Message.Content

	if role == "thinking" {
		printer.Print("thinking", content)
		return true
	}

	if role == "" || role == "assistant" {
		if splitter.HasState() || containsThinkingTags(content) {
			for _, msg := range splitter.Process(content) {
				printer.Print(msg.Role, msg.Content)
			}
			return true
		}
	}

	return false
}

func printNonStreaming(printer *rolePrinter, result *schema.ChatResponse) string {
	if result == nil {
		return ""
	}
	if result.Thinking != nil && result.Thinking.Content != "" {
		printer.Print(result.Thinking.Role, result.Thinking.Content)
		printer.Print(result.Message.Role, result.Message.Content)
		return result.Message.Content
	}
	thinking, content, ok := parseThinkingText(result.Message.Content)
	if ok {
		printer.Print("thinking", thinking)
		printer.Print(result.Message.Role, content)
		return content
	}
	printer.Print(result.Message.Role, result.Message.Content)
	return result.Message.Content
}

// thinkingStreamSplitter splits <think>...</think> blocks into role-based messages.
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

func (f *thinkingStreamSplitter) HasState() bool {
	return f.inThinking || f.buffer != ""
}

func (f *thinkingStreamSplitter) Process(chunk string) []schema.ChatMessage {
	return f.process(chunk, false)
}

func (f *thinkingStreamSplitter) Flush() []schema.ChatMessage {
	return f.process("", true)
}

func (f *thinkingStreamSplitter) process(chunk string, flush bool) []schema.ChatMessage {
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

	var out []schema.ChatMessage
	var current strings.Builder
	currentRole := "assistant"
	if f.inThinking {
		currentRole = "thinking"
	}

	flushCurrent := func() {
		if current.Len() == 0 {
			return
		}
		out = append(out, schema.ChatMessage{Role: currentRole, Content: current.String()})
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

func parseThinkingText(text string) (string, string, bool) {
	if text == "" {
		return "", "", false
	}

	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?s)<think>(.*?)</think>`),
		regexp.MustCompile(`(?s)<reasoning>(.*?)</reasoning>`),
		regexp.MustCompile(`(?s)<scratchpad>(.*?)</scratchpad>`),
		regexp.MustCompile(`(?s)<thought>(.*?)</thought>`),
		regexp.MustCompile(`(?s)<internal>(.*?)</internal>`),
	}

	remaining := text
	var allThinking []string
	for _, pattern := range patterns {
		matches := pattern.FindAllStringSubmatch(remaining, -1)
		for _, match := range matches {
			if len(match) >= 2 {
				allThinking = append(allThinking, strings.TrimSpace(match[1]))
			}
		}
		remaining = pattern.ReplaceAllString(remaining, "")
	}

	if len(allThinking) == 0 {
		return "", text, false
	}

	return strings.Join(allThinking, "\n\n"), strings.TrimSpace(remaining), true
}

func containsThinkingTags(text string) bool {
	if text == "" {
		return false
	}
	for _, tag := range thinkingOpenTags {
		if strings.Contains(text, tag) {
			return true
		}
	}
	for _, tag := range thinkingCloseTags {
		if strings.Contains(text, tag) {
			return true
		}
	}
	return false
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

// stdinHasData returns true if stdin is piped or redirected.
func stdinHasData() bool {
	info, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return (info.Mode() & os.ModeCharDevice) == 0
}
