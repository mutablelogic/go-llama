package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	// Packages
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

	// Get system prompt from argument or stdin
	system := cmd.System
	if system == "" {
		system, err = readStdin()
		if err != nil {
			return err
		}
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

	// Add streaming callback if requested
	if cmd.Stream {
		opts = append(opts, httpclient.WithChunkCallback(func(chunk *schema.CompletionChunk) error {
			fmt.Print(chunk.Text)
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
		fmt.Print(result.Message.Content)
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
			break
		}
	}

	// Handle last message if exists
	if currentRole != "" {
		messages = append(messages, schema.ChatMessage{
			Role:    currentRole,
			Content: strings.TrimSpace(currentContent.String()),
		})
	}

	return messages, nil
}
