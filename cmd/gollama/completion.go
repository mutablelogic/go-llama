package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	httpclient "github.com/mutablelogic/go-llama/pkg/llamacpp/httpclient"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type CompletionCommands struct {
	Complete CompleteCommand `cmd:"" name:"complete" help:"Generate text completion." group:"COMPLETION"`
}

type CompleteCommand struct {
	Model       string   `arg:"" name:"model" help:"Model name or path"`
	Prompt      string   `arg:"" name:"prompt" optional:"" help:"Prompt text (or use stdin)"`
	MaxTokens   *int32   `name:"max-tokens" help:"Maximum tokens to generate"`
	Temperature *float32 `name:"temperature" help:"Sampling temperature (0-2)"`
	TopP        *float32 `name:"top-p" help:"Nucleus sampling parameter (0-1)"`
	TopK        *int32   `name:"top-k" help:"Top-k sampling parameter"`
	Seed        *uint32  `name:"seed" help:"RNG seed for reproducibility"`
	Stop        []string `name:"stop" help:"Stop sequences"`
	PrefixCache *bool    `name:"prefix-cache" help:"Enable prefix caching"`
	Stream      bool     `name:"stream" help:"Stream output tokens" default:"true"`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *CompleteCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "CompleteCommand")
	defer func() { endSpan(err) }()

	// Get prompt from argument or stdin
	prompt := cmd.Prompt
	if prompt == "" {
		prompt, err = readStdin()
		if err != nil {
			return err
		}
	}
	if prompt == "" {
		return fmt.Errorf("prompt is required")
	}

	// Build options
	opts := []httpclient.Opt{}
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
	if cmd.Seed != nil {
		opts = append(opts, httpclient.WithSeed(*cmd.Seed))
	}
	if len(cmd.Stop) > 0 {
		opts = append(opts, httpclient.WithStop(cmd.Stop...))
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

	// Complete
	result, err := client.Complete(parent, cmd.Model, prompt, opts...)
	if err != nil {
		return err
	}

	// Print result if not streaming (streaming already printed chunks)
	if !cmd.Stream {
		fmt.Print(result.Text)
	}
	fmt.Println() // Ensure newline at end

	return nil
}

// readStdin reads all input from stdin if it's not a terminal
func readStdin() (string, error) {
	info, err := os.Stdin.Stat()
	if err != nil {
		return "", err
	}

	// Check if stdin has data (pipe or redirect)
	if (info.Mode() & os.ModeCharDevice) != 0 {
		// stdin is a terminal, no piped input
		return "", nil
	}

	// Read from stdin
	reader := bufio.NewReader(os.Stdin)
	var builder strings.Builder
	for {
		line, err := reader.ReadString('\n')
		builder.WriteString(line)
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}

	return strings.TrimSpace(builder.String()), nil
}
