package main

import (
	"fmt"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	httpclient "github.com/mutablelogic/go-llama/pkg/llamacpp/httpclient"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type EmbedCommands struct {
	Embed EmbedCommand `cmd:"" name:"embed" help:"Generate embeddings." group:"EMBEDDING"`
}

type EmbedCommand struct {
	Model     string   `arg:"" name:"model" help:"Model name or path"`
	Input     []string `arg:"" name:"input" help:"Text(s) to embed"`
	Normalize *bool    `name:"normalize" help:"L2-normalize embeddings"`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *EmbedCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "EmbedCommand")
	defer func() { endSpan(err) }()

	// Build options
	opts := []httpclient.Opt{}
	if cmd.Normalize != nil {
		opts = append(opts, httpclient.WithNormalize(*cmd.Normalize))
	}

	// Embed
	result, err := client.Embed(parent, cmd.Model, cmd.Input, opts...)
	if err != nil {
		return err
	}

	// Print result
	fmt.Println(result)
	return nil
}
