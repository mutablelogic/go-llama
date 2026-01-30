package main

import (
	"fmt"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	httpclient "github.com/mutablelogic/go-llama/pkg/llamacpp/httpclient"
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type TokenizerCommands struct {
	Tokenize   TokenizeCommand   `cmd:"" name:"tokenize" help:"Convert text to tokens." group:"TOKENIZER"`
	Detokenize DetokenizeCommand `cmd:"" name:"detokenize" help:"Convert tokens to text." group:"TOKENIZER"`
}

type TokenizeCommand struct {
	Model        string `arg:"" name:"model" help:"Model name or path"`
	Text         string `arg:"" name:"text" help:"Text to tokenize"`
	AddSpecial   *bool  `name:"add-special" help:"Add BOS/EOS tokens"`
	ParseSpecial *bool  `name:"parse-special" help:"Parse special tokens in text"`
}

type DetokenizeCommand struct {
	Model          string           `arg:"" name:"model" help:"Model name or path"`
	Tokens         []llamacpp.Token `arg:"" name:"tokens" help:"Tokens to detokenize"`
	RemoveSpecial  *bool            `name:"remove-special" help:"Remove BOS/EOS tokens"`
	UnparseSpecial *bool            `name:"unparse-special" help:"Render special tokens as text"`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *TokenizeCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "TokenizeCommand")
	defer func() { endSpan(err) }()

	// Build options
	opts := []httpclient.Opt{}
	if cmd.AddSpecial != nil {
		opts = append(opts, httpclient.WithAddSpecial(*cmd.AddSpecial))
	}
	if cmd.ParseSpecial != nil {
		opts = append(opts, httpclient.WithParseSpecial(*cmd.ParseSpecial))
	}

	// Tokenize
	result, err := client.Tokenize(parent, cmd.Model, cmd.Text, opts...)
	if err != nil {
		return err
	}

	// Print result
	fmt.Println(result)
	return nil
}

func (cmd *DetokenizeCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "DetokenizeCommand")
	defer func() { endSpan(err) }()

	// Build options
	opts := []httpclient.Opt{}
	if cmd.RemoveSpecial != nil {
		opts = append(opts, httpclient.WithRemoveSpecial(*cmd.RemoveSpecial))
	}
	if cmd.UnparseSpecial != nil {
		opts = append(opts, httpclient.WithUnparseSpecial(*cmd.UnparseSpecial))
	}

	// Detokenize
	result, err := client.Detokenize(parent, cmd.Model, cmd.Tokens, opts...)
	if err != nil {
		return err
	}

	// Print result
	fmt.Println(result.Text)
	return nil
}
