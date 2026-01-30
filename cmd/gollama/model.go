package main

import (
	"fmt"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	httpclient "github.com/mutablelogic/go-llama/pkg/llamacpp/httpclient"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type ModelCommands struct {
	ListModels  ListModelsCommand  `cmd:"" name:"models" help:"List models." group:"MODEL"`
	GetModel    GetModelCommand    `cmd:"" name:"model" help:"Get model." group:"MODEL"`
	LoadModel   LoadModelCommand   `cmd:"" name:"load-model" help:"Load model into memory." group:"MODEL"`
	UnloadModel UnloadModelCommand `cmd:"" name:"unload-model" help:"Unload model from memory." group:"MODEL"`
}

type ListModelsCommand struct{}

type GetModelCommand struct {
	ID string `arg:"" name:"id" help:"Model ID or path"`
}

type LoadModelCommand struct {
	Name   string `arg:"" name:"name" help:"Model name or path"`
	Gpu    *int32 `name:"gpu" help:"Main GPU index"`
	Layers *int32 `name:"layers" help:"Number of layers to offload to GPU (-1 = all)"`
	Mmap   *bool  `name:"mmap" help:"Use memory mapping for model loading"`
	Mlock  *bool  `name:"mlock" help:"Lock model in memory"`
}

type UnloadModelCommand struct {
	ID string `arg:"" name:"id" help:"Model ID or path"`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *ListModelsCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "ListModelsCommand")
	defer func() { endSpan(err) }()

	// List models
	models, err := client.ListModels(parent)
	if err != nil {
		return err
	}

	// Print
	for _, model := range models {
		fmt.Println(model)
	}
	return nil
}

func (cmd *GetModelCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "GetModelCommand")
	defer func() { endSpan(err) }()

	// Get model
	model, err := client.GetModel(parent, cmd.ID)
	if err != nil {
		return err
	}

	// Print
	fmt.Println(model)
	return nil
}

func (cmd *LoadModelCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "LoadModelCommand")
	defer func() { endSpan(err) }()

	// Build options
	opts := []httpclient.Opt{}
	if cmd.Gpu != nil {
		opts = append(opts, httpclient.WithGpu(*cmd.Gpu))
	}
	if cmd.Layers != nil {
		opts = append(opts, httpclient.WithLayers(*cmd.Layers))
	}
	if cmd.Mmap != nil {
		opts = append(opts, httpclient.WithMmap(*cmd.Mmap))
	}
	if cmd.Mlock != nil {
		opts = append(opts, httpclient.WithMlock(*cmd.Mlock))
	}

	// Load model
	model, err := client.LoadModel(parent, cmd.Name, opts...)
	if err != nil {
		return err
	}

	// Print
	fmt.Println(model)
	return nil
}

func (cmd *UnloadModelCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "UnloadModelCommand")
	defer func() { endSpan(err) }()

	// Unload model
	err = client.UnloadModel(parent, cmd.ID)
	if err != nil {
		return err
	}

	// Print success message
	fmt.Printf("Model %s unloaded successfully\n", cmd.ID)
	return nil
}
