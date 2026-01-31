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
	PullModel   PullModelCommand   `cmd:"" name:"pull" help:"Download a model from URL." group:"MODEL"`
	LoadModel   LoadModelCommand   `cmd:"" name:"load" help:"Load model into memory." group:"MODEL"`
	UnloadModel UnloadModelCommand `cmd:"" name:"unload" help:"Unload model from memory." group:"MODEL"`
}

type ListModelsCommand struct{}

type GetModelCommand struct {
	ID string `arg:"" name:"id" help:"Model ID or path"`
}

type PullModelCommand struct {
	URL      string `arg:"" name:"url" help:"Model URL (supports hf:// and https://)"`
	Progress bool   `name:"progress" help:"Show download progress" default:"true"`
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

func (cmd *PullModelCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "PullModelCommand")
	defer func() { endSpan(err) }()

	// Build options
	opts := []httpclient.Opt{}
	if cmd.Progress {
		opts = append(opts, httpclient.WithProgressCallback(func(filename string, received, total uint64) error {
			if total > 0 {
				pct := float64(received) * 100.0 / float64(total)
				fmt.Printf("\rDownloading %s: %.1f%% (%d/%d bytes)", filename, pct, received, total)
			} else {
				fmt.Printf("\rDownloading %s: %d bytes", filename, received)
			}
			return nil
		}))
	}

	// Pull model
	model, err := client.PullModel(parent, cmd.URL, opts...)
	if err != nil {
		return err
	}

	// Clear progress line and print result
	if cmd.Progress {
		fmt.Printf("\n")
	}
	fmt.Println("Model downloaded successfully:")
	fmt.Println(model)
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
