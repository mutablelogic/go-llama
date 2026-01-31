package main

import (
	"encoding/json"
	"fmt"
	"os"
	"text/tabwriter"

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
	DeleteModel DeleteModelCommand `cmd:"" name:"delete" help:"Delete model from disk." group:"MODEL"`
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

type DeleteModelCommand struct {
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
	if ctx.Debug {
		if b, err := json.MarshalIndent(models, "", "  "); err == nil {
			fmt.Fprintln(os.Stderr, string(b))
		}
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "PATH\tNAME\tLOADED\tPARAMS\tSIZE\tCTX_TRAIN")
	for _, model := range models {
		loaded := "no"
		params := "-"
		size := "-"
		ctxTrain := "-"
		if !model.LoadedAt.IsZero() {
			loaded = "yes"
			if model.Runtime != nil {
				if model.Runtime.NParams > 0 {
					params = formatParams(model.Runtime.NParams)
				}
				if model.Runtime.ModelSize > 0 {
					size = formatBytes(model.Runtime.ModelSize)
				}
				if model.Runtime.NCtxTrain > 0 {
					ctxTrain = fmt.Sprintf("%d", model.Runtime.NCtxTrain)
				}
			}
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", model.Path, model.Name, loaded, params, size, ctxTrain)
	}
	_ = w.Flush()
	return nil
}

func formatBytes(bytes uint64) string {
	if bytes == 0 {
		return "0B"
	}
	units := []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
	size := float64(bytes)
	unit := 0
	for size >= 1024.0 && unit < len(units)-1 {
		size /= 1024.0
		unit++
	}
	if size >= 10.0 || unit == 0 {
		return fmt.Sprintf("%.0f%s", size, units[unit])
	}
	return fmt.Sprintf("%.1f%s", size, units[unit])
}

func formatParams(params uint64) string {
	if params == 0 {
		return "0"
	}
	units := []string{"", "K", "M", "B", "T"}
	value := float64(params)
	unit := 0
	for value >= 1000.0 && unit < len(units)-1 {
		value /= 1000.0
		unit++
	}
	if value >= 10.0 || unit == 0 {
		return fmt.Sprintf("%.0f%s", value, units[unit])
	}
	return fmt.Sprintf("%.1f%s", value, units[unit])
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
		return fmt.Errorf("failed to pull model from %q: %w", cmd.URL, err)
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
	model, err := client.UnloadModel(parent, cmd.ID)
	if err != nil {
		return err
	}

	// Print result
	fmt.Println(model)
	return nil
}

func (cmd *DeleteModelCommand) Run(ctx *Globals) (err error) {
	client, err := ctx.Client()
	if err != nil {
		return err
	}

	// OTEL
	parent, endSpan := otel.StartSpan(ctx.tracer, ctx.ctx, "DeleteModelCommand")
	defer func() { endSpan(err) }()

	// Delete model
	if err := client.DeleteModel(parent, cmd.ID); err != nil {
		return err
	}

	// Print success message
	fmt.Printf("Model %s deleted successfully\n", cmd.ID)
	return nil
}
