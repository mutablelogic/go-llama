package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	// Packages
	kong "github.com/alecthomas/kong"
	otel "github.com/mutablelogic/go-client/pkg/otel"
	server "github.com/mutablelogic/go-server"
	logger "github.com/mutablelogic/go-server/pkg/logger"
	trace "go.opentelemetry.io/otel/trace"
	terminal "golang.org/x/term"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type Globals struct {
	// Debug option
	Debug   bool             `name:"debug" help:"Enable debug logging"`
	Version kong.VersionFlag `name:"version" help:"Print version and exit"`

	// HTTP server options
	HTTP struct {
		Prefix  string        `name:"prefix" help:"HTTP path prefix" default:"/api"`
		Addr    string        `name:"addr" env:"GOLLAMA_ADDR" help:"HTTP Listen address" default:"localhost:8083"`
		Timeout time.Duration `name:"timeout" help:"HTTP server read/write timeout" default:"10m"`
	} `embed:"" prefix:"http."`

	// Open Telemetry options
	OTel struct {
		Endpoint string `env:"OTEL_EXPORTER_OTLP_ENDPOINT" help:"Open Telemetry endpoint" default:""`
		Header   string `env:"OTEL_EXPORTER_OTLP_HEADERS" help:"OpenTelemetry collector headers"`
		Name     string `env:"OTEL_SERVICE_NAME" help:"OpenTelemetry service name" default:"${EXECUTABLE_NAME}"`
	} `embed:"" prefix:"otel."`

	// Private fields
	ctx    context.Context
	cancel context.CancelFunc
	logger server.Logger
	tracer trace.Tracer
}

type CLI struct {
	Globals
	GpuInfo GpuInfoCmd `cmd:"gpuinfo" help:"Show GPU information"`
	ServerCommands
}

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func main() {
	// Get executable name
	execName := "gollama"
	if exe, err := os.Executable(); err == nil {
		execName = filepath.Base(exe)
	}

	// Parse command-line arguments
	cli := new(CLI)
	ctx := kong.Parse(cli,
		kong.Name(execName),
		kong.Description("gollama command line interface"),
		kong.Vars{
			"version":         VersionJSON(),
			"EXECUTABLE_NAME": execName,
		},
		kong.UsageOnError(),
		kong.ConfigureHelp(kong.HelpOptions{
			Compact: true,
		}),
	)

	// Run the command
	os.Exit(run(ctx, &cli.Globals))
}

func run(ctx *kong.Context, globals *Globals) int {
	parent := context.Background()

	// Create Logger
	if isTerminal(os.Stderr) {
		globals.logger = logger.New(os.Stderr, logger.Term, globals.Debug)
	} else {
		globals.logger = logger.New(os.Stderr, logger.JSON, globals.Debug)
	}

	// Create the context and cancel function
	globals.ctx, globals.cancel = signal.NotifyContext(parent, os.Interrupt)
	defer globals.cancel()

	// Open Telemetry
	if globals.OTel.Endpoint != "" {
		provider, err := otel.NewProvider(globals.OTel.Endpoint, globals.OTel.Header, globals.OTel.Name)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
			return -2
		}
		defer provider.Shutdown(context.Background())

		// Store tracer for creating spans
		globals.tracer = provider.Tracer(globals.OTel.Name)
	}

	// Call the Run() method of the selected parsed command.
	if err := ctx.Run(globals); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		return -1
	}

	return 0
}

func isTerminal(w io.Writer) bool {
	if fd, ok := w.(*os.File); ok {
		return terminal.IsTerminal(int(fd.Fd()))
	}
	return false
}
