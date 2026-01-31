//go:build !client

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync"

	// Packages
	otel "github.com/mutablelogic/go-client/pkg/otel"
	pkg "github.com/mutablelogic/go-llama/pkg/llamacpp"
	httphandler "github.com/mutablelogic/go-llama/pkg/llamacpp/httphandler"
	version "github.com/mutablelogic/go-llama/pkg/version"
	httpserver "github.com/mutablelogic/go-server/pkg/httpserver"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type ServerCommands struct {
	GpuInfo   GpuInfoCmd `cmd:"" name:"gpuinfo" help:"Show GPU information"`
	RunServer RunServer  `cmd:"" name:"run" help:"Run server." group:"SERVER"`
}

type RunServer struct {
	Models string `name:"models" env:"GOLLAMA_DIR" help:"Models directory path" default:""`

	// TLS server options
	TLS struct {
		ServerName string `name:"name" help:"TLS server name"`
		CertFile   string `name:"cert" help:"TLS certificate file"`
		KeyFile    string `name:"key" help:"TLS key file"`
	} `embed:"" prefix:"tls."`
}

///////////////////////////////////////////////////////////////////////////////
// COMMANDS

func (cmd *RunServer) Run(ctx *Globals) error {
	// Set models path - use default if not specified
	modelsPath := cmd.Models
	if modelsPath == "" {
		// Use default from environment or cache dir
		if dir, err := os.UserCacheDir(); err == nil {
			modelsPath = filepath.Join(dir, "gollama")
		} else {
			modelsPath = filepath.Join(os.TempDir(), "gollama")
		}
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(modelsPath, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	// Report models path
	ctx.logger.With("models", modelsPath).Print(ctx.ctx, "using models directory")

	// Build options
	managerOpts := []pkg.Opt{}
	if ctx.tracer != nil {
		managerOpts = append(managerOpts, pkg.WithTracer(ctx.tracer))
	}

	// Create the whisper manager
	manager, err := pkg.New(modelsPath, managerOpts...)
	if err != nil {
		return err
	}
	defer func() {
		if err := manager.Close(); err != nil {
			ctx.logger.Print(ctx.ctx, "error closing manager: ", err)
		}
	}()

	// Set logging middleware
	middleware := httphandler.HTTPMiddlewareFuncs{
		ctx.logger.HandleFunc,
	}

	// If we have an OTEL tracer, add tracing middleware
	if ctx.tracer != nil {
		middleware = append(middleware, otel.HTTPHandlerFunc(ctx.tracer))
	}

	// Register HTTP handlers
	router := http.NewServeMux()
	prefix, err := url.JoinPath(ctx.HTTP.Prefix, "gollama")
	if err != nil {
		return err
	}
	httphandler.RegisterHandlers(router, prefix, manager, middleware)

	// Create a TLS config
	var tlsconfig *tls.Config
	if cmd.TLS.CertFile != "" || cmd.TLS.KeyFile != "" {
		tlsconfig, err = httpserver.TLSConfig(cmd.TLS.ServerName, true, cmd.TLS.CertFile, cmd.TLS.KeyFile)
		if err != nil {
			return err
		}
	}

	// Create a HTTP server with timeouts
	httpopts := []httpserver.Opt{}
	if ctx.HTTP.Timeout > 0 {
		httpopts = append(httpopts, httpserver.WithReadTimeout(ctx.HTTP.Timeout))
		httpopts = append(httpopts, httpserver.WithWriteTimeout(ctx.HTTP.Timeout))
	}
	server, err := httpserver.New(ctx.HTTP.Addr, router, tlsconfig, httpopts...)
	if err != nil {
		return err
	}

	// We run the server
	var wg sync.WaitGroup
	var result error

	// Output the version
	if version.GitTag != "" {
		ctx.logger.Printf(ctx.ctx, "gollama@%s", version.GitTag)
	} else if version.GitHash != "" {
		ctx.logger.Printf(ctx.ctx, "gollama@%s", version.GitHash[:8])
	} else {
		ctx.logger.Printf(ctx.ctx, "gollama")
	}

	// Run the HTTP server
	wg.Add(1)
	go func() {
		defer wg.Done()

		// Output listening information
		ctx.logger.With("addr", ctx.HTTP.Addr, "prefix", ctx.HTTP.Prefix).Print(ctx.ctx, "http server starting")

		// Run the server
		if err := server.Run(ctx.ctx); err != nil {
			if !errors.Is(err, context.Canceled) {
				result = errors.Join(result, fmt.Errorf("http server error: %w", err))
			}
			ctx.cancel()
		}
	}()

	// Wait for goroutine to finish
	wg.Wait()

	// Terminated message
	if result == nil {
		ctx.logger.With("addr", ctx.HTTP.Addr, "prefix", ctx.HTTP.Prefix).Print(ctx.ctx, "terminated gracefully")
	} else {
		ctx.logger.With("addr", ctx.HTTP.Addr, "prefix", ctx.HTTP.Prefix).Print(ctx.ctx, result)
	}

	// Return any error
	return result
}
