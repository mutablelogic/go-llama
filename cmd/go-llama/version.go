package main

import (
	"encoding/json"
	"runtime"

	// Packages
	version "github.com/mutablelogic/go-llama/pkg/version"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func VersionJSON(ctx *Globals) string {
	metadata := map[string]string{
		"name":       ctx.execName,
		"compiler":   runtime.Version(),
		"source":     version.GitSource,
		"tag":        version.GitTag,
		"branch":     version.GitBranch,
		"hash":       version.GitHash,
		"build_time": version.GoBuildTime,
	}
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(data)
}
