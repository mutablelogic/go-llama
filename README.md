# go-llama

[![Go Reference](https://pkg.go.dev/badge/github.com/mutablelogic/go-llama.svg)](https://pkg.go.dev/github.com/mutablelogic/go-llama)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

Go bindings and a unified server/CLI for [llama.cpp](https://github.com/ggerganov/llama.cpp).

Run a local LLM server with a REST API, manage GGUF models, and use the `go-llama` CLI for chat, completion, embeddings, and tokenization.

## Features

- **Command Line Interface**: Interactive chat and completion tooling
- **HTTP API Server**: REST endpoints for chat, completion, embeddings, and model management (not yet OpenAI or Anthropic compatible)
- **Model Management**: Pull, cache, load, unload, and delete GGUF models
- **Streaming**: Incremental token streaming for chat and completion
- **GPU Support**: CUDA, Vulkan, and Metal (macOS) acceleration via llama.cpp
- **Docker Support**: Pre-built images for CPU, CUDA, and Vulkan targets

## Quick Start

Start the server with Docker:

~~~bash
docker volume create go-llama
docker run -d --name go-llama \
  -v go-llama:/data -p 8083:8083 \
  ghcr.io/mutablelogic/go-llama run
~~~

Then use the CLI to interact with the server:

~~~bash
export GOLLAMA_ADDR="localhost:8083"

# Pull a model (Hugging Face URL or hf:// scheme)
go-llama pull https://huggingface.co/unsloth/phi-4-GGUF/blob/main/phi-4-q4_k_m.gguf

# List models
go-llama models

# Load a model into memory
go-llama load phi-4-q4_k_m.gguf

# Chat (interactive)
go-llama chat phi-4-q4_k_m.gguf "You are a helpful assistant"
# Completion
go-llama complete phi-4-q4_k_m.gguf "Explain KV cache in two sentences"
~~~

## Model Support

`go-llama` works with GGUF models supported by llama.cpp. Models can be pulled from Hugging Face using:

- `https://huggingface.co/<org>/<repo>/blob/<branch>/<file>.gguf`
- `hf://<org>/<repo>/<file>.gguf`

The default model cache directory is the system cache folder (e.g., `~/.cache/go-llama` on Linux, `~/Library/Caches/go-llama` on macOS) and can be overridden with `GOLLAMA_DIR` or `--models`.

## Docker Deployment

Docker containers are published for Linux AMD64 and ARM64. Variants include:

- **CPU**: `ghcr.io/mutablelogic/go-llama`
- **CUDA**: `ghcr.io/mutablelogic/go-llama-cuda`
- **Vulkan**: `ghcr.io/mutablelogic/go-llama-vulkan`

Use the `run` command inside the container to start the server. For GPU usage, ensure the host has the appropriate drivers and runtime.

## CLI Usage Examples

| Command | Description | Example |
|---------|-------------|---------|
| `gpuinfo` | Show GPU information | `go-llama gpuinfo` |
| `models` | List available models | `go-llama models` |
| `model` | Get model details | `go-llama model phi-4-q4_k_m.gguf` |
| `pull` | Download a model | `go-llama pull hf://org/repo/model.gguf` |
| `load` | Load a model into memory | `go-llama load phi-4-q4_k_m.gguf` |
| `unload` | Unload a model from memory | `go-llama unload phi-4-q4_k_m.gguf` |
| `delete` | Delete a model | `go-llama delete phi-4-q4_k_m.gguf` |
| `chat` | Interactive chat | `go-llama chat phi-4-q4_k_m.gguf "system"` |
| `complete` | Text completion | `go-llama complete phi-4-q4_k_m.gguf "prompt"` |
| `embed` | Generate embeddings | `go-llama embed phi-4-q4_k_m.gguf "text"` |
| `tokenize` | Convert text to tokens | `go-llama tokenize phi-4-q4_k_m.gguf "text"` |
| `detokenize` | Convert tokens to text | `go-llama detokenize phi-4-q4_k_m.gguf 1 2 3` |
| `run` | Run the HTTP server | `go-llama run --http.addr localhost:8083` |

Use `go-llama --help` or `go-llama <command> --help` for full options.

## Development

### Project Structure

- `cmd` contains the CLI and server entrypoint
- `pkg/llamacpp` contains the high-level service and HTTP handlers
  - `httpclient/` - client for the server API
  - `httphandler/` - HTTP handlers and routing
  - `schema/` - API types
- `sys/llamacpp` contains native bindings to llama.cpp
- `sys/gguf` contains GGUF parsing helpers
- `third_party/llama.cpp` is the upstream llama.cpp submodule
- `etc/` contains Dockerfiles

### Building

~~~bash
# Build server binary
make go-llama

# Build client-only binary
make go-llama-client

# Build Docker images
make docker
~~~

Use `GGML_CUDA=1` or `GGML_VULKAN=1` to build GPU variants.

## Contributing & License

Please file issues and feature requests in GitHub issues. Licensed under Apache 2.0.
