# go-llama

[![Go Reference](https://pkg.go.dev/badge/github.com/mutablelogic/go-llama.svg)](https://pkg.go.dev/github.com/mutablelogic/go-llama)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

Go bindings and a unified server/CLI for [llama.cpp](https://github.com/ggerganov/llama.cpp).

Run a local LLM server with a REST API, manage GGUF models, and use the `gollama` CLI for chat, completion, embeddings, and tokenization.

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
docker volume create gollama
docker run -d --name gollama \
  -v gollama:/data -p 8083:8083 \
  ghcr.io/mutablelogic/go-llama run
~~~

Then use the CLI to interact with the server:

~~~bash
export GOLLAMA_ADDR="localhost:8083"

# Pull a model (Hugging Face URL or hf:// scheme)
gollama pull https://huggingface.co/unsloth/phi-4-GGUF/blob/main/phi-4-q4_k_m.gguf

# List models
gollama models

# Load a model into memory
gollama load phi-4-q4_k_m.gguf

# Chat (interactive)
gollama chat phi-4-q4_k_m.gguf "You are a helpful assistant"

# Completion
gollama complete phi-4-q4_k_m.gguf "Explain KV cache in two sentences"
~~~

## Model Support

`gollama` works with GGUF models supported by llama.cpp. Models can be pulled from Hugging Face using:

- `https://huggingface.co/<org>/<repo>/blob/<branch>/<file>.gguf`
- `hf://<org>/<repo>/<file>.gguf`

The default model cache directory is `${XDG_CACHE_HOME}/gollama` (or system temp) and can be overridden with `GOLLAMA_DIR`.

## Docker Deployment

Docker containers are published for Linux AMD64 and ARM64. Variants include:

- **CPU**: `ghcr.io/mutablelogic/go-llama`
- **CUDA**: `ghcr.io/mutablelogic/go-llama-cuda`
- **Vulkan**: `ghcr.io/mutablelogic/go-llama-vulkan`

Use the `run` command inside the container to start the server. For GPU usage, ensure the host has the appropriate drivers and runtime.

## CLI Usage Examples

| Command | Description | Example |
|---------|-------------|---------|
| `models` | List available models | `gollama models` |
| `model` | Get model details | `gollama model phi-4-q4_k_m.gguf` |
| `pull` | Download a model | `gollama pull hf://org/repo/model.gguf` |
| `load` | Load a model into memory | `gollama load phi-4-q4_k_m.gguf` |
| `unload` | Unload a model from memory | `gollama unload phi-4-q4_k_m.gguf` |
| `delete` | Delete a model | `gollama delete phi-4-q4_k_m.gguf` |
| `chat` | Interactive chat | `gollama chat phi-4-q4_k_m.gguf "system"` |
| `complete` | Text completion | `gollama complete phi-4-q4_k_m.gguf "prompt"` |
| `embed` | Generate embeddings | `gollama embed phi-4-q4_k_m.gguf "text"` |
| `tokenize` | Convert text to tokens | `gollama tokenize phi-4-q4_k_m.gguf "text"` |
| `detokenize` | Convert tokens to text | `gollama detokenize phi-4-q4_k_m.gguf 1 2 3` |
| `run` | Run the HTTP server | `gollama run --http.addr localhost:8083` |

Use `gollama --help` or `gollama <command> --help` for full options.

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
make gollama

# Build client-only binary
make gollama-client

# Build Docker images
make docker
~~~

Use `GGML_CUDA=1` or `GGML_VULKAN=1` to build GPU variants.

## Contributing & License

Please file issues and feature requests in GitHub issues. Licensed under Apache 2.0.
