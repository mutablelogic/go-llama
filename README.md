# go-llama

[![Go Reference](https://pkg.go.dev/badge/github.com/mutablelogic/go-llama.svg)](https://pkg.go.dev/github.com/mutablelogic/go-llama)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

Go bindings and a unified server/CLI for [llama.cpp](https://github.com/ggerganov/llama.cpp).

Run a local LLM server with a REST API, manage GGUF models, and use the `gollama` CLI for chat, completion, embeddings, and tokenization.

## Features

- **Command Line Interface**: Interactive chat and completion tooling
- **HTTP API Server**: REST endpoints for chat, completion, embeddings, and model management
- **Model Management**: Pull, cache, load, unload, and delete GGUF models
- **Streaming**: Incremental token streaming for chat and completion
- **GPU Support**: CUDA, Vulkan, and Metal (macOS) acceleration via llama.cpp
- **Docker Support**: Pre-built images for CPU, CUDA, and Vulkan targets

Some work still to do on the chat endpoint. The following are not yet included, but will eventually be supported:

- Multi-modal support (images, audio, PDF's, etc)
- Reasoning/Thinking support
- OpenAI or Anthropic compatible API
- Tool calling
- Grammar (JSON format output)
- Text-to-Speech (Audio output)

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
gollama pull hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# List models
gollama models

# Load a model into memory
gollama load Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Chat (interactive)
gollama chat Llama-3.2-1B-Instruct-Q4_K_M.gguf "You are a helpful assistant"

# Completion
gollama complete Llama-3.2-1B-Instruct-Q4_K_M.gguf "Explain KV cache in two sentences"
~~~

## Model Support

`gollama` works with GGUF models supported by llama.cpp. Models can be pulled from Hugging Face using:

- `https://huggingface.co/<org>/<repo>/blob/<branch>/<file>.gguf`
- `hf://<org>/<repo>/<file>.gguf`

The default model cache directory is `${XDG_CACHE_HOME}/gollama` (or system temp) and can be overridden with `GOLLAMA_DIR`.

## Docker Deployment

Docker containers are published for Linux AMD64 and ARM64. Variants include:

- **CPU and Vulkan**: `ghcr.io/mutablelogic/go-llama`
- **CUDA**: `ghcr.io/mutablelogic/go-llama-cuda`

Use the `run` command inside the container to start the server. For GPU usage, ensure the host has the appropriate drivers and runtime.

## CLI Usage Examples

Client-only commands:

| Command | Description | Example |
|---------|-------------|---------|
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

Use `go-llama --help` or `go-llama <command> --help` for full options. Server commands:

| Command | Description | Example |
|---------|-------------|---------|
| `gpuinfo` | Show GPU information | `go-llama gpuinfo` |
| `run` | Run the HTTP server | `go-llama run --http.addr localhost:8083` |

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
