# Test Data

This directory contains small model files for testing.

## stories260K.gguf

A tiny 260K parameter LLaMA model trained on children's stories.

- **Source:** [ggml-org/models-moved](https://huggingface.co/ggml-org/models-moved/tree/main/tinyllamas) on Hugging Face
- **Original:** [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) - Andrej Karpathy's TinyStories models
- **Size:** ~1.1 MB
- **Parameters:** 260K
- **Training:** Trained on [TinyStories](https://arxiv.org/abs/2305.07759) dataset

This model is intentionally small for fast CI testing. It generates coherent-ish children's story text but is not suitable for real use.
