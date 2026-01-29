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

## tall-MiniLM-L6-v2-Q4_K_M.gguf

A small 22 MB quantized version of the MiniLM-L6-v2 model for embeddings.

- **Source:** [second-state/All-MiniLM-L6-v2-Embedding-GGUF](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) on Hugging Face
- **Original:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence Transformers
- **Size:** ~22 MB
- **Dimensions:** 384-dimensional embeddings
- **Quantization:** Q4_K_M format for reduced size
