# BDH Project TODO List

This document outlines the planned enhancements to make the Baby Dragon Hatchling (BDH) architecture production-ready for training on diverse datasets and deploying models for local inference.

## 🚀 Phase 1: Core Training Infrastructure

### 1. Add HuggingFace Datasets Integration
**Priority: High**

Implement data loading from HuggingFace datasets hub. Create a new `data.py` module with functions to load and preprocess datasets (text datasets like WikiText, OpenWebText, etc.). Support both character-level and tokenizer-based approaches. Update `train.py` to accept dataset configuration via command line arguments or config file.

**Key Tasks:**
- Create `data.py` module with dataset loading utilities
- Support streaming datasets for large corpora
- Implement data preprocessing and batching
- Add dataset configuration options

### 2. Implement Tokenizer Support
**Priority: High**

Add support for standard tokenizers (BPE, WordPiece, SentencePiece) via HuggingFace tokenizers library. Currently the model uses byte-level encoding (vocab_size=256). Add tokenizer configuration to BDHConfig, update embedding layers to handle larger vocabularies, and create utilities to save/load tokenizer with model.

**Key Tasks:**
- Extend `BDHConfig` with tokenizer parameters
- Update embedding and output layers for variable vocab sizes
- Create tokenizer training utilities
- Implement tokenizer save/load functionality

### 3. Implement Checkpoint Saving/Loading During Training
**Priority: High**

Add checkpoint management to `train.py`: save model state_dict, optimizer state_dict, training step, loss history, and config at regular intervals. Implement resume functionality to continue training from saved checkpoints. Use PyTorch's `torch.save`/`torch.load`. Store checkpoints in a configurable directory with timestamped or step-numbered filenames.

**Key Tasks:**
- Create checkpoint saving function
- Implement checkpoint loading with state restoration
- Add automatic checkpoint cleanup (keep last N checkpoints)
- Support checkpoint averaging

### 4. Add Training Monitoring and Logging
**Priority: Medium**

Integrate Weights & Biases or TensorBoard for experiment tracking. Log training loss, validation loss, learning rate, gradient norms, and sample generations. Add evaluation loop that runs periodically during training to compute validation metrics (perplexity, loss). This will help monitor long training runs.

**Key Tasks:**
- Add W&B or TensorBoard integration
- Implement validation evaluation loop
- Log comprehensive metrics
- Add sample generation during training

## 📦 Phase 2: Model Export and Inference

### 5. Implement Final Model Export Functionality
**Priority: High**

Create a `save_pretrained()` method similar to HuggingFace transformers that saves: model weights, config.json, tokenizer files, and a README with model info. Create corresponding `load_pretrained()` class method. This will be the standard format for distributing trained BDH models.

**Key Tasks:**
- Implement `save_pretrained()` method in BDH class
- Create `from_pretrained()` class method
- Generate model cards with training info
- Support safetensors format

### 6. Create Inference Script with Generation Utilities
**Priority: High**

Build a standalone `inference.py` script that loads a trained model and provides interactive text generation. Support various sampling strategies (temperature, top-k, top-p/nucleus sampling, beam search). Add options for batch generation and streaming output. This will be useful for testing models locally.

**Key Tasks:**
- Create interactive CLI for text generation
- Implement advanced sampling methods (nucleus, beam search)
- Add streaming generation support
- Create simple Python API for programmatic use

## 🔧 Phase 3: Deployment (PyTorch-Based)

> **Architecture Note:** BDH's novel architecture (unique attention mechanism, sparse activations, custom parameter structure) is **not compatible with GGUF/llama.cpp** without significant custom C++ work. The recommended deployment path uses PyTorch natively with an optional API wrapper.

### 7. Implement OpenAI-Compatible API Server
**Priority: High**

Build a FastAPI-based server (`serve.py`) that wraps BDH inference in OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`). Support streaming responses, multiple concurrent requests, and standard parameters (temperature, max_tokens, stop sequences). Include proper error handling and rate limiting. Add Docker support for easy deployment.

**This is the recommended deployment solution for BDH**, providing:
- Native PyTorch inference (no architecture conversion needed)
- Standard API interface for easy integration
- Simple deployment with Docker
- Support for model quantization and optimization

**Key Tasks:**
- Create FastAPI application with OpenAI endpoints
- Implement streaming SSE responses
- Add request queuing and batching
- Create Dockerfile and docker-compose setup
- Add authentication and rate limiting
- Support FP16/INT8 quantization for faster inference

## ⚡ Phase 4: Optimization and Scaling

### 8. Add Distributed Training Support
**Priority: Medium**

Implement multi-GPU training using PyTorch DDP (DistributedDataParallel) or FSDP (Fully Sharded Data Parallel) for training larger models on multiple GPUs or nodes. Add gradient accumulation for effective larger batch sizes. This is important for scaling up to larger models and datasets.

**Key Tasks:**
- Implement DDP training mode
- Add FSDP support for large models
- Implement gradient accumulation
- Create multi-node training scripts

### 9. Create Comprehensive Configuration System
**Priority: High**

Replace hardcoded training hyperparameters with a flexible configuration system using YAML/JSON files or dataclasses. Support configuration for: model architecture, training hyperparameters, data loading, checkpointing, and logging. Add CLI argument parsing with configuration file support. Use hydra or simple argparse + yaml.

**Key Tasks:**
- Design configuration schema
- Implement config loading/validation
- Add CLI argument parsing
- Create example config files for common scenarios

### 10. Add Model Quantization Support
**Priority: Medium**

Implement post-training quantization (INT8, INT4) for reduced model size and faster inference. Support bitsandbytes quantization for efficient loading of large models. Add quantization-aware training option. This is crucial for running larger models on consumer hardware.

**Key Tasks:**
- Implement PTQ (Post-Training Quantization)
- Add bitsandbytes integration
- Support QAT (Quantization-Aware Training)
- Benchmark quantized models

## 📚 Phase 5: Documentation and Examples

### 11. Write Comprehensive Documentation
**Priority: High**

Create detailed documentation covering: training custom models, dataset preparation, hyperparameter tuning, checkpoint management, model export formats, inference options, and API server deployment. Add examples for common use cases. Update README.md with links to documentation.

**Key Tasks:**
- Create `docs/` directory structure
- Write training guide
- Document inference options
- Create deployment guides (PyTorch-based)
- Add API reference documentation
- Document why GGUF is not suitable for BDH architecture

### 12. Create Example Notebooks and Tutorials
**Priority: Medium**

Develop Jupyter notebooks demonstrating: fine-tuning BDH on custom datasets, using pretrained models for inference, analyzing attention patterns and interpretability, comparing BDH to Transformers on benchmarks. These will help users understand and adopt the architecture.

**Key Tasks:**
- Create `notebooks/` directory
- Tutorial: Fine-tuning on custom data
- Tutorial: Model inference and sampling
- Tutorial: Interpretability analysis
- Tutorial: Architecture comparison benchmarks

---

## 🎯 Recommended Implementation Order

For getting started quickly with a production-ready system, implement in this order:

### **Essential Foundation** (Complete These First)
1. **Configuration System** (Item 9) - Makes everything else easier to implement
2. **Checkpoint Management** (Item 3) - Essential for long training runs
3. **HuggingFace Datasets** (Item 1) - Enables training on diverse datasets
4. **Tokenizer Support** (Item 2) - Required for most real-world applications

### **Usability** (Next Priority)
5. **Model Export** (Item 5) - Enables sharing and reusing trained models
6. **Inference Script** (Item 6) - Makes models actually usable locally

### **Deployment** (Production Ready)
7. **OpenAI-Compatible API** (Item 7) - Simple deployment with standard interface
8. **Model Quantization** (Item 10) - Optimize for consumer hardware

### **Optional Enhancements**
9. Training Monitoring (Item 4)
10. Distributed Training (Item 8)
11. Documentation (Items 11-12)

## 📝 Important Notes

### Current Limitations
- The current implementation uses byte-level encoding (vocab_size=256), which is limiting for most applications
- Training currently uses a hardcoded Shakespeare dataset
- No checkpoint saving means training runs can't be resumed
- No configuration system - all parameters are hardcoded

### Architecture-Specific Considerations
- **BDH is NOT compatible with GGUF/llama.cpp** due to its novel architecture
- The attention mechanism is fundamentally different from standard transformers:
  - No softmax normalization
  - K and Q are shared (K is Q)
  - Unique sparse activation pattern
- **Recommended deployment: PyTorch-based inference** with optional FastAPI wrapper
- Custom parameter structure (encoder/decoder/encoder_v) doesn't map to transformer blocks
- vLLM and Ollama integration would require extensive custom work (not recommended)

### Why Not GGUF?
GGUF/llama.cpp expects standard transformer architectures. BDH uses:
1. Non-standard attention: `(QR @ KR.mT).tril(diagonal=-1) @ V` instead of `softmax(QK/√d)V`
2. Element-wise multiplication of sparse representations
3. No attention softmax normalization
4. Custom parameter organization

Adding BDH support to llama.cpp would require:
- Implementing custom attention in C++
- Adding new architecture type to GGUF spec
- Maintaining compatibility across llama.cpp updates
- Much more complex than PyTorch inference

## 🤝 Contributing

When implementing these TODOs, consider:
- Maintaining the interpretability that makes BDH unique
- Keeping the codebase simple and research-friendly
- Adding comprehensive tests for new functionality
- Documenting any architecture-specific considerations
