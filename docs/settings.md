# BDH Configuration Settings Documentation

This document describes all configuration settings in the BDH (Byte-level Dense Hybrid) model training system, their impact on training performance, and their effect on the final model.

---

## Table of Contents
1. [Model Configuration (`BDHConfig`)](#model-configuration-bdhconfig)
2. [Training Configuration (`train.py`)](#training-configuration-trainpy)
3. [System Configuration](#system-configuration)
4. [Quick Reference Tables](#quick-reference-tables)

---

## Model Configuration (`BDHConfig`)

These settings are defined in `bdh.py` and control the model architecture and size.

### `n_layer: int = 6`

**Description:** Number of transformer layers in the model.

**Impact on Training:**
- **Memory Usage:** Linear scaling - each layer adds activation memory during forward/backward passes
- **Training Speed:** Linear scaling - more layers = slower training
- **Gradient Flow:** Deeper models may have gradient flow challenges

**Impact on Final Model:**
- **Parameter Count:** **None** - weights are shared across all layers in this architecture
- **Model Capacity:** Higher capacity with more layers, can learn more complex patterns
- **Inference Speed:** Slower inference with more layers
- **Model Quality:** Generally better with more layers (up to a point)

**Typical Values:** 4-12 layers
- Small models: 4-6 layers
- Medium models: 8-12 layers
- Large models: 12+ layers

---

### `n_embd: int = 256`

**Description:** The embedding dimension (hidden size). This is the fundamental dimension `D` used throughout the model.

**Impact on Training:**
- **Memory Usage:** **VERY HIGH IMPACT** - Quadratic effect on most weight matrices
- **Training Speed:** Significant impact - larger dimensions = more computation
- **VRAM Usage:** Major contributor to memory consumption

**Impact on Final Model:**
- **Parameter Count:** **HIGHEST IMPACT** - Affects all major weight matrices:
  - Embedding table: `vocab_size × n_embd`
  - Encoder: `n_head × n_embd × N`
  - Decoder: `(n_head × N) × n_embd`
  - Encoder_v: `n_head × n_embd × N`
  - LM head: `n_embd × vocab_size`
- **Model Capacity:** Core determinant of model expressiveness
- **Inference Speed:** Quadratic impact on computation time
- **Model Quality:** Larger dimensions generally improve quality but with diminishing returns

**Typical Values:** 128-1024
- Tiny models: 128-192
- Small models: 256-384
- Medium models: 512-768
- Large models: 1024+

**Recommendation:** This is the primary knob for scaling model size. Adjust first when changing model scale.

---

### `dropout: float = 0.1`

**Description:** Dropout probability applied to activations during training for regularization.

**Impact on Training:**
- **Memory Usage:** Minimal - only affects dropout masks
- **Training Speed:** Minimal overhead
- **Regularization:** Prevents overfitting by randomly dropping activations
- **Training Stability:** Can improve generalization

**Impact on Final Model:**
- **Parameter Count:** None - dropout doesn't add parameters
- **Model Quality:** Improves generalization, reduces overfitting
- **Inference:** Disabled during inference (no effect)

**Typical Values:** 0.0-0.3
- No regularization: 0.0
- Light regularization: 0.05-0.1
- Medium regularization: 0.1-0.2
- Heavy regularization: 0.2-0.3

**Recommendation:** Use 0.1 as a starting point. Increase if overfitting occurs.

---

### `n_head: int = 4`

**Description:** Number of attention heads used in the model.

**Impact on Training:**
- **Memory Usage:** Moderate - affects how internal dimension N is divided
- **Training Speed:** Moderate - more heads = more parallel operations
- **Computation:** Affects the split of `mlp_internal_dim_multiplier`

**Impact on Final Model:**
- **Parameter Count:** **INDIRECT IMPACT** - Inversely affects N: `N = mlp_internal_dim_multiplier × n_embd // n_head`
  - More heads → smaller N per head (fewer params per head, but more heads)
- **Model Capacity:** Multi-head attention allows learning different attention patterns
- **Inference Speed:** Moderate impact
- **Model Quality:** Multiple heads capture different features/relationships

**Typical Values:** 4-16
- Small models: 4-8 heads
- Medium models: 8-12 heads
- Large models: 12-16+ heads

**Note:** Usually chosen such that `n_embd` is divisible by `n_head` for clean splits.

---

### `mlp_internal_dim_multiplier: int = 128`

**Description:** Multiplier that determines the internal dimension of the MLP/attention mechanism. The actual internal dimension is: `N = mlp_internal_dim_multiplier × n_embd // n_head`

**Impact on Training:**
- **Memory Usage:** **VERY HIGH IMPACT** - Directly affects the size of major weight matrices
- **Training Speed:** Significant - larger internal dimensions require more computation
- **VRAM Usage:** Major contributor to memory consumption

**Impact on Final Model:**
- **Parameter Count:** **VERY HIGH IMPACT** - Second most important after `n_embd`:
  - Encoder: `n_head × n_embd × N`
  - Decoder: `(n_head × N) × n_embd`
  - Encoder_v: `n_head × n_embd × N`
- **Model Capacity:** Higher values increase the model's ability to learn complex transformations
- **Inference Speed:** Significant impact on computation time
- **Model Quality:** Larger values generally improve capacity but with diminishing returns

**Typical Values:** 32-256
- Tiny models: 32-64
- Small models: 64-128
- Medium models: 128-192
- Large models: 192-256+

**Recommendation:** This is the second most important parameter for model scaling after `n_embd`.

---

### `vocab_size: int = 256`

**Description:** Size of the vocabulary. For byte-level models, this is fixed at 256 (all possible byte values 0-255).

**Impact on Training:**
- **Memory Usage:** Moderate - affects embedding and output layers
- **Training Speed:** Minimal direct impact
- **Data Handling:** 256 = byte-level (no tokenization needed)

**Impact on Final Model:**
- **Parameter Count:** **MODERATE IMPACT**:
  - Embedding table: `vocab_size × n_embd`
  - LM head: `n_embd × vocab_size`
  - Total: `2 × vocab_size × n_embd` parameters
- **Model Flexibility:** Byte-level (256) handles all text, any language
- **Inference Speed:** Minimal impact
- **Model Quality:** Byte-level is more flexible but may require longer sequences

**Typical Values:**
- Byte-level: 256 (fixed)
- Character-level: 50-500
- Subword/BPE: 8000-50000+

**Note:** This project uses byte-level encoding (256), so this should not be changed unless switching to a different tokenization scheme.

---

## Training Configuration (`train.py`)

These settings control the training process, hardware utilization, and optimization.

### `BLOCK_SIZE: int = 512`

**Description:** Maximum sequence length (context window) for training. The model sees `BLOCK_SIZE` tokens at a time.

**Impact on Training:**
- **Memory Usage:** **VERY HIGH IMPACT** - Quadratic effect due to attention mechanism
  - Attention matrix: `BLOCK_SIZE × BLOCK_SIZE` per head
- **Training Speed:** Quadratic impact - longer sequences are much slower
- **VRAM Usage:** One of the biggest contributors to memory usage
- **Context:** Determines how much context the model can see at once

**Impact on Final Model:**
- **Parameter Count:** None - doesn't affect model weights
- **Model Capability:** Longer sequences allow learning longer-range dependencies
- **Inference Speed:** Quadratic impact on generation speed
- **Model Quality:** Longer context generally improves quality for tasks requiring long-range understanding

**Typical Values:** 128-2048+
- Very small: 128-256
- Small: 256-512
- Medium: 512-1024
- Large: 1024-2048
- Very large: 2048-8192+

**Recommendation:** Most impactful for VRAM reduction. Reduce this first if memory is limited.

---

### `BATCH_SIZE: int = 32`

**Description:** Number of sequences processed in parallel during each training step.

**Impact on Training:**
- **Memory Usage:** **HIGHEST IMPACT** - Directly linear
  - Total memory = `BATCH_SIZE × BLOCK_SIZE × activations`
- **Training Speed:** Larger batches = better GPU utilization but more memory
- **Gradient Quality:** Larger batches = more stable gradients, less noise
- **Training Dynamics:** Affects convergence behavior and generalization
- **VRAM Usage:** Primary lever for memory management

**Impact on Final Model:**
- **Parameter Count:** None - purely a training hyperparameter
- **Model Quality:** Can affect final performance:
  - Too small: noisy gradients, may not converge well
  - Too large: may converge to sharper minima (worse generalization)
- **Training Efficiency:** Smaller batches may require more steps to converge

**Typical Values:** 8-128+
- Limited VRAM: 4-16
- Moderate VRAM: 16-64
- High VRAM: 64-128+

**Recommendation:** 
- **For VRAM reduction:** Reduce this first - most effective with no architectural changes
- Can compensate smaller batches with more training steps or gradient accumulation

---

### `MAX_ITERS: int = 3000`

**Description:** Total number of training iterations (optimization steps).

**Impact on Training:**
- **Training Time:** Linear - more iterations = longer training
- **Memory Usage:** None - doesn't affect memory per step
- **Convergence:** More iterations allow model to learn better (up to convergence)
- **Overfitting Risk:** Too many iterations may lead to overfitting on small datasets

**Impact on Final Model:**
- **Parameter Count:** None
- **Model Quality:** Needs sufficient iterations to converge
  - Too few: underfitting
  - Optimal: converged but not overfit
  - Too many: may overfit

**Typical Values:** 1000-100000+
- Quick test: 1000-5000
- Small dataset: 5000-20000
- Medium dataset: 20000-100000
- Large dataset: 100000+

**Recommendation:** Monitor loss curves. Stop when validation loss plateaus or starts increasing.

---

### `LEARNING_RATE: float = 1e-3`

**Description:** Step size for the optimizer. Controls how much to update weights based on gradients.

**Impact on Training:**
- **Convergence Speed:** Higher LR = faster initial progress but may be unstable
- **Stability:** Too high = divergence; too low = slow convergence
- **Final Performance:** Critical for reaching optimal performance
- **Memory Usage:** None

**Impact on Final Model:**
- **Parameter Count:** None
- **Model Quality:** **CRITICAL** - Wrong LR can prevent convergence or cause poor results
- **Optimization:** Affects what local optimum the model reaches

**Typical Values:** 1e-5 to 1e-2
- Conservative: 1e-4 to 3e-4
- Standard: 3e-4 to 1e-3
- Aggressive: 1e-3 to 1e-2

**Recommendation:** 
- Start with 1e-3 for small models
- Use 3e-4 for medium/large models
- Consider learning rate warmup and decay schedules for longer training

---

### `WEIGHT_DECAY: float = 0.1`

**Description:** L2 regularization penalty applied to model weights by AdamW optimizer.

**Impact on Training:**
- **Regularization:** Prevents weights from growing too large
- **Overfitting:** Helps prevent overfitting
- **Training Dynamics:** Can improve stability and generalization
- **Memory Usage:** None

**Impact on Final Model:**
- **Parameter Count:** None
- **Model Quality:** Generally improves generalization
- **Weight Magnitudes:** Keeps weights smaller, more regularized

**Typical Values:** 0.0-0.3
- No regularization: 0.0
- Light: 0.01-0.05
- Medium: 0.05-0.1
- Heavy: 0.1-0.3

**Recommendation:** 0.1 is a good default for AdamW. Reduce if the model struggles to fit the training data.

---

### `LOG_FREQ: int = 100`

**Description:** How often (in iterations) to log training metrics.

**Impact on Training:**
- **Monitoring:** More frequent logging = better visibility into training
- **Performance:** Very minimal overhead
- **Memory Usage:** None

**Impact on Final Model:**
- **Parameter Count:** None
- **Model Quality:** None - purely for monitoring

**Typical Values:** 10-1000
- Frequent monitoring: 10-100
- Standard: 100-500
- Infrequent: 500-1000

---

## System Configuration

These settings are automatically detected or configured based on hardware.

### `device: torch.device`

**Description:** Computation device (CUDA GPU, CPU, or MPS for Mac).

**Auto-detected:** `"cuda"` if available, else `"cpu"`

**Impact:**
- GPU (CUDA): 10-100× faster than CPU
- CPU: Much slower, but no VRAM limitations
- MPS (Mac): Good performance on Apple Silicon

---

### `dtype: str`

**Description:** Floating-point precision for training.

**Auto-detected:** `"bfloat16"` if supported, else `"float16"`, else `"float32"`

**Impact on Training:**
- **Memory Usage:**
  - `float32`: Baseline (full precision)
  - `bfloat16` / `float16`: ~50% memory reduction
- **Speed:** Mixed precision often 2-3× faster
- **Stability:** `bfloat16` more stable than `float16`

**Impact on Final Model:**
- **Quality:** Minimal difference with proper scaling
- **Numerical Stability:** `bfloat16` > `float16` in most cases

**Recommendation:** Use auto-detection. `bfloat16` is preferred when available.

---

### TF32 Settings

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Description:** Enables TensorFloat-32 for faster matrix operations on Ampere+ GPUs.

**Impact:**
- **Speed:** Up to 10× faster on supported hardware
- **Precision:** Slightly reduced but usually negligible
- **Memory:** None

**Recommendation:** Keep enabled for modern NVIDIA GPUs (A100, RTX 30/40 series).

---

## Quick Reference Tables

### Impact on VRAM Usage (Highest to Lowest)

| Setting | Impact Level | Scaling | Recommendation for 14GB VRAM |
|---------|-------------|---------|------------------------------|
| `BATCH_SIZE` | ⚠️ **CRITICAL** | Linear | Reduce to 16 or 8 |
| `BLOCK_SIZE` | ⚠️ **CRITICAL** | Quadratic | Reduce to 256-384 |
| `n_embd` | 🔴 **Very High** | Quadratic | Keep at 256 or reduce to 192 |
| `mlp_internal_dim_multiplier` | 🔴 **Very High** | Linear×n_embd | Reduce to 96 or 64 |
| `n_layer` | 🟡 **Medium** | Linear | Keep at 6 or reduce to 4-5 |
| `vocab_size` | 🟡 **Medium** | Linear | Keep at 256 (fixed) |
| `n_head` | 🟢 **Low** | Indirect | Keep at 4 |
| `dtype` | 🔵 **System** | ~50% | Use bfloat16/float16 |

### Impact on Parameter Count (Highest to Lowest)

| Setting | Impact Level | Formula Contribution |
|---------|-------------|---------------------|
| `n_embd` | ⚠️ **HIGHEST** | Appears in all weight matrices (quadratic) |
| `mlp_internal_dim_multiplier` | 🔴 **Very High** | Determines N (encoder, decoder, encoder_v sizes) |
| `vocab_size` | 🟡 **Medium** | Embedding + LM head: `2 × vocab × n_embd` |
| `n_head` | 🟡 **Medium** | Inverse effect on N: `N = mult × n_embd / n_head` |
| `n_layer` | 🟢 **None** | Weights are shared across layers |
| `dropout` | 🟢 **None** | No parameters |
| `BATCH_SIZE` | 🟢 **None** | Training-only |
| `BLOCK_SIZE` | 🟢 **None** | Training-only |

### Impact on Training Speed (Highest to Lowest)

| Setting | Impact Level | Notes |
|---------|-------------|-------|
| `BLOCK_SIZE` | ⚠️ **CRITICAL** | Quadratic due to attention |
| `BATCH_SIZE` | 🔴 **Very High** | More batches = better GPU util |
| `n_embd` | 🔴 **Very High** | Affects all operations |
| `mlp_internal_dim_multiplier` | 🔴 **Very High** | Large matrices = slow |
| `n_layer` | 🟡 **Medium** | Linear scaling |
| `MAX_ITERS` | 🟡 **Medium** | Total training time |
| `dtype` | 🟢 **Helpful** | Mixed precision 2-3× speedup |

### Impact on Model Quality

| Setting | Impact | Tuning Guidance |
|---------|--------|-----------------|
| `n_embd` | ⚠️ **Critical** | Primary capacity control - bigger usually better |
| `mlp_internal_dim_multiplier` | 🔴 **Very High** | Secondary capacity control |
| `n_layer` | 🔴 **Very High** | Depth helps with complex patterns |
| `LEARNING_RATE` | 🔴 **Very High** | Must be tuned correctly |
| `BLOCK_SIZE` | 🟡 **Medium** | Longer context helps some tasks |
| `BATCH_SIZE` | 🟡 **Medium** | Sweet spot exists (not too small/large) |
| `dropout` | 🟢 **Low-Medium** | Helps with overfitting |
| `WEIGHT_DECAY` | 🟢 **Low-Medium** | Regularization benefit |
| `n_head` | 🟢 **Low** | Usually 4-8 is fine |

---

## Example Configurations

### Minimal VRAM (8-10GB)
```python
# Model Config
BDHConfig(
    n_layer=4,
    n_embd=192,
    n_head=4,
    mlp_internal_dim_multiplier=64,
    dropout=0.1,
    vocab_size=256
)

# Training Config
BLOCK_SIZE = 256
BATCH_SIZE = 8
```

### Target 14GB VRAM
```python
# Model Config
BDHConfig(
    n_layer=6,
    n_embd=256,
    n_head=4,
    mlp_internal_dim_multiplier=96,
    dropout=0.1,
    vocab_size=256
)

# Training Config
BLOCK_SIZE = 384
BATCH_SIZE = 16
```

### High Performance (24GB+ VRAM)
```python
# Model Config
BDHConfig(
    n_layer=8,
    n_embd=512,
    n_head=8,
    mlp_internal_dim_multiplier=128,
    dropout=0.1,
    vocab_size=256
)

# Training Config
BLOCK_SIZE = 1024
BATCH_SIZE = 32
```

---

## Notes

- All settings interact with each other - test changes carefully
- For memory issues, prioritize: `BATCH_SIZE` → `BLOCK_SIZE` → `n_embd`
- For quality improvements, prioritize: `n_embd` → `n_layer` → `mlp_internal_dim_multiplier`
- Monitor training metrics and adjust accordingly
- Consider gradient accumulation to simulate larger batch sizes with limited VRAM

---

*Last updated: October 16, 2025*
