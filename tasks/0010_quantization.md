# Task 0010: Add Model Quantization Support

## Priority
**Medium** - Important for deploying models on consumer hardware

## Purpose
Implement post-training quantization (PTQ) and quantization-aware training (QAT) to reduce model size and improve inference speed. This enables running larger BDH models on resource-constrained devices.

## Current State
- Models use full precision (float32/float16/bfloat16)
- No quantization support
- Large memory footprint for inference
- Slower inference on CPU

## Expected Outcome
1. INT8 post-training quantization
2. INT4 quantization for extreme compression
3. bitsandbytes integration for 8-bit/4-bit loading
4. Quantization-aware training (optional)
5. Benchmarking tools for quantized models
6. Minimal accuracy degradation

## Implementation Steps

### Step 1: Add PyTorch Native Quantization

Create `quantization.py`:
```python
"""Model quantization utilities for BDH."""
import torch
import torch.quantization as quant


def quantize_model_int8(model, calibration_data=None):
    """Post-training dynamic quantization to INT8."""
    model.eval()
    
    # Dynamic quantization (weights + activations)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Embedding},
        dtype=torch.qint8
    )
    
    return quantized_model


def quantize_model_static(model, calibration_loader):
    """Static quantization with calibration."""
    model.eval()
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    with torch.no_grad():
        for batch in calibration_loader:
            model(batch)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model
```

### Step 2: Add bitsandbytes Support

```python
def load_model_8bit(model_path, device='cuda'):
    """Load model with 8-bit quantization using bitsandbytes."""
    import bitsandbytes as bnb
    from bdh import BDH
    
    # Load model
    model = BDH.from_pretrained(model_path, device='cpu')
    
    # Quantize linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Replace with 8-bit linear
            quantized = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
            )
            # Copy weights
            quantized.weight.data = module.weight.data
            if module.bias is not None:
                quantized.bias.data = module.bias.data
            
            # Replace module
            parent = model
            atoms = name.split('.')
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], quantized)
    
    return model.to(device)


def load_model_4bit(model_path, device='cuda'):
    """Load model with 4-bit quantization."""
    import bitsandbytes as bnb
    from bdh import BDH
    
    # Similar to 8-bit but with NF4 quantization
    # ... implementation
```

### Step 3: Create Quantization Script

Create `quantize_model.py`:
```python
#!/usr/bin/env python3
"""Quantize a trained BDH model."""
import argparse
import torch
from bdh import BDH
from quantization import quantize_model_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--method', default='int8', choices=['int8', '8bit', '4bit'])
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = BDH.from_pretrained(args.model)
    
    # Quantize
    print(f"Quantizing with {args.method}...")
    if args.method == 'int8':
        quantized = quantize_model_int8(model)
    elif args.method == '8bit':
        from quantization import load_model_8bit
        quantized = load_model_8bit(args.model)
    else:
        from quantization import load_model_4bit
        quantized = load_model_4bit(args.model)
    
    # Save
    print(f"Saving to {args.output}...")
    quantized.save_pretrained(args.output)
    
    # Benchmark
    print("Benchmarking...")
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    quantized_size = sum(p.numel() * p.element_size() for p in quantized.parameters()) / 1024 / 1024
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")


if __name__ == '__main__':
    main()
```

### Step 4: Add Benchmarking

Create `benchmark_quantized.py`:
```python
"""Benchmark quantized vs original model."""
import torch
import time
from bdh import BDH


def benchmark_inference(model, prompt, num_runs=10):
    """Benchmark generation speed."""
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        _ = model.generate_text(prompt, max_new_tokens=100)
        times.append(time.time() - start)
    
    return {
        'mean': sum(times) / len(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def main():
    # Load models
    original = BDH.from_pretrained("models/original")
    quantized = BDH.from_pretrained("models/quantized")
    
    prompt = "Once upon a time"
    
    # Benchmark
    print("Original model:")
    orig_stats = benchmark_inference(original, prompt)
    print(f"  Time: {orig_stats['mean']:.3f}±{orig_stats['std']:.3f}s")
    
    print("Quantized model:")
    quant_stats = benchmark_inference(quantized, prompt)
    print(f"  Time: {quant_stats['mean']:.3f}±{quant_stats['std']:.3f}s")
    print(f"  Speedup: {orig_stats['mean'] / quant_stats['mean']:.2f}x")
```

### Step 5: Update serve.py for Quantized Models

```python
# In serve.py
parser.add_argument('--quantization', choices=['none', 'int8', '8bit', '4bit'], default='none')

# Load model
if config.get('quantization') == 'int8':
    from quantization import quantize_model_int8
    model = BDH.from_pretrained(model_path)
    model = quantize_model_int8(model)
elif config.get('quantization') == '8bit':
    from quantization import load_model_8bit
    model = load_model_8bit(model_path)
else:
    model = BDH.from_pretrained(model_path)
```

## Testing Plan

1. Quantize small model to INT8
2. Test generation quality
3. Benchmark speed improvement
4. Test 8-bit bitsandbytes loading
5. Verify memory reduction
6. Test API server with quantized model

## Copilot Implementation Prompt

```
Implement quantization support for BDH models:

1. Create quantization.py with functions:
   - quantize_model_int8(model): Dynamic quantization using torch.quantization.quantize_dynamic()
   - load_model_8bit(model_path): Load with bitsandbytes 8-bit quantization
   - load_model_4bit(model_path): Load with bitsandbytes 4-bit quantization

2. For torch quantization:
   - Use torch.quantization.quantize_dynamic()
   - Quantize torch.nn.Linear and torch.nn.Embedding layers
   - Use torch.qint8 dtype
   - Model should remain in eval mode

3. For bitsandbytes:
   - Import bitsandbytes as bnb
   - Replace Linear layers with bnb.nn.Linear8bitLt
   - Copy weights from original to quantized layers
   - Requires CUDA device

4. Create quantize_model.py script:
   - Load model with BDH.from_pretrained()
   - Quantize with selected method
   - Save with save_pretrained()
   - Print size comparison

5. Create benchmark_quantized.py:
   - Load original and quantized models
   - Time generation for both
   - Print speedup and memory usage

6. Update serve.py:
   - Add --quantization argument
   - Load quantized model if specified
   - Support int8, 8bit, 4bit options

Test with small model first to verify quality is acceptable.
```

## Files to Create
- **Create**: `quantization.py`
- **Create**: `quantize_model.py`
- **Create**: `benchmark_quantized.py`
- **Modify**: `serve.py`
- **Modify**: `requirements.txt` (add bitsandbytes)

## Dependencies
- `bitsandbytes>=0.41.0` (for 8-bit/4-bit)

## Success Criteria
- [ ] INT8 quantization works
- [ ] bitsandbytes 8-bit loading works
- [ ] Model size reduced 2-4x
- [ ] Inference speed improved
- [ ] Quality degradation < 5%
- [ ] API server supports quantized models

## Notes
- INT8 dynamic quantization is easiest
- bitsandbytes requires CUDA
- 4-bit may have quality issues
- Test quality on validation set
- Quantization works best for larger models
