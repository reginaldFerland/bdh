# Task 0012: Create Example Notebooks and Tutorials

## Priority
**Medium** - Helps users understand and adopt the architecture

## Purpose
Develop Jupyter notebooks demonstrating key BDH features through interactive, hands-on tutorials. Notebooks provide an accessible way to learn the architecture, experiment with models, and understand the unique properties of BDH.

## Current State
- No example notebooks
- No interactive tutorials
- No visualization examples
- No comparison benchmarks
- Difficult for new users to get started

## Expected Outcome
1. Jupyter notebooks in `notebooks/` directory
2. Fine-tuning tutorial
3. Inference and sampling tutorial
4. Interpretability analysis notebook
5. Architecture comparison notebook
6. Ready-to-run examples with outputs

## Notebook Structure

```
notebooks/
├── 01_quickstart.ipynb              # Quick start guide
├── 02_training_shakespeare.ipynb    # Train on Shakespeare
├── 03_fine_tuning.ipynb             # Fine-tune pretrained model
├── 04_tokenizer_training.ipynb      # Train custom tokenizer
├── 05_generation_sampling.ipynb     # Sampling strategies
├── 06_interpretability.ipynb        # Analyze attention/activations
├── 07_architecture_comparison.ipynb # BDH vs Transformer
├── 08_quantization.ipynb            # Model quantization
└── 09_deployment.ipynb              # Deploy with API server
```

## Implementation Steps

### Step 1: Create Quickstart Notebook

Create `notebooks/01_quickstart.ipynb`:
```python
# Cell 1
"""
# BDH Quickstart Tutorial

This notebook shows how to:
1. Load a pretrained BDH model
2. Generate text
3. Experiment with sampling parameters
"""

# Cell 2
!pip install -q torch transformers

# Cell 3
from bdh import BDH
import torch

# Load pretrained model
print("Loading model...")
model = BDH.from_pretrained("models/bdh-shakespeare")
print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Cell 4
# Simple generation
prompt = "To be or not to be"
generated = model.generate_text(prompt, max_new_tokens=50, temperature=0.8)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")

# Cell 5
# Try different temperatures
for temp in [0.5, 0.8, 1.0, 1.5]:
    print(f"\nTemperature: {temp}")
    text = model.generate_text(
        "Once upon a time",
        max_new_tokens=50,
        temperature=temp
    )
    print(text)

# Cell 6
# Try different sampling strategies
strategies = {
    "Temperature only": {"temperature": 0.8},
    "Top-k": {"temperature": 0.8, "top_k": 40},
    "Top-p": {"temperature": 0.8, "top_p": 0.95},
    "Combined": {"temperature": 0.8, "top_k": 50, "top_p": 0.95},
}

prompt = "The meaning of life is"
for name, params in strategies.items():
    print(f"\n{name}:")
    text = model.generate_text(prompt, max_new_tokens=30, **params)
    print(text)
```

### Step 2: Create Training Tutorial

Create `notebooks/02_training_shakespeare.ipynb`:
```python
# Training from scratch tutorial
# - Load Shakespeare dataset
# - Configure small model
# - Train for a few iterations
# - Visualize loss curve
# - Generate samples during training
# - Save trained model
```

### Step 3: Create Fine-Tuning Tutorial

Create `notebooks/03_fine_tuning.ipynb`:
```python
# Fine-tuning tutorial
# - Load pretrained model
# - Prepare custom dataset
# - Fine-tune on new data
# - Compare before/after generation
# - Evaluate on validation set
# - Save fine-tuned model
```

### Step 4: Create Tokenizer Tutorial

Create `notebooks/04_tokenizer_training.ipynb`:
```python
# Tokenizer training tutorial
# - Load dataset for tokenizer training
# - Train BPE tokenizer
# - Visualize vocabulary
# - Compare tokenization with byte-level
# - Train model with custom tokenizer
# - Evaluate compression ratio
```

### Step 5: Create Sampling Tutorial

Create `notebooks/05_generation_sampling.ipynb`:
```python
# Comprehensive sampling tutorial
# - Explain temperature, top-k, top-p
# - Visualize probability distributions
# - Interactive parameter tuning
# - Compare sampling strategies
# - Repetition penalty demonstration
# - Best practices for different use cases
```

### Step 6: Create Interpretability Notebook

Create `notebooks/06_interpretability.ipynb`:
```python
# Interpretability analysis
# - Visualize attention patterns
# - Analyze sparse activations
# - Plot activation sparsity over layers
# - Examine learned features
# - Compare to transformer attention
# - Identify interpretable patterns
```

Example cells:
```python
# Cell: Analyze sparse activations
import matplotlib.pyplot as plt
import numpy as np

def analyze_sparsity(model, text):
    """Analyze activation sparsity in BDH model."""
    # Get activations
    with torch.no_grad():
        inputs = encode(text)
        activations = []
        
        # Hook to capture activations
        def hook(module, input, output):
            activations.append(output.detach())
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if 'encoder' in name:
                handles.append(module.register_forward_hook(hook))
        
        # Forward pass
        model(inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    # Calculate sparsity (percentage of zeros after ReLU)
    sparsities = []
    for act in activations:
        sparsity = (act == 0).float().mean().item()
        sparsities.append(sparsity * 100)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sparsities)), sparsities)
    plt.xlabel('Layer')
    plt.ylabel('Sparsity (%)')
    plt.title('Activation Sparsity Across Layers')
    plt.show()
    
    return sparsities

# Analyze on sample text
text = "To be or not to be, that is the question"
sparsities = analyze_sparsity(model, text)
print(f"Average sparsity: {np.mean(sparsities):.1f}%")
```

### Step 7: Create Comparison Notebook

Create `notebooks/07_architecture_comparison.ipynb`:
```python
# Architecture comparison
# - Train BDH and GPT-2 on same data
# - Compare training curves
# - Benchmark inference speed
# - Compare model sizes
# - Evaluate perplexity
# - Compare interpretability
# - Discuss tradeoffs
```

### Step 8: Create Quantization Notebook

Create `notebooks/08_quantization.ipynb`:
```python
# Quantization tutorial
# - Load pretrained model
# - Apply INT8 quantization
# - Apply 8-bit bitsandbytes
# - Compare sizes
# - Benchmark speed
# - Evaluate quality degradation
# - Deploy quantized model
```

### Step 9: Create Deployment Notebook

Create `notebooks/09_deployment.ipynb`:
```python
# Deployment tutorial
# - Save trained model
# - Start API server programmatically
# - Make API requests
# - Build client application
# - Test streaming responses
# - Deploy with Docker
# - Monitor performance
```

## Additional Features

### Add Visualizations

```python
# Attention visualization helper
def visualize_attention(model, text):
    """Visualize BDH attention patterns."""
    import seaborn as sns
    
    # Get attention weights
    with torch.no_grad():
        inputs = encode(text)
        # Extract attention from forward pass
        attn_weights = model.get_attention_weights(inputs)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn_weights[0].cpu().numpy(), cmap='viridis')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('BDH Attention Pattern')
    plt.show()
```

### Add Interactive Widgets

```python
# Interactive generation with ipywidgets
from ipywidgets import interact, FloatSlider, IntSlider

@interact(
    temperature=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.8),
    top_k=IntSlider(min=1, max=100, step=1, value=50),
    max_tokens=IntSlider(min=10, max=200, step=10, value=100)
)
def interactive_generation(temperature, top_k, max_tokens):
    text = model.generate_text(
        "Once upon a time",
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    print(text)
```

## Testing Plan

1. Run all notebooks end-to-end
2. Verify outputs are correct
3. Check visualizations display properly
4. Test on different environments
5. Ensure notebooks are beginner-friendly
6. Get feedback from test users

## Copilot Implementation Prompt

```
Create Jupyter notebooks for BDH tutorials:

1. Create notebooks/ directory with 9 tutorial notebooks

2. Notebook 01_quickstart.ipynb:
   - Load BDH model with from_pretrained()
   - Generate text with generate_text()
   - Demonstrate temperature, top_k, top_p
   - Show examples with different prompts
   - Include clear markdown explanations

3. Notebook 02_training_shakespeare.ipynb:
   - Import train.py functions
   - Create small BDHConfig
   - Load Shakespeare data
   - Train for 1000 steps
   - Plot loss curve
   - Generate samples
   - Save model

4. Notebook 03_fine_tuning.ipynb:
   - Load pretrained model
   - Prepare custom text dataset
   - Fine-tune with lower learning rate
   - Compare generations before/after
   - Show training loss

5. Notebook 06_interpretability.ipynb:
   - Define analyze_sparsity() function to measure activation sparsity
   - Visualize attention patterns with matplotlib/seaborn
   - Plot sparsity across layers
   - Compare different inputs
   - Explain what makes BDH interpretable

6. Notebook 08_quantization.ipynb:
   - Load original model
   - Quantize to INT8 with torch.quantization
   - Compare model sizes
   - Benchmark inference speed
   - Generate samples from both
   - Show speed/quality tradeoff

7. For each notebook:
   - Start with overview markdown cell
   - Add explanatory markdown between code cells
   - Include visualizations with matplotlib/seaborn
   - Show expected outputs
   - End with summary and next steps

8. Use clear variable names and comments
9. Make notebooks runnable top-to-bottom
10. Include !pip install commands for dependencies

Base notebooks on actual BDH implementation in bdh.py and train.py.
```

## Files to Create
- **Create**: 9 Jupyter notebooks in `notebooks/`
- **Create**: `notebooks/README.md` with notebook descriptions
- **Create**: `notebooks/requirements.txt` for notebook dependencies

## Dependencies
- `jupyter>=1.0.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `ipywidgets>=8.0.0` (for interactive widgets)

## Success Criteria
- [ ] All notebooks run without errors
- [ ] Outputs are saved in notebooks
- [ ] Visualizations are clear and informative
- [ ] Explanations are beginner-friendly
- [ ] Notebooks demonstrate key features
- [ ] Interactive elements work
- [ ] Notebooks are well-documented

## Related Tasks
- **Task 0011**: Documentation (notebooks complement written docs)
- All other tasks (notebooks demonstrate features)

## Notes
- Save notebook outputs for GitHub preview
- Use Google Colab links for easy access
- Keep notebooks focused and concise
- Include links to relevant documentation
- Consider creating video walkthroughs
- Make notebooks reproducible with fixed seeds
