# Task 0005: Implement Final Model Export Functionality

## Priority
**High** - Essential for sharing trained models and enabling inference

## Purpose
Create a standardized format for saving and loading trained BDH models, similar to HuggingFace's `save_pretrained()` and `from_pretrained()` methods. This enables model distribution, sharing, and easy loading for inference without needing to know implementation details.

## Current State
- No standard way to save trained models
- Training script only saves final generated sample
- No config persistence
- No tokenizer saving
- No model metadata or documentation
- Users must manually reconstruct model from code and weights

## Expected Outcome
After implementing this task, the project should have:
1. `save_pretrained(directory)` method on BDH class
2. `from_pretrained(directory)` class method on BDH class
3. Saved models contain: weights, config, tokenizer (if used), and metadata
4. Support for safetensors format (safer than pickle)
5. Automatic generation of model cards with training info
6. Backward compatibility with PyTorch `.pt` format
7. Version tracking for model format

## Detailed Requirements

### 1. Directory Structure for Saved Models
```
model_name/
    config.json              # BDHConfig as JSON
    model.safetensors        # Model weights (preferred)
    model.pt                 # Model weights (fallback)
    tokenizer.json           # Tokenizer config (if used)
    tokenizer_config.json    # Tokenizer metadata
    generation_config.json   # Default generation parameters
    README.md                # Model card with info
    training_args.json       # Training hyperparameters
```

### 2. Config Serialization
Make BDHConfig serializable:
```python
@dataclasses.dataclass
class BDHConfig:
    # ... existing fields ...
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BDHConfig':
        """Load config from dictionary."""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save config to JSON file."""
        pass
    
    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'BDHConfig':
        """Load config from directory."""
        pass
```

### 3. Model Saving Method
Add to BDH class:
```python
def save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    save_tokenizer: bool = True,
    create_model_card: bool = True,
    training_args: dict = None,
):
    """
    Save model, config, and tokenizer to directory.
    
    Args:
        save_directory: Directory to save to
        safe_serialization: Use safetensors instead of pickle
        save_tokenizer: Save tokenizer if present
        create_model_card: Generate README.md model card
        training_args: Training hyperparameters to save
    """
    pass
```

### 4. Model Loading Method
Add to BDH class:
```python
@classmethod
def from_pretrained(
    cls,
    model_directory: str,
    device: str = "auto",
    torch_dtype: torch.dtype = None,
) -> 'BDH':
    """
    Load model, config, and tokenizer from directory.
    
    Args:
        model_directory: Directory containing saved model
        device: Device to load model on ("auto", "cuda", "cpu")
        torch_dtype: Override dtype (e.g., torch.float16)
    
    Returns:
        Loaded BDH model
    """
    pass
```

### 5. Model Card Generation
Auto-generate informative README:
```markdown
# BDH Model Card

## Model Description
Baby Dragon Hatchling (BDH) language model trained on [dataset].

## Model Architecture
- Layers: 6
- Embedding Dimension: 256
- Vocabulary Size: 32000
- Attention Heads: 4
- Parameters: ~15M

## Training Details
- Dataset: wikitext-103
- Training Steps: 50000
- Final Loss: 2.34
- Training Time: 8 hours
- Hardware: NVIDIA RTX 4090

## Usage
```python
from bdh import BDH

# Load model
model = BDH.from_pretrained("path/to/model")

# Generate text
text = model.generate_text("Once upon a time", max_new_tokens=100)
print(text)
```

## Citation
If you use this model, please cite:
[Citation info]
```

### 6. Generation Config
Save default generation parameters:
```json
{
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": true
}
```

## Implementation Steps

### Step 1: Add Dependencies
Update `requirements.txt`:
```
safetensors>=0.4.0
```

### Step 2: Make BDHConfig Serializable
Update `bdh.py`:
```python
import json
from pathlib import Path

@dataclasses.dataclass
class BDHConfig:
    # ... existing fields ...
    
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BDHConfig':
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'BDHConfig':
        config_path = Path(save_directory) / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
```

### Step 3: Implement save_pretrained()
Add to BDH class:
```python
def save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    save_tokenizer: bool = True,
    create_model_card: bool = True,
    training_args: dict = None,
):
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    self.config.save_pretrained(save_directory)
    
    # Save model weights
    if safe_serialization:
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), save_path / "model.safetensors")
        except ImportError:
            print("safetensors not available, falling back to .pt")
            torch.save(self.state_dict(), save_path / "model.pt")
    else:
        torch.save(self.state_dict(), save_path / "model.pt")
    
    # Save tokenizer
    if save_tokenizer and self.tokenizer is not None:
        self.tokenizer.save_pretrained(str(save_path))
    
    # Save generation config
    generation_config = {
        "max_new_tokens": 100,
        "temperature": 0.8,
        "top_k": 50,
    }
    with open(save_path / "generation_config.json", 'w') as f:
        json.dump(generation_config, f, indent=2)
    
    # Save training args
    if training_args is not None:
        with open(save_path / "training_args.json", 'w') as f:
            json.dump(training_args, f, indent=2)
    
    # Create model card
    if create_model_card:
        self._create_model_card(save_path, training_args)
```

### Step 4: Implement from_pretrained()
Add to BDH class:
```python
@classmethod
def from_pretrained(
    cls,
    model_directory: str,
    device: str = "auto",
    torch_dtype: torch.dtype = None,
) -> 'BDH':
    model_path = Path(model_directory)
    
    # Load config
    config = BDHConfig.from_pretrained(model_directory)
    
    # Load tokenizer if present
    tokenizer = None
    if (model_path / "tokenizer.json").exists():
        from tokenizer_utils import TokenizerManager
        tokenizer = TokenizerManager.load_tokenizer(str(model_path))
    
    # Create model
    model = cls(config, tokenizer=tokenizer)
    
    # Load weights
    if (model_path / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(model_path / "model.safetensors")
    elif (model_path / "model.pt").exists():
        state_dict = torch.load(model_path / "model.pt", map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_directory}")
    
    model.load_state_dict(state_dict)
    
    # Move to device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Set dtype
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    
    return model
```

### Step 5: Implement Model Card Generation
Add helper method:
```python
def _create_model_card(self, save_path: Path, training_args: dict = None):
    """Generate README.md model card."""
    
    # Count parameters
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    model_card = f"""# BDH Language Model

## Model Description
Baby Dragon Hatchling (BDH) language model.

## Model Architecture
- Layers: {self.config.n_layer}
- Embedding Dimension: {self.config.n_embd}
- Vocabulary Size: {self.config.vocab_size}
- Attention Heads: {self.config.n_head}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}

## Architecture Details
BDH uses a unique architecture with:
- RoPE-based positional encoding
- Non-standard attention mechanism (no softmax)
- Sparse activations with ReLU
- Element-wise multiplication of representations

"""
    
    if training_args:
        model_card += f"""## Training Details
- Training Steps: {training_args.get('max_iters', 'N/A')}
- Batch Size: {training_args.get('batch_size', 'N/A')}
- Learning Rate: {training_args.get('learning_rate', 'N/A')}
- Dataset: {training_args.get('dataset_name', 'N/A')}

"""
    
    model_card += """## Usage
```python
from bdh import BDH

# Load model
model = BDH.from_pretrained("path/to/model")

# Generate text
prompt = "Once upon a time"
text = model.generate_text(prompt, max_new_tokens=100, temperature=0.8)
print(text)
```

## Model Architecture
This model uses the Baby Dragon Hatchling (BDH) architecture, which differs significantly from standard transformers. See the paper/documentation for details.

## License
[Specify license]

## Citation
```bibtex
@misc{bdh,
  title={Baby Dragon Hatchling Language Model},
  author={Your Name},
  year={2025}
}
```
"""
    
    with open(save_path / "README.md", 'w') as f:
        f.write(model_card)
```

### Step 6: Update Training Script
Modify `train.py` to save model at end:
```python
# At end of training
print("Training complete, saving model...")
training_args = {
    'max_iters': MAX_ITERS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'dataset_name': args.dataset_name,
}

model.save_pretrained(
    save_directory="models/bdh-shakespeare",
    training_args=training_args,
)
print("Model saved to models/bdh-shakespeare")
```

### Step 7: Create Example Loading Script
Create `examples/load_model.py`:
```python
"""Example of loading and using a saved BDH model."""
from bdh import BDH

# Load model
model = BDH.from_pretrained("models/bdh-shakespeare")

# Generate text
prompt = "To be or not to be"
generated = model.generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)

print(generated)
```

## Testing Plan

1. **Test Config Serialization**
   - Save config to JSON
   - Load config from JSON
   - Verify all fields are preserved
   - Check data types are correct

2. **Test Model Saving**
   - Train small model
   - Save with `save_pretrained()`
   - Verify all files are created
   - Check file sizes are reasonable
   - Inspect JSON files for correctness

3. **Test Model Loading**
   - Load saved model with `from_pretrained()`
   - Verify config matches original
   - Check weights are identical
   - Verify generation produces same output (with fixed seed)

4. **Test Safetensors Format**
   - Save with `safe_serialization=True`
   - Verify safetensors file is created
   - Load and compare to .pt format
   - Check loading is faster than pickle

5. **Test Tokenizer Saving/Loading**
   - Save model with tokenizer
   - Load in new process
   - Verify tokenizer works correctly
   - Check encode/decode consistency

6. **Test Cross-Device Loading**
   - Save on GPU
   - Load on CPU
   - Verify no errors
   - Check device placement is correct

7. **Test Model Card Generation**
   - Create model card with various configs
   - Verify parameter counts
   - Check markdown formatting
   - Ensure info is accurate

## Code Example

Example usage after implementation:

```python
# After training
from bdh import BDH, BDHConfig
import torch

# Train model
config = BDHConfig(n_layer=6, n_embd=256, vocab_size=32000)
model = BDH(config)
# ... training code ...

# Save model
training_args = {
    'max_iters': 50000,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'dataset_name': 'wikitext-103',
}
model.save_pretrained("models/bdh-wikitext", training_args=training_args)

# Later: Load model
model = BDH.from_pretrained("models/bdh-wikitext")

# Generate text
text = model.generate_text("The quick brown", max_new_tokens=50)
print(text)

# Load on specific device
model = BDH.from_pretrained("models/bdh-wikitext", device="cuda", torch_dtype=torch.float16)
```

## Copilot Implementation Prompt

```
Implement save_pretrained() and from_pretrained() methods for the BDH model:

1. Update BDHConfig in bdh.py to add:
   - to_dict() method that returns dataclasses.asdict(self)
   - from_dict(config_dict) classmethod that returns cls(**config_dict)
   - save_pretrained(save_directory) that saves config as JSON
   - from_pretrained(save_directory) classmethod that loads config from JSON

2. Add save_pretrained() method to BDH class:
   - Parameters: save_directory, safe_serialization=True, save_tokenizer=True, create_model_card=True, training_args=None
   - Create directory if it doesn't exist
   - Save config.json using config.save_pretrained()
   - Save model weights as model.safetensors (if safe_serialization=True) or model.pt
   - Save tokenizer if present using tokenizer.save_pretrained()
   - Save generation_config.json with default parameters: max_new_tokens=100, temperature=0.8, top_k=50
   - Save training_args.json if training_args provided
   - Create README.md model card if create_model_card=True

3. Add from_pretrained() classmethod to BDH class:
   - Parameters: model_directory, device="auto", torch_dtype=None
   - Load config using BDHConfig.from_pretrained()
   - Load tokenizer if tokenizer.json exists
   - Create model instance: model = cls(config, tokenizer=tokenizer)
   - Load weights from model.safetensors (preferred) or model.pt
   - Use safetensors.torch.load_file() for .safetensors, torch.load() for .pt
   - Move model to device ("auto" means cuda if available, else cpu)
   - Convert to torch_dtype if specified
   - Return model

4. Add _create_model_card() helper method:
   - Count total and trainable parameters
   - Generate README.md with sections: Description, Architecture, Training Details, Usage, Citation
   - Include model hyperparameters from config
   - Include training info if training_args provided
   - Use markdown formatting

5. Update train.py:
   - At end of training (after line 126), save model with save_pretrained()
   - Create training_args dict with: max_iters, batch_size, learning_rate, dataset_name
   - Save to "models/bdh-shakespeare" or configurable path

6. Handle edge cases:
   - Directory already exists (overwrite warning)
   - Safetensors import failure (fallback to .pt)
   - Missing model files (clear error message)
   - Incompatible config versions (warning)

7. Use these imports:
   - from safetensors.torch import save_file, load_file (try/except for ImportError)
   - from pathlib import Path
   - import json

The BDH class is defined in bdh.py lines 75-172. Current generate() method is lines 158-172. Build on this for generate_text() wrapper if needed.

Ensure saved models can be loaded in a fresh Python process without needing access to training code.
```

## Files to Modify/Create
- **Modify**: `bdh.py` - Add save/load methods to BDH and BDHConfig
- **Modify**: `train.py` - Save model at end of training
- **Create**: `examples/load_model.py` - Example of loading saved model
- **Modify**: `requirements.txt` - Add safetensors dependency

## Dependencies
- `safetensors>=0.4.0` (safer serialization format)
- Task 0002 (Tokenizer Support) for saving tokenizers
- Task 0003 (Checkpoints) can use this for final model saving

## Success Criteria
- [ ] Can save trained model with `save_pretrained()`
- [ ] Can load model with `from_pretrained()` in new process
- [ ] Config is correctly serialized and deserialized
- [ ] Tokenizer is saved and loaded correctly
- [ ] Model card is generated with accurate info
- [ ] Safetensors format works and is faster than pickle
- [ ] Loading on different device (CPU/GPU) works
- [ ] Generated directory structure matches specification
- [ ] Model can be shared and loaded by others

## Related Tasks
- **Task 0002**: Tokenizer Support (tokenizer saving integration)
- **Task 0003**: Checkpoint Management (uses similar save/load)
- **Task 0006**: Inference Script (will use from_pretrained)
- **Task 0007**: API Server (will use from_pretrained)

## Notes
- Safetensors is preferred over pickle for security and performance
- Model cards improve model discoverability and documentation
- This format is compatible with HuggingFace conventions
- Consider adding model version tracking for future updates
- Large models may need sharded weight files (future enhancement)
