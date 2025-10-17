# Task 0002: Implement Tokenizer Support

## Priority
**High** - Essential for scaling beyond toy examples and achieving good performance on real-world text

## Purpose
Add support for modern tokenizers (BPE, WordPiece, SentencePiece) to replace the current byte-level encoding. This enables the model to work with larger vocabularies (32k-50k tokens) which is more efficient for natural language and leads to better compression and performance.

## Current State
- Model uses byte-level encoding with `vocab_size=256`
- Input text is converted directly to UTF-8 bytes
- No tokenizer training or management
- Embedding layer is hardcoded to 256 vocabulary size
- No way to save/load tokenizer with model

## Expected Outcome
After implementing this task, the project should have:
1. Support for training and using tokenizers from HuggingFace tokenizers library
2. Extended `BDHConfig` with tokenizer configuration
3. Utilities to train tokenizers on datasets
4. Methods to save/load tokenizers with trained models
5. Updated embedding layers to handle variable vocabulary sizes
6. Backward compatibility with byte-level encoding

## Detailed Requirements

### 1. Extend BDHConfig
Add tokenizer-related fields to the configuration:
```python
@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256  # Will be updated based on tokenizer
    
    # New tokenizer fields
    tokenizer_type: str = "byte"  # "byte", "bpe", "wordpiece", "unigram"
    tokenizer_vocab_size: int = 256  # Target vocab size for training
```

### 2. Create Tokenizer Management Module
Add tokenizer utilities to `data.py` or create `tokenizer_utils.py`:

```python
class TokenizerManager:
    """Manages tokenizer training, loading, and encoding."""
    
    def __init__(self, tokenizer_type: str = "byte", vocab_size: int = 256):
        pass
    
    def train_tokenizer(self, dataset, output_path: str):
        """Train a tokenizer on a dataset and save it."""
        pass
    
    def load_tokenizer(self, path: str):
        """Load a pre-trained tokenizer."""
        pass
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        pass
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    def save_tokenizer(self, path: str):
        """Save tokenizer to disk."""
        pass
```

### 3. Update BDH Model Class
Modify `bdh.py` to:
- Accept variable vocabulary sizes in embeddings
- Store tokenizer reference or path in the model
- Provide methods to encode/decode text using the model's tokenizer

```python
class BDH(nn.Module):
    def __init__(self, config: BDHConfig, tokenizer=None):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        self.tokenizer = tokenizer  # Optional tokenizer instance
        
        # ... rest of initialization using config.vocab_size
```

### 4. Update Training Script
Modify `train.py` to:
- Support tokenizer training before model training
- Use tokenizer for encoding data batches
- Save tokenizer alongside model checkpoints
- Add command-line flags for tokenizer configuration

### 5. Update Generation Method
Update the `generate()` method to:
- Accept text prompts (not just token IDs)
- Use tokenizer for encoding prompts
- Use tokenizer for decoding generated tokens
- Handle special tokens (BOS, EOS, PAD)

## Implementation Steps

### Step 1: Add Dependencies
Update `requirements.txt`:
```
tokenizers>=0.15.0
```

### Step 2: Create TokenizerManager Class
Implement the tokenizer management utilities:
- Support for ByteLevel (current behavior)
- BPE tokenizer (most common, used by GPT-2/GPT-3)
- WordPiece tokenizer (used by BERT)
- Unigram tokenizer (used by T5)

### Step 3: Implement Tokenizer Training
- Take text dataset as input
- Train tokenizer to target vocabulary size
- Add special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`
- Save tokenizer in HuggingFace format

### Step 4: Update BDHConfig
Add tokenizer fields to dataclass and ensure they're serializable:
```python
tokenizer_type: str = "byte"
tokenizer_vocab_size: int = 256
tokenizer_path: Optional[str] = None  # Path to saved tokenizer
```

### Step 5: Update DatasetLoader
Modify `data.py` (from Task 0001):
- Accept tokenizer instance in constructor
- Use tokenizer.encode() instead of byte conversion when tokenizer is provided
- Ensure batches have correct token IDs
- Handle padding if sequences are different lengths

### Step 6: Update BDH Model
- Remove hardcoded vocab_size=256 assumption
- Update embedding layer to use config.vocab_size
- Update lm_head to output correct number of logits
- Store tokenizer metadata

### Step 7: Add Save/Load for Tokenizers
Create methods to save/load tokenizer with model:
```python
def save_pretrained(self, save_directory: str):
    """Save model and tokenizer."""
    # Save model weights
    # Save config
    # Save tokenizer if present
    
def from_pretrained(cls, model_directory: str):
    """Load model and tokenizer."""
    # Load config
    # Load tokenizer
    # Load model weights
```

### Step 8: Update Generation Interface
Make generation user-friendly:
```python
def generate_text(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
    """Generate text from a string prompt."""
    token_ids = self.tokenizer.encode(prompt)
    generated_ids = self.generate(token_ids, max_new_tokens, **kwargs)
    return self.tokenizer.decode(generated_ids)
```

### Step 9: Add Training Command for Tokenizer
Add script or command to train tokenizer:
```bash
python train_tokenizer.py \
    --dataset_name openwebtext \
    --tokenizer_type bpe \
    --vocab_size 32000 \
    --output_path tokenizers/bpe-32k
```

## Testing Plan

1. **Test Byte-Level Tokenizer** (backward compatibility)
   - Should work exactly as before
   - vocab_size=256
   - No external tokenizer file needed

2. **Test BPE Tokenizer Training**
   - Train on small dataset (WikiText-2)
   - Verify vocab_size matches target
   - Check special tokens are present
   - Verify encode/decode roundtrip

3. **Test Model Training with BPE**
   - Train small model with BPE tokenizer
   - Verify loss converges
   - Check generation produces valid text
   - Compare to byte-level baseline

4. **Test Save/Load**
   - Save model with tokenizer
   - Load in new process
   - Verify generation works identically
   - Check tokenizer files are present

5. **Test Edge Cases**
   - Unknown tokens (OOV handling)
   - Empty strings
   - Very long sequences
   - Special characters and Unicode

## Code Example

Example usage after implementation:

```python
# Train a BPE tokenizer
from tokenizer_utils import TokenizerManager
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tok_manager = TokenizerManager(tokenizer_type="bpe", vocab_size=32000)
tok_manager.train_tokenizer(dataset["text"], "tokenizers/bpe-32k")

# Train model with tokenizer
config = BDHConfig(vocab_size=32000, tokenizer_type="bpe")
tokenizer = tok_manager.load_tokenizer("tokenizers/bpe-32k")
model = BDH(config, tokenizer=tokenizer)

# Generate text
prompt = "Once upon a time"
generated = model.generate_text(prompt, max_new_tokens=100, temperature=0.8)
print(generated)

# Save model with tokenizer
model.save_pretrained("models/bdh-bpe-32k")

# Load model with tokenizer
model = BDH.from_pretrained("models/bdh-bpe-32k")
```

## Copilot Implementation Prompt

```
Implement tokenizer support for the BDH model:

1. Create a TokenizerManager class (in tokenizer_utils.py or data.py) that:
   - Supports byte-level, BPE, WordPiece, and Unigram tokenizers
   - Uses the HuggingFace tokenizers library
   - Has methods to train_tokenizer(dataset, output_path), load_tokenizer(path), encode(text), decode(token_ids), and save_tokenizer(path)
   - Handles special tokens: <PAD>, <BOS>, <EOS>, <UNK>
   - For byte-level mode, maintains current behavior (UTF-8 bytes, vocab_size=256)

2. Update BDHConfig in bdh.py to add:
   - tokenizer_type: str = "byte"
   - tokenizer_vocab_size: int = 256
   - tokenizer_path: Optional[str] = None
   - Make vocab_size dynamically set based on tokenizer

3. Update BDH class in bdh.py to:
   - Accept optional tokenizer parameter in __init__
   - Store tokenizer reference
   - Update embed and lm_head to use config.vocab_size (not hardcoded 256)
   - Add generate_text(prompt: str) method that encodes prompt, calls generate(), and decodes output
   - Add save_pretrained(directory) that saves model weights, config, and tokenizer
   - Add @classmethod from_pretrained(directory) that loads everything

4. Update DatasetLoader in data.py to:
   - Accept optional tokenizer parameter
   - Use tokenizer.encode() when tokenizer is provided
   - Fall back to byte-level encoding when tokenizer is None
   - Handle token ID conversion properly

5. Create a train_tokenizer.py script that:
   - Loads a dataset from HuggingFace
   - Trains a tokenizer with specified vocab_size and type
   - Saves tokenizer to specified path
   - Accepts command-line arguments

6. Update train.py to:
   - Accept --tokenizer_type, --tokenizer_vocab_size, --tokenizer_path arguments
   - Load or train tokenizer before creating model
   - Pass tokenizer to DatasetLoader and BDH
   - Save tokenizer with checkpoints

7. Ensure backward compatibility:
   - Default tokenizer_type="byte" works exactly as before
   - No breaking changes to existing API
   - Byte-level mode doesn't require external files

Example tokenizer training should look like:
```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# For BPE
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"])
```

Reference the current BDH architecture in bdh.py (vocab_size in embed and lm_head) and generation code (lines 158-172) to ensure compatibility.
```

## Files to Modify/Create
- **Create**: `tokenizer_utils.py` - Tokenizer management utilities
- **Modify**: `bdh.py` - Update config, model class, and generation
- **Modify**: `data.py` - Support tokenizer in data loading
- **Modify**: `train.py` - Add tokenizer training/loading
- **Create**: `train_tokenizer.py` - Standalone tokenizer training script
- **Modify**: `requirements.txt` - Add tokenizers library

## Dependencies
- `tokenizers>=0.15.0` (HuggingFace tokenizers library)
- Task 0001 (HuggingFace Datasets) should be completed first for dataset integration

## Success Criteria
- [ ] Can train BPE tokenizer on WikiText-2
- [ ] Can train model with BPE tokenizer (vocab_size=32000)
- [ ] Generation works with text prompts (not just token IDs)
- [ ] Save/load preserves tokenizer
- [ ] Byte-level mode still works (backward compatible)
- [ ] Encode/decode roundtrip is correct
- [ ] Model trains successfully with larger vocabulary
- [ ] Generated text quality is improved over byte-level

## Related Tasks
- **Task 0001**: HuggingFace Datasets (prerequisite)
- **Task 0005**: Model Export (will use tokenizer saving)
- **Task 0006**: Inference Script (will use tokenizer for text generation)

## Notes
- Larger vocabulary (32k-50k) will make embeddings and lm_head much larger
- This increases model size but improves performance
- BPE is recommended as it's most common and well-tested
- Consider starting with smaller vocab_size (8k-16k) for experiments
