# Task 0001: Add HuggingFace Datasets Integration

## Priority
**High** - This is foundational for training on diverse datasets beyond Shakespeare

## Purpose
Replace the hardcoded Shakespeare dataset loading with a flexible data loading system that can work with any HuggingFace dataset. This enables training BDH models on WikiText, OpenWebText, C4, and other large text corpora.

## Current State
- Training only works with `input.txt` (Shakespeare dataset)
- Data loading in `train.py` uses `np.memmap` directly on a single text file
- Only supports byte-level (ASCII) encoding (vocab_size=256)
- No support for streaming large datasets

## Expected Outcome
After implementing this task, the project should have:
1. A new `data.py` module with dataset loading utilities
2. Support for loading any text dataset from HuggingFace hub
3. Streaming support for datasets that don't fit in memory
4. Flexible preprocessing pipeline
5. Command-line arguments to specify which dataset to use
6. Backward compatibility with the existing byte-level Shakespeare training

## Detailed Requirements

### 1. Create `data.py` Module
Create a new file `data.py` with the following components:

#### Dataset Loader Class
```python
class DatasetLoader:
    """Handles loading and preprocessing datasets from various sources."""
    
    def __init__(self, 
                 dataset_name: str,
                 streaming: bool = False,
                 tokenizer = None,  # None means byte-level
                 block_size: int = 512,
                 train_split: float = 0.9):
        pass
    
    def load_dataset(self):
        """Load dataset from HuggingFace hub or local file."""
        pass
    
    def get_batch(self, split: str, batch_size: int):
        """Generate a batch of data for training or validation."""
        pass
```

#### Supported Data Sources
- HuggingFace datasets hub (via `datasets` library)
- Local text files (backward compatibility)
- Streaming mode for large datasets

#### Preprocessing Pipeline
- Text extraction from various dataset formats
- Tokenization (byte-level or tokenizer-based)
- Chunking into blocks of specified size
- Train/validation splitting

### 2. Integration with Training Script

Update `train.py` to:
- Accept dataset configuration via command-line arguments
- Use the new `DatasetLoader` class
- Maintain backward compatibility with Shakespeare dataset
- Support both streaming and non-streaming modes

### 3. Configuration Options

Add these command-line arguments:
```bash
--dataset_name       # HuggingFace dataset name or "shakespeare" for default
--dataset_config     # Dataset configuration/subset name
--streaming          # Enable streaming mode for large datasets
--text_column        # Column name containing text data
--block_size         # Context window size
--train_split        # Train/val split ratio
```

## Implementation Steps

### Step 1: Setup Dependencies
Ensure `datasets` library is in requirements.txt:
```
datasets>=2.14.0
```

### Step 2: Create `data.py` with Basic Structure
```python
from datasets import load_dataset
import numpy as np
import torch
from typing import Optional, Union

class DatasetLoader:
    # Implement the class structure
```

### Step 3: Implement HuggingFace Dataset Loading
- Use `load_dataset()` from HuggingFace
- Handle different dataset formats (text, json, csv)
- Support streaming mode
- Extract text from appropriate column

### Step 4: Implement Batching Logic
- Create random batch sampling for non-streaming
- Create sequential batching for streaming
- Handle boundary cases (end of dataset, short sequences)
- Support pin_memory and GPU transfer

### Step 5: Add Byte-Level Encoding Support
- Maintain current byte-level encoding as default
- Convert text to bytes using UTF-8 encoding
- Ensure vocab_size=256 compatibility

### Step 6: Update `train.py`
- Add argparse for command-line arguments
- Replace `fetch_data()` and `get_batch()` with `DatasetLoader`
- Add example usage in main block
- Keep Shakespeare as default for backward compatibility

### Step 7: Create Example Configurations
Document common dataset configurations:
- WikiText-103: `--dataset_name wikitext --dataset_config wikitext-103-raw-v1 --text_column text`
- OpenWebText: `--dataset_name openwebtext --text_column text --streaming`
- C4: `--dataset_name c4 --dataset_config en --text_column text --streaming`

## Testing Plan

1. **Test with Shakespeare** (backward compatibility)
   - Run training with default settings
   - Verify same behavior as before

2. **Test with WikiText-2** (small dataset)
   - Non-streaming mode
   - Verify batches are correctly formatted
   - Check training loss converges

3. **Test with OpenWebText** (large dataset)
   - Enable streaming mode
   - Verify memory usage stays low
   - Check batch generation is continuous

4. **Edge Cases**
   - Empty text handling
   - Very short documents
   - Unicode characters with byte-level encoding

## Code Example

Example usage after implementation:
```python
# Train on Shakespeare (default)
python train.py

# Train on WikiText-103
python train.py --dataset_name wikitext --dataset_config wikitext-103-raw-v1 --text_column text

# Train on OpenWebText with streaming
python train.py --dataset_name openwebtext --text_column text --streaming

# Train on custom local file
python train.py --dataset_name shakespeare
```

## Copilot Implementation Prompt

```
Create a data.py module for the BDH project that:

1. Implements a DatasetLoader class that can:
   - Load datasets from HuggingFace hub using the datasets library
   - Support streaming mode for large datasets
   - Handle byte-level encoding (UTF-8 to bytes, vocab_size=256)
   - Generate batches of data for training with shape (batch_size, block_size)
   - Support train/validation splitting

2. The DatasetLoader should:
   - Accept parameters: dataset_name, streaming, tokenizer (None for byte-level), block_size, train_split
   - Have a get_batch(split, batch_size) method that returns (x, y) tensors
   - x should be input tokens, y should be targets (x shifted by 1)
   - Handle GPU pinning and transfer like the current implementation
   - Support both random sampling (non-streaming) and sequential batching (streaming)

3. Update train.py to:
   - Add argparse for command-line arguments: --dataset_name, --dataset_config, --streaming, --text_column, --block_size, --train_split
   - Replace the hardcoded fetch_data() and get_batch() with DatasetLoader
   - Keep "shakespeare" as the default dataset for backward compatibility
   - For shakespeare mode, use the existing input.txt file

4. Make sure:
   - The code works with datasets.load_dataset() for HuggingFace datasets
   - Streaming mode uses datasets with streaming=True
   - Byte-level encoding converts text to UTF-8 bytes
   - Batches are properly formatted as torch.int64 tensors
   - The implementation matches the existing BDH architecture (vocab_size=256 by default)

5. Add examples in comments showing how to use different datasets:
   - wikitext/wikitext-103-raw-v1
   - openwebtext
   - c4/en

The current implementation is in train.py lines 46-83, which loads Shakespeare from input.txt as a memory-mapped file and samples random chunks. Preserve this behavior as the default while adding HuggingFace dataset support.
```

## Files to Modify/Create
- **Create**: `data.py` - New data loading module
- **Modify**: `train.py` - Add argparse and use DatasetLoader
- **Modify**: `requirements.txt` - Add datasets library
- **Create**: `examples/` - Optional directory with example training commands

## Dependencies
- `datasets>=2.14.0` (HuggingFace datasets library)
- `torch` (already present)
- `numpy` (already present)

## Success Criteria
- [ ] Can train on Shakespeare dataset (backward compatible)
- [ ] Can train on WikiText-2 with command-line flag
- [ ] Streaming mode works with OpenWebText
- [ ] Memory usage is reasonable with streaming
- [ ] Training loss converges on multiple datasets
- [ ] Code is well-documented with examples
- [ ] No breaking changes to existing API

## Related Tasks
- **Task 0002**: Tokenizer Support (will extend this to support BPE tokenizers)
- **Task 0009**: Configuration System (will replace argparse with config files)
