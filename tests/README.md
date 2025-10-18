# BDH Test Suite

This directory contains tests for the BDH (Bidirectional Dot-product Hypernetwork) project.

## Test Organization

Tests are organized by the module or component being tested:

### `test_tokenizer_core.py`
Core tokenizer functionality tests for both byte and BPE tokenizers:
- Encoding and decoding (ASCII, Unicode, empty strings)
- Batch operations
- Special token handling
- Input validation and error handling
- Vocab size validation

### `test_tokenizer_persistence.py`
Tokenizer save/load and file I/O tests:
- Basic save/load for byte and BPE tokenizers
- Atomic save with temporary file cleanup
- Error handling during save operations
- Vocab size validation on load
- Special token ID validation on load
- Missing file detection

### `test_tokenizer_training.py`
Tokenizer training and preparation tests:
- Text extraction from dataset records
- Priority column detection
- Iterator utilities (`_CountingIterator`, `_validate_and_prepare_training_iterator`)
- Training configuration constants
- Error message quality
- Import error handling

### `test_data_loader.py`
Dataset loader functionality tests:
- Batch generation and NumPy optimization
- RNG seeding and reproducibility
- Device handling (CPU/GPU)
- Iterator protocol (`__iter__`, `__next__`)
- Context manager protocol (`__enter__`, `__exit__`)
- Configuration options (batch_size, cache_dir)
- Resource cleanup

## Running Tests

### Run all tests
```bash
# From project root
python -m pytest tests/

# Or with verbose output
python -m pytest tests/ -v

# Run with output from print statements
python -m pytest tests/ -s
```

### Run specific test file
```bash
python -m pytest tests/test_tokenizer_core.py
python -m pytest tests/test_tokenizer_persistence.py
python -m pytest tests/test_tokenizer_training.py
python -m pytest tests/test_data_loader.py

# Or run directly
python tests/test_tokenizer_core.py
```

### Run specific test
```bash
python -m pytest tests/test_tokenizer_core.py::test_byte_tokenizer_encode_decode_ascii
```

### Run tests matching a pattern
```bash
python -m pytest tests/ -k "bpe"  # All BPE-related tests
python -m pytest tests/ -k "save"  # All save-related tests
```

## Test Requirements

### Required
- PyTorch
- NumPy

### Optional (for BPE tests)
- `tokenizers` (HuggingFace Tokenizers library)
  ```bash
  pip install tokenizers
  ```

### Optional (for dataset tests)
- `datasets` (HuggingFace Datasets library)
  ```bash
  pip install datasets
  ```

Tests that require optional dependencies are automatically skipped if the dependency is not installed (using `@pytest.mark.skipif`).

## Adding New Tests

When adding new test files:
1. Name them `test_*.py` for pytest discovery
2. Organize by module/component being tested
3. Include docstrings explaining what is tested
4. Group related tests within the file using comments
5. Update this README with a description
6. Use `@pytest.mark.skipif` for optional dependencies
7. Ensure tests can run independently

## Test Structure

Each test file follows this structure:
```python
#!/usr/bin/env python
"""Brief description of what this file tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
# ... other imports ...

# ============================================================================
# Test Category Name
# ============================================================================

def test_something():
    """Test description."""
    # ... test implementation ...
    print("  ✓ Success message")

# ... more tests ...

if __name__ == "__main__":
    print("=" * 70)
    print("Running Test Suite Name")
    print("=" * 70)
    
    # Run tests directly
    test_something()
    # ...
```

## Continuous Integration

Tests should pass before merging pull requests. To run the full test suite:
```bash
./run_tests.sh
```

## Test Coverage

To run tests with coverage reporting:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.
