# BDH Test Suite

This directory contains tests for the BDH (Bidirectional Dot-product Hypernetwork) project.

## Test Files

### `test_data_fixes.py`
Tests for dataset loader fixes and improvements:
- Streaming batch with NumPy optimization
- RNG initialization with seed for reproducibility
- Device parameter validation

### `test_new_features.py`
Tests for new dataset loader features:
- Iterator protocol (`__iter__` and `__next__`)
- Cache directory support
- Batch size configuration
- NumPy contiguous arrays optimization

## Running Tests

### Run all tests
```bash
# From project root
python -m pytest tests/

# Or run individually
python tests/test_data_fixes.py
python tests/test_new_features.py
```

### Run specific test file
```bash
python tests/test_data_fixes.py
python tests/test_new_features.py
```

## Test Requirements

- PyTorch
- NumPy
- datasets (HuggingFace, optional for some tests)

## Adding New Tests

When adding new test files:
1. Name them `test_*.py` for pytest discovery
2. Include docstrings explaining what is tested
3. Add entry to this README
4. Ensure tests can run independently
