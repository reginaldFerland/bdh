#!/usr/bin/env python
"""Test the new features added to data.py."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from data import DatasetLoader, DatasetLoaderConfig


def test_iterator_protocol():
    """Test that the dataset loader can be used as an iterator."""
    print("Test 1: Iterator protocol (__iter__ and __next__)...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        batch_size=4,
        seed=42,
    )
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    # Test that we can iterate
    batch_count = 0
    for x_batch, y_batch in loader:
        assert x_batch.shape == (4, 64), f"Expected shape (4, 64), got {x_batch.shape}"
        assert y_batch.shape == (4, 64), f"Expected shape (4, 64), got {y_batch.shape}"
        batch_count += 1
        if batch_count >= 3:  # Just test a few iterations
            break
    
    assert batch_count == 3, f"Expected 3 batches, got {batch_count}"
    print(f"  ✓ Iterator protocol working! Got {batch_count} batches")
    print(f"  ✓ Batch shapes correct: {x_batch.shape}")


def test_cache_dir_support():
    """Test that cache_dir parameter is accepted (can't easily verify it's used)."""
    print("\nTest 2: cache_dir parameter support...")
    
    cache_path = Path("/tmp/bdh_cache_test")
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        cache_dir=cache_path,
        seed=42,
    )
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    # Verify loader works with cache_dir set
    x, y = loader.get_batch("train", batch_size=2)
    assert x.shape == (2, 64), f"Expected shape (2, 64), got {x.shape}"
    
    print(f"  ✓ cache_dir parameter accepted: {cache_path}")
    print(f"  ✓ Loader works correctly with cache_dir set")


def test_batch_size_in_config():
    """Test that batch_size can be set in config and used by iterator."""
    print("\nTest 3: batch_size in DatasetLoaderConfig...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        batch_size=6,  # Custom batch size
        seed=42,
    )
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    # Use iterator which should use config.batch_size
    for x_batch, y_batch in loader:
        assert x_batch.shape == (6, 64), f"Expected shape (6, 64), got {x_batch.shape}"
        assert y_batch.shape == (6, 64), f"Expected shape (6, 64), got {y_batch.shape}"
        break  # Just test one batch
    
    print(f"  ✓ Custom batch_size={config.batch_size} used by iterator")
    print(f"  ✓ Batch shape correct: {x_batch.shape}")


def test_numpy_contiguous_arrays():
    """Test that arrays are contiguous for optimal PyTorch performance."""
    print("\nTest 4: NumPy contiguous arrays optimization...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        seed=42,
    )
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    x, y = loader.get_batch("train", batch_size=4)
    
    # Check if tensors are contiguous
    assert x.is_contiguous(), "x tensor should be contiguous"
    assert y.is_contiguous(), "y tensor should be contiguous"
    
    print(f"  ✓ x tensor is contiguous: {x.is_contiguous()}")
    print(f"  ✓ y tensor is contiguous: {y.is_contiguous()}")
    print(f"  ✓ Optimal memory layout for PyTorch!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing new data.py features")
    print("=" * 60)
    
    test_iterator_protocol()
    test_cache_dir_support()
    test_batch_size_in_config()
    test_numpy_contiguous_arrays()
    
    print("\n" + "=" * 60)
    print("✅ All new feature tests passed!")
    print("=" * 60)
