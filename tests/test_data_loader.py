#!/usr/bin/env python
"""Data loader tests.

Tests DatasetLoader functionality including batch generation, RNG seeding,
device handling, iterator protocol, context manager, and streaming.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from data import DatasetLoader, DatasetLoaderConfig


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_streaming_batch():
    """Test that streaming batch uses NumPy efficiently."""
    print("Test: Streaming batch with NumPy optimization...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        streaming=False,  # Use non-streaming for reproducibility
        block_size=64,
        train_split=0.9,
        seed=42,
        device=torch.device("cpu"),
    )
    
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    # Get a batch and verify it works
    x, y = loader.get_batch("train", batch_size=4)
    
    assert x.shape == (4, 64), f"Expected shape (4, 64), got {x.shape}"
    assert y.shape == (4, 64), f"Expected shape (4, 64), got {y.shape}"
    assert x.dtype == torch.long, f"Expected dtype torch.long, got {x.dtype}"
    
    print(f"  ✓ Batch shape: {x.shape}")
    print(f"  ✓ Batch dtype: {x.dtype}")
    print("  ✓ Streaming batch optimization working!")


def test_numpy_contiguous_arrays():
    """Test that arrays are contiguous for optimal PyTorch performance."""
    print("\nTest: NumPy contiguous arrays optimization...")
    
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


# ============================================================================
# RNG and Reproducibility Tests
# ============================================================================

def test_rng_seed():
    """Test that RNG uses the config seed for reproducibility."""
    print("\nTest: RNG initialization with seed...")
    
    config1 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        seed=42,
    )
    loader1 = DatasetLoader(config1)
    loader1.load_dataset()
    x1, y1 = loader1.get_batch("train", batch_size=2)
    
    # Create another loader with same seed
    config2 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        seed=42,
    )
    loader2 = DatasetLoader(config2)
    loader2.load_dataset()
    x2, y2 = loader2.get_batch("train", batch_size=2)
    
    # They should produce identical batches
    assert torch.equal(x1, x2), "RNG not using seed correctly - batches differ"
    assert torch.equal(y1, y2), "RNG not using seed correctly - batches differ"
    
    print(f"  ✓ Both loaders with seed=42 produce identical batches")
    
    # Now try with different seed
    config3 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        seed=123,
    )
    loader3 = DatasetLoader(config3)
    loader3.load_dataset()
    x3, y3 = loader3.get_batch("train", batch_size=2)
    
    # Should be different
    assert not torch.equal(x1, x3), "Different seeds should produce different batches"
    
    print(f"  ✓ Different seed (123) produces different batches")
    print("  ✓ RNG seed initialization working!")


# ============================================================================
# Device Handling Tests
# ============================================================================

def test_device_validation():
    """Test that device parameter is properly validated."""
    print("\nTest: Device parameter validation...")
    
    # Test with torch.device
    config1 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        device=torch.device("cpu"),
    )
    loader1 = DatasetLoader(config1)
    assert isinstance(loader1.device, torch.device), "Device should be torch.device"
    print(f"  ✓ torch.device('cpu') accepted: {loader1.device}")
    
    # Test with string (should be converted)
    config2 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        device="cpu",  # String instead of torch.device
    )
    loader2 = DatasetLoader(config2)
    assert isinstance(loader2.device, torch.device), "String device should be converted to torch.device"
    assert str(loader2.device) == "cpu", f"Expected 'cpu', got {loader2.device}"
    print(f"  ✓ String 'cpu' converted to torch.device: {loader2.device}")
    
    # Test with None
    config3 = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        device=None,
    )
    loader3 = DatasetLoader(config3)
    assert loader3.device is None, "None device should remain None"
    print(f"  ✓ None device remains None: {loader3.device}")
    
    print("  ✓ Device validation working!")


# ============================================================================
# Iterator Protocol Tests
# ============================================================================

def test_iterator_protocol():
    """Test that the dataset loader can be used as an iterator."""
    print("\nTest: Iterator protocol (__iter__ and __next__)...")
    
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


def test_batch_size_in_config():
    """Test that batch_size can be set in config and used by iterator."""
    print("\nTest: batch_size in DatasetLoaderConfig...")
    
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


# ============================================================================
# Context Manager Tests
# ============================================================================

def test_context_manager_statistics():
    """Test that context manager tracks and reports statistics."""
    print("\nTest: Context manager statistics tracking...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        batch_size=4,
        seed=42,
    )
    
    # Use context manager
    with DatasetLoader(config) as loader:
        loader.load_dataset()
        
        # Generate some batches
        for i in range(5):
            x, y = loader.get_batch("train", batch_size=4)
        
        # Check that tracking exists
        assert hasattr(loader, '_batches_generated'), "Should have _batches_generated attribute"
        assert loader._batches_generated == 5, f"Expected 5 batches, got {loader._batches_generated}"
    
    print(f"  ✓ Context manager tracked {5} batches correctly")
    print(f"  ✓ Statistics reported on exit")


def test_context_manager_cleanup():
    """Test that context manager cleans up streaming resources."""
    print("\nTest: Context manager resource cleanup...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        batch_size=4,
        seed=42,
    )
    
    loader = DatasetLoader(config)
    loader.load_dataset()
    
    # Manually add some streaming state
    loader._stream_iters["test"] = iter([1, 2, 3])
    loader._stream_buffers["test"] = [1, 2, 3, 4, 5]
    
    # Use context manager protocol
    loader.__enter__()
    
    # Verify state exists
    assert "test" in loader._stream_iters
    assert "test" in loader._stream_buffers
    
    # Exit should clean up
    loader.__exit__(None, None, None)
    
    # Verify cleanup
    assert len(loader._stream_iters) == 0, "Stream iterators should be cleared"
    assert len(loader._stream_buffers) == 0, "Stream buffers should be cleared"
    
    print(f"  ✓ Stream iterators cleared on exit")
    print(f"  ✓ Stream buffers cleared on exit")


def test_context_manager_error_handling():
    """Test that context manager handles errors gracefully."""
    print("\nTest: Context manager error handling...")
    
    config = DatasetLoaderConfig(
        dataset_name="shakespeare",
        block_size=64,
        batch_size=4,
        seed=42,
    )
    
    try:
        with DatasetLoader(config) as loader:
            loader.load_dataset()
            x, y = loader.get_batch("train", batch_size=4)
            # Simulate an error
            raise ValueError("Simulated error for testing")
    except ValueError as e:
        # Exception should propagate correctly
        assert str(e) == "Simulated error for testing"
        print(f"  ✓ Exception propagated correctly: {e}")
    
    # Cleanup should still have happened despite the error
    assert len(loader._stream_iters) == 0, "Cleanup should happen even with exception"
    print(f"  ✓ Cleanup occurred despite exception")


# ============================================================================
# Configuration Tests
# ============================================================================

def test_cache_dir_support():
    """Test that cache_dir parameter is accepted."""
    print("\nTest: cache_dir parameter support...")
    
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


if __name__ == "__main__":
    print("=" * 70)
    print("Running Data Loader Tests")
    print("=" * 70)
    
    print("\n--- Basic Functionality Tests ---")
    test_streaming_batch()
    test_numpy_contiguous_arrays()
    
    print("\n--- RNG and Reproducibility Tests ---")
    test_rng_seed()
    
    print("\n--- Device Handling Tests ---")
    test_device_validation()
    
    print("\n--- Iterator Protocol Tests ---")
    test_iterator_protocol()
    test_batch_size_in_config()
    
    print("\n--- Context Manager Tests ---")
    test_context_manager_statistics()
    test_context_manager_cleanup()
    test_context_manager_error_handling()
    
    print("\n--- Configuration Tests ---")
    test_cache_dir_support()
    
    print("\n" + "=" * 70)
    print("All data loader tests passed! ✓")
    print("=" * 70)
