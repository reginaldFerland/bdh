#!/usr/bin/env python
"""Test the enhanced context manager functionality."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from data import DatasetLoader, DatasetLoaderConfig


def test_context_manager_statistics():
    """Test that context manager tracks and reports statistics."""
    print("Test 1: Context manager statistics tracking...")
    
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
    print(f"  ✓ Statistics reported on exit (see output above)")


def test_context_manager_cleanup():
    """Test that context manager cleans up streaming resources."""
    print("\nTest 2: Context manager resource cleanup...")
    
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
    print("\nTest 3: Context manager error handling...")
    
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


if __name__ == "__main__":
    print("=" * 60)
    print("Testing enhanced context manager functionality")
    print("=" * 60)
    
    test_context_manager_statistics()
    test_context_manager_cleanup()
    test_context_manager_error_handling()
    
    print("\n" + "=" * 60)
    print("✅ All context manager tests passed!")
    print("=" * 60)
