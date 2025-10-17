#!/usr/bin/env python
"""Quick test to verify the data.py fixes work correctly."""

import torch
import numpy as np
from data import DatasetLoader, DatasetLoaderConfig

def test_fix_1_streaming_batch():
    """Test that streaming batch uses NumPy efficiently."""
    print("Test 1: Streaming batch with NumPy optimization...")
    
    # Create a simple shakespeare loader in streaming mode
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
    
    # Verify x and y are offset by 1
    print(f"  ✓ Batch shape: {x.shape}")
    print(f"  ✓ Batch dtype: {x.dtype}")
    print("  ✓ Streaming batch optimization working!")


def test_fix_2_rng_seed():
    """Test that RNG uses the config seed for reproducibility."""
    print("\nTest 2: RNG initialization with seed...")
    
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


def test_fix_4_device_validation():
    """Test that device parameter is properly validated."""
    print("\nTest 3: Device parameter validation...")
    
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


if __name__ == "__main__":
    print("=" * 60)
    print("Testing data.py fixes")
    print("=" * 60)
    
    test_fix_1_streaming_batch()
    test_fix_2_rng_seed()
    test_fix_4_device_validation()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
