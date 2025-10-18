#!/usr/bin/env python
"""Tokenizer persistence tests.

Tests save/load functionality, atomic saves with cleanup on error,
special token validation, and vocab size validation.
"""

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer_utils import TokenizerManager

# Check if tokenizers library is available
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


# ============================================================================
# Basic Save/Load Tests
# ============================================================================

def test_byte_tokenizer_save_load():
    """Test byte tokenizer persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = TokenizerManager("byte")
        save_path = manager1.save(tmpdir)
        
        assert (save_path / "tokenizer_config.json").exists()
        
        manager2 = TokenizerManager.from_directory(save_path)
        assert manager2.tokenizer_type == "byte"
        assert manager2.vocab_size == 256
        
        text = "Test text"
        encoded1 = manager1.encode(text)
        encoded2 = manager2.encode(text)
        assert encoded1 == encoded2
        
        print(f"  ✓ Byte tokenizer saved to and loaded from {save_path}")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_save_load():
    """Test BPE tokenizer persistence."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        original_vocab_size = manager1.vocab_size
        
        manager2 = TokenizerManager.from_directory(tmpdir)
        assert manager2.tokenizer_type == "bpe"
        assert manager2.vocab_size == original_vocab_size
        
        test_text = "hello world"
        encoded1 = manager1.encode(test_text)
        encoded2 = manager2.encode(test_text)
        assert encoded1 == encoded2
        
        print(f"  ✓ BPE tokenizer saved and loaded successfully")
        print(f"    Vocab size preserved: {original_vocab_size}")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_training():
    """Test BPE tokenizer training and persistence."""
    texts = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love",
        "love and happiness",
    ] * 200
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        save_path = manager.train_tokenizer(texts, tmpdir)
        
        assert manager.vocab_size <= 500
        assert manager.vocab_size > 0
        
        assert manager.pad_token_id is not None
        assert manager.bos_token_id is not None
        assert manager.eos_token_id is not None
        assert manager.unk_token_id is not None
        
        print(f"  ✓ Trained BPE tokenizer with vocab_size={manager.vocab_size}")
        print(f"    Special tokens: PAD={manager.pad_token_id}, BOS={manager.bos_token_id}, "
              f"EOS={manager.eos_token_id}, UNK={manager.unk_token_id}")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_special_token_preservation():
    """Test that BPE tokenizer special tokens are preserved after loading."""
    texts = ["hello world"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        test_text = "hello"
        encoded_with_special = manager1.encode(test_text, add_special_tokens=True)
        
        manager2 = TokenizerManager.from_directory(tmpdir)
        encoded_loaded_with_special = manager2.encode(test_text, add_special_tokens=True)
        
        assert encoded_with_special == encoded_loaded_with_special, \
            "Original and loaded tokenizer should produce identical results"
        
        assert encoded_loaded_with_special[0] == manager2.bos_token_id
        assert encoded_loaded_with_special[-1] == manager2.eos_token_id
        
        print(f"  ✓ Loaded tokenizer preserves special token behavior")


# ============================================================================
# Atomic Save Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_atomic_save_creates_temp_files():
    """Test that atomic save creates and cleans up temporary files correctly."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        # Check that final files exist
        config_path = Path(tmpdir) / "tokenizer_config.json"
        tokenizer_path = Path(tmpdir) / "tokenizer.json"
        assert config_path.exists(), "Config file should exist"
        assert tokenizer_path.exists(), "Tokenizer file should exist"
        
        # Check that temp files don't exist after save
        temp_config = Path(tmpdir) / "tokenizer_config.json.tmp"
        temp_tokenizer = Path(tmpdir) / "tokenizer.json.tmp"
        assert not temp_config.exists(), "Temp config file should be cleaned up"
        assert not temp_tokenizer.exists(), "Temp tokenizer file should be cleaned up"
        
        print("  ✓ Atomic save creates final files and cleans up temp files")


def test_byte_tokenizer_save_cleanup_on_json_error():
    """Test that temp config file is cleaned up when JSON write fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("byte")
        
        # Mock json.dump to raise an error
        with patch('json.dump', side_effect=IOError("Simulated write error")):
            tmpdir_path = Path(tmpdir)
            temp_config = tmpdir_path / "tokenizer_config.json.tmp"
            
            # Save should fail
            with pytest.raises(IOError, match="Simulated write error"):
                manager.save(tmpdir)
            
            # Temp file should NOT exist after failure
            assert not temp_config.exists(), \
                f"Temp config file should be cleaned up on error, but {temp_config} exists"
            
            print("  ✓ Temp config file cleaned up on JSON write error")


def test_byte_tokenizer_save_cleanup_on_rename_error():
    """Test cleanup when rename operation fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("byte")
        tmpdir_path = Path(tmpdir)
        temp_config = tmpdir_path / "tokenizer_config.json.tmp"
        
        # Patch replace to fail
        original_replace = Path.replace
        def mock_replace(self, target):
            if self.name == "tokenizer_config.json.tmp":
                raise OSError("Simulated rename error")
            return original_replace(self, target)
        
        with patch.object(Path, 'replace', mock_replace):
            # Save should fail
            with pytest.raises(OSError, match="Simulated rename error"):
                manager.save(tmpdir)
            
            # Temp file should be cleaned up
            assert not temp_config.exists(), \
                f"Temp config file should be cleaned up on rename error"
            
            print("  ✓ Temp config file cleaned up on rename error")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_save_cleanup_on_tokenizer_save_error():
    """Test that temp tokenizer file is cleaned up when save fails."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a tokenizer
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        # Now try to save again but simulate an error during tokenizer.json save
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2_path = Path(tmpdir2)
            temp_tokenizer = tmpdir2_path / "tokenizer.json.tmp"
            
            # Patch the tokenizer's save method to fail
            original_save = manager.tokenizer.save
            def mock_save(path):
                # Create temp file first
                Path(path).touch()
                raise IOError("Simulated tokenizer save error")
            
            with patch.object(manager.tokenizer, 'save', mock_save):
                # Save should fail
                with pytest.raises(IOError, match="Simulated tokenizer save error"):
                    manager.save(tmpdir2)
                
                # Temp tokenizer file should be cleaned up
                assert not temp_tokenizer.exists(), \
                    f"Temp tokenizer file should be cleaned up on save error"
                
                print("  ✓ Temp tokenizer file cleaned up on save error")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_config_saved_even_if_tokenizer_fails():
    """Test that if tokenizer.json save fails, config is not corrupted."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a tokenizer
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        # Save successfully first
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2_path = Path(tmpdir2)
            config_path = tmpdir2_path / "tokenizer_config.json"
            
            # Patch tokenizer save to fail
            with patch.object(manager.tokenizer, 'save', side_effect=IOError("Mock error")):
                with pytest.raises(IOError):
                    manager.save(tmpdir2)
                
                # Config should still be saved (it's saved first)
                assert config_path.exists(), \
                    "Config file should exist even if tokenizer save fails (saved first)"
                
                # Verify config is valid JSON
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                assert config_data['tokenizer_type'] == 'bpe'
                
                print("  ✓ Config saved successfully even when tokenizer save fails")


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_vocab_size_validation():
    """Test that loading validates vocab size matches metadata."""
    texts = ["hello world"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        config_path = Path(tmpdir) / "tokenizer_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        original_vocab = config["vocab_size"]
        config["vocab_size"] = 999999
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        try:
            manager2 = TokenizerManager.from_directory(tmpdir)
            assert False, "Should have raised ValueError for vocab size mismatch"
        except ValueError as e:
            assert "vocab size mismatch" in str(e).lower()
            print(f"  ✓ Vocab size validation caught mismatch: {str(e)[:80]}...")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_special_token_validation_on_load():
    """Test that loading validates special token IDs match metadata."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train and save a tokenizer
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        # Corrupt the metadata by changing a special token ID
        config_path = Path(tmpdir) / "tokenizer_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Change BOS token ID to something invalid
        config["bos_token_id"] = 99999
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Try to load - should fail with special token mismatch error
        try:
            manager2 = TokenizerManager.from_directory(tmpdir)
            assert False, "Should have raised ValueError for special token mismatch"
        except ValueError as e:
            error_msg = str(e)
            assert "special token id mismatch" in error_msg.lower(), \
                f"Error should mention special token mismatch: {error_msg}"
            assert "BOS" in error_msg, \
                f"Error should identify BOS token as mismatched: {error_msg}"
            assert "99999" in error_msg, \
                f"Error should show the mismatched value: {error_msg}"
            print(f"  ✓ Special token validation caught mismatch")
            print(f"    Error message: {error_msg[:120]}...")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_special_token_validation_passes_for_valid_tokenizer():
    """Test that valid tokenizers pass special token validation."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train and save a tokenizer
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        # Load it back - should work fine
        manager2 = TokenizerManager.from_directory(tmpdir)
        
        # Verify all special tokens match
        assert manager2.pad_token_id == manager1.pad_token_id
        assert manager2.bos_token_id == manager1.bos_token_id
        assert manager2.eos_token_id == manager1.eos_token_id
        assert manager2.unk_token_id == manager1.unk_token_id
        
        print("  ✓ Valid tokenizer passes special token validation")
        print(f"    PAD={manager2.pad_token_id}, BOS={manager2.bos_token_id}, "
              f"EOS={manager2.eos_token_id}, UNK={manager2.unk_token_id}")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_multiple_special_token_mismatches():
    """Test error message when multiple special tokens are mismatched."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train and save a tokenizer
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        # Corrupt multiple special token IDs
        config_path = Path(tmpdir) / "tokenizer_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["pad_token_id"] = 88888
        config["eos_token_id"] = 99999
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Try to load - should fail with error listing all mismatches
        try:
            manager2 = TokenizerManager.from_directory(tmpdir)
            assert False, "Should have raised ValueError for special token mismatch"
        except ValueError as e:
            error_msg = str(e)
            assert "PAD" in error_msg, f"Error should identify PAD token: {error_msg}"
            assert "EOS" in error_msg, f"Error should identify EOS token: {error_msg}"
            assert "88888" in error_msg, f"Error should show PAD mismatch value: {error_msg}"
            assert "99999" in error_msg, f"Error should show EOS mismatch value: {error_msg}"
            print(f"  ✓ Multiple mismatches reported correctly")
            print(f"    Error lists both PAD and EOS mismatches")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_missing_tokenizer_file():
    """Test error when tokenizer file is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "tokenizer_type": "bpe",
                "vocab_size": 500,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "unk_token_id": 3,
            }, f)
        
        with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
            TokenizerManager.from_directory(tmpdir)
        
        print("  ✓ Missing tokenizer file correctly detected")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tokenizer Persistence Tests")
    print("=" * 70)
    
    print("\n--- Basic Save/Load Tests ---")
    test_byte_tokenizer_save_load()
    
    if HAS_TOKENIZERS:
        test_bpe_tokenizer_save_load()
        test_bpe_tokenizer_training()
        test_bpe_special_token_preservation()
        
        print("\n--- Atomic Save Tests ---")
        test_atomic_save_creates_temp_files()
        test_byte_tokenizer_save_cleanup_on_json_error()
        test_byte_tokenizer_save_cleanup_on_rename_error()
        test_bpe_tokenizer_save_cleanup_on_tokenizer_save_error()
        test_bpe_tokenizer_config_saved_even_if_tokenizer_fails()
        
        print("\n--- Validation Tests ---")
        test_vocab_size_validation()
        test_special_token_validation_on_load()
        test_special_token_validation_passes_for_valid_tokenizer()
        test_multiple_special_token_mismatches()
        test_missing_tokenizer_file()
    else:
        print("\n--- Skipping BPE Tests (tokenizers library not installed) ---")
    
    print("\n" + "=" * 70)
    print("All persistence tests passed! ✓")
    print("=" * 70)
