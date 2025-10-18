#!/usr/bin/env python
"""Core tokenizer functionality tests.

Tests encoding, decoding, batch operations, special tokens, and validation
for both byte and BPE tokenizers.
"""

import sys
import tempfile
import warnings
from pathlib import Path

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
# Byte Tokenizer Tests
# ============================================================================

def test_byte_tokenizer_encode_decode_ascii():
    """Test byte tokenizer with ASCII text."""
    manager = TokenizerManager("byte", vocab_size=256)
    text = "Hello, World!"
    encoded = manager.encode(text)
    decoded = manager.decode(encoded)
    
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    assert all(0 <= tid < 256 for tid in encoded), "Token IDs should be in byte range"
    print(f"  ✓ ASCII round-trip: '{text}' -> {len(encoded)} tokens -> '{decoded}'")


def test_byte_tokenizer_encode_decode_unicode():
    """Test byte tokenizer with Unicode text."""
    manager = TokenizerManager("byte")
    text = "Hello, 世界! 🌍"
    encoded = manager.encode(text)
    decoded = manager.decode(encoded)
    
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    assert all(0 <= tid < 256 for tid in encoded), "Token IDs should be in byte range"
    print(f"  ✓ Unicode round-trip: '{text}' -> {len(encoded)} tokens -> '{decoded}'")


def test_byte_tokenizer_empty_string():
    """Test byte tokenizer with empty input."""
    manager = TokenizerManager("byte")
    encoded = manager.encode("")
    decoded = manager.decode([])
    
    assert encoded == []
    assert decoded == ""
    print("  ✓ Empty string handling works")


def test_byte_tokenizer_batch_encode_decode():
    """Test batch encoding/decoding with byte tokenizer."""
    manager = TokenizerManager("byte")
    texts = ["Hello", "World", "Testing 123"]
    
    encoded_batch = manager.encode_batch(texts)
    assert len(encoded_batch) == 3
    
    decoded_batch = manager.decode_batch(encoded_batch)
    assert decoded_batch == texts
    
    print(f"  ✓ Batch encode/decode: {texts} -> {decoded_batch}")


def test_byte_tokenizer_special_tokens():
    """Test that byte tokenizer handles special tokens correctly (none by design)."""
    manager = TokenizerManager("byte")
    
    assert manager.pad_token_id is None
    assert manager.bos_token_id is None
    assert manager.eos_token_id is None
    assert manager.unk_token_id is None
    
    text = "Test"
    encoded_no_special = manager.encode(text, add_special_tokens=False)
    encoded_with_special = manager.encode(text, add_special_tokens=True)
    assert encoded_no_special == encoded_with_special
    
    print("  ✓ Byte tokenizer correctly has no special tokens")


def test_byte_tokenizer_no_training():
    """Test that byte tokenizer cannot be trained."""
    manager = TokenizerManager("byte")
    with pytest.raises(ValueError, match="do not require training"):
        manager.train_tokenizer(["text"], "/tmp/test")
    print("  ✓ Byte tokenizer correctly rejects training attempt")


# ============================================================================
# BPE Tokenizer Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_encode_decode():
    """Test BPE tokenizer encoding and decoding."""
    texts = ["hello world", "testing tokenizer"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        test_text = "hello world"
        encoded = manager.encode(test_text)
        assert len(encoded) > 0
        assert all(0 <= tid < manager.vocab_size for tid in encoded)
        
        decoded = manager.decode(encoded, skip_special_tokens=True)
        assert "hello" in decoded.lower() and "world" in decoded.lower()
        
        print(f"  ✓ Encoded '{test_text}' to {len(encoded)} tokens")
        print(f"    Decoded back to: '{decoded}'")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_special_tokens():
    """Test BPE tokenizer special token handling."""
    texts = ["hello world"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        test_text = "hello"
        
        encoded_no_special = manager.encode(test_text, add_special_tokens=False)
        encoded_with_special = manager.encode(test_text, add_special_tokens=True)
        
        assert len(encoded_with_special) > len(encoded_no_special), \
            "add_special_tokens=True should add BOS/EOS tokens"
        assert encoded_with_special[0] == manager.bos_token_id, \
            f"First token should be BOS ({manager.bos_token_id}), got {encoded_with_special[0]}"
        assert encoded_with_special[-1] == manager.eos_token_id, \
            f"Last token should be EOS ({manager.eos_token_id}), got {encoded_with_special[-1]}"
        
        print(f"  ✓ Special tokens correctly added")
        print(f"    Without special tokens: {len(encoded_no_special)} tokens")
        print(f"    With special tokens: {len(encoded_with_special)} tokens (BOS={manager.bos_token_id}, EOS={manager.eos_token_id})")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_decoder_restoration():
    """Test that ByteLevel decoder is properly restored when loading from disk."""
    texts = ["hello world", "test text"] * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = TokenizerManager("bpe", vocab_size=500)
        manager1.train_tokenizer(texts, tmpdir)
        
        test_text = "hello world"
        encoded = manager1.encode(test_text, add_special_tokens=False)
        decoded1 = manager1.decode(encoded, skip_special_tokens=True)
        
        manager2 = TokenizerManager.from_directory(tmpdir)
        decoded2 = manager2.decode(encoded, skip_special_tokens=True)
        
        assert decoded1 == decoded2, \
            f"Decoded text mismatch: original='{decoded1}', loaded='{decoded2}'"
        
        assert decoded1.strip() == test_text or decoded1 == test_text or test_text in decoded1, \
            f"Decoded text should contain or match original: '{decoded1}' vs '{test_text}'"
        
        print(f"  ✓ Decoder properly restored")
        print(f"    Original: '{test_text}'")
        print(f"    Decoded (trained): '{decoded1}'")
        print(f"    Decoded (loaded):  '{decoded2}'")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_bpe_tokenizer_batch_operations():
    """Test BPE tokenizer batch encoding/decoding."""
    texts = ["hello world", "test text", "batch processing"] * 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        manager.train_tokenizer(texts, tmpdir)
        
        test_texts = ["hello", "world", "test"]
        encoded_batch = manager.encode_batch(test_texts)
        assert len(encoded_batch) == 3
        assert all(isinstance(enc, list) for enc in encoded_batch)
        
        decoded_batch = manager.decode_batch(encoded_batch, skip_special_tokens=True)
        assert len(decoded_batch) == 3
        
        print(f"  ✓ Batch encoding: {test_texts} -> {[len(e) for e in encoded_batch]} tokens")
        print(f"  ✓ Batch decoding: {decoded_batch}")


# ============================================================================
# Validation and Error Handling Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_untrained_tokenizer_error():
    """Test that untrained non-byte tokenizer raises error."""
    manager = TokenizerManager("bpe", vocab_size=500)
    
    with pytest.raises(RuntimeError, match="not been loaded or trained"):
        manager.encode("test")
    
    print("  ✓ Untrained tokenizer correctly raises error on encode")


def test_invalid_tokenizer_type():
    """Test that invalid tokenizer type is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        texts = ["test"] * 10
        manager = TokenizerManager("invalid_type", vocab_size=500)
        
        if HAS_TOKENIZERS:
            with pytest.raises(ValueError, match="Unsupported tokenizer_type"):
                manager.train_tokenizer(texts, tmpdir)
            print("  ✓ Invalid tokenizer type correctly rejected")


def test_invalid_vocab_size():
    """Test that invalid vocab_size values are rejected."""
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        TokenizerManager("byte", vocab_size=-1)
    
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        TokenizerManager("byte", vocab_size=0)
    
    with pytest.raises(ValueError, match="unusually large"):
        TokenizerManager("bpe", vocab_size=2_000_000)
    
    print("  ✓ Invalid vocab_size values correctly rejected")


def test_vocab_size_too_small_for_trainable():
    """Test that vocab_size < minimum for trainable tokenizers is rejected."""
    if not HAS_TOKENIZERS:
        pytest.skip("tokenizers library not installed")
    
    with pytest.raises(ValueError, match="too small"):
        TokenizerManager("bpe", vocab_size=50)
    
    with pytest.raises(ValueError, match="too small"):
        TokenizerManager("wordpiece", vocab_size=100)
    
    manager = TokenizerManager("bpe", vocab_size=104)
    assert manager.requested_vocab_size == 104
    
    print("  ✓ Too-small vocab_size correctly rejected for trainable tokenizers")
    print("  ✓ Minimum vocab_size (104) accepted")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_empty_training_data():
    """Test that training with no valid data raises an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        
        with pytest.raises(ValueError, match="No valid text samples"):
            manager.train_tokenizer([], tmpdir)
        
        with pytest.raises(ValueError, match="No valid text samples"):
            manager.train_tokenizer([None, "", 123, []], tmpdir)
        
        print("  ✓ Empty training data correctly rejected")


@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_small_dataset_warning():
    """Test that training with small dataset issues a warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TokenizerManager("bpe", vocab_size=500)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.train_tokenizer(["hello world"] * 50, tmpdir)
            
            # Should get two warnings: one for small dataset, one for vocab size mismatch
            assert len(w) == 2
            
            # Check that both warnings are present
            messages = [str(warning.message) for warning in w]
            assert any("50" in msg or "samples" in msg for msg in messages), \
                "Should warn about small sample count"
            assert any("vocab size" in msg.lower() and ("smaller" in msg.lower() or "insufficient" in msg.lower()) 
                for msg in messages), \
                "Should warn about vocab size being smaller than requested"
        
        print("  ✓ Small dataset and vocab size warnings issued correctly")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Core Tokenizer Tests")
    print("=" * 70)
    
    print("\n--- Byte Tokenizer Tests ---")
    test_byte_tokenizer_encode_decode_ascii()
    test_byte_tokenizer_encode_decode_unicode()
    test_byte_tokenizer_empty_string()
    test_byte_tokenizer_batch_encode_decode()
    test_byte_tokenizer_special_tokens()
    test_byte_tokenizer_no_training()
    
    if HAS_TOKENIZERS:
        print("\n--- BPE Tokenizer Tests ---")
        test_bpe_tokenizer_encode_decode()
        test_bpe_tokenizer_special_tokens()
        test_bpe_decoder_restoration()
        test_bpe_tokenizer_batch_operations()
        
        print("\n--- Validation Tests ---")
        test_untrained_tokenizer_error()
        test_invalid_tokenizer_type()
        test_invalid_vocab_size()
        test_vocab_size_too_small_for_trainable()
        test_empty_training_data()
        test_small_dataset_warning()
    else:
        print("\n--- Skipping BPE Tests (tokenizers library not installed) ---")
    
    print("\n" + "=" * 70)
    print("All core tokenizer tests passed! ✓")
    print("=" * 70)
