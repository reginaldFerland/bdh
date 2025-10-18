#!/usr/bin/env python
"""Tokenizer training tests.

Tests text extraction, iterator utilities, training data validation,
and error messages.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer_utils import (
    TokenizerManager,
    extract_text_from_record,
    DEFAULT_TEXT_COLUMNS,
    _validate_training_iterator,
)

# Check if tokenizers library is available
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


# ============================================================================
# Text Extraction Tests
# ============================================================================

def test_extract_text_from_record_explicit_column():
    """Test extract_text_from_record with explicit text_column."""
    record = {
        "custom_field": "Custom text",
        "text": "Default text",
        "other": "Other data",
    }
    
    result = extract_text_from_record(record, text_column="custom_field")
    assert result == "Custom text", f"Expected 'Custom text', got '{result}'"
    print("  ✓ Explicit text_column parameter works")


def test_extract_text_from_record_priority_columns():
    """Test extract_text_from_record with priority column detection."""
    # Test priority: text > content > article > body
    record1 = {"body": "Body text", "content": "Content text"}
    result1 = extract_text_from_record(record1)
    assert result1 == "Content text", "Should prioritize 'content' over 'body'"
    
    record2 = {"article": "Article text", "body": "Body text"}
    result2 = extract_text_from_record(record2)
    assert result2 == "Article text", "Should prioritize 'article' over 'body'"
    
    record3 = {"text": "Text field", "content": "Content field"}
    result3 = extract_text_from_record(record3)
    assert result3 == "Text field", "Should prioritize 'text' over all others"
    
    print("  ✓ Priority column detection works correctly")


def test_extract_text_from_record_fallback():
    """Test extract_text_from_record fallback to any string value."""
    record = {
        "some_field": "Found text",
        "number": 123,
        "boolean": True,
    }
    
    result = extract_text_from_record(record)
    assert result == "Found text", f"Expected 'Found text', got '{result}'"
    print("  ✓ Fallback to any string value works")


def test_extract_text_from_record_empty():
    """Test extract_text_from_record with no valid text."""
    record = {"number": 123, "boolean": True, "none": None}
    
    result = extract_text_from_record(record)
    assert result == "", f"Expected empty string, got '{result}'"
    print("  ✓ Empty result for records with no text")


def test_extract_text_from_record_custom_priority():
    """Test extract_text_from_record with custom priority columns."""
    record = {
        "summary": "Summary text",
        "description": "Description text",
        "text": "Text field",
    }
    
    # With custom priority, should find 'summary' first
    result = extract_text_from_record(
        record, 
        priority_columns=("summary", "description")
    )
    assert result == "Summary text", f"Expected 'Summary text', got '{result}'"
    print("  ✓ Custom priority columns work")


def test_default_text_columns_constant():
    """Test that DEFAULT_TEXT_COLUMNS is accessible and correct."""
    assert DEFAULT_TEXT_COLUMNS == ("text", "content", "article", "body")
    print(f"  ✓ DEFAULT_TEXT_COLUMNS = {DEFAULT_TEXT_COLUMNS}")


# ============================================================================
# Iterator Utility Tests
# ============================================================================

def test_validate_training_iterator_preserves_data():
    """Test that _validate_training_iterator doesn't lose data during iteration."""
    data = ["hello", "world", "test", "data"]
    
    validated_iter = _validate_training_iterator(iter(data), max_peek=10)
    
    # Iterate and verify all items are yielded
    result = list(validated_iter)
    assert result == data, f"Expected {data}, got {result}"
    
    print(f"  ✓ _validate_training_iterator preserves all {len(data)} items")


def test_validate_training_iterator_with_mixed_data():
    """Test the validation of training iterators with mixed valid and empty strings."""
    data = ["valid1", "", "valid2", "", "valid3"]
    
    validated_iter = _validate_training_iterator(iter(data), max_peek=10)
    
    result = list(validated_iter)
    # The function validates but may not filter empty strings - check actual behavior
    # Empty strings are filtered during tokenizer training, not in this utility
    expected = ["valid1", "", "valid2", "", "valid3"]
    assert result == expected, f"Expected {expected}, got {result}"
    
    print(f"  ✓ Validation iterator preserves all items including empty strings")


# ============================================================================
# Constants and Configuration Tests
# ============================================================================

def test_tokenizer_manager_constants():
    """Test that TokenizerManager validation config constants are defined correctly."""
    # Now these are in the validation config
    assert hasattr(TokenizerManager, 'validation')
    assert hasattr(TokenizerManager.validation, 'min_vocab_buffer')
    assert TokenizerManager.validation.min_vocab_buffer == 100
    print(f"  ✓ validation.min_vocab_buffer = {TokenizerManager.validation.min_vocab_buffer}")
    
    assert hasattr(TokenizerManager.validation, 'default_peek_limit')
    assert TokenizerManager.validation.default_peek_limit == 1000
    print(f"  ✓ validation.default_peek_limit = {TokenizerManager.validation.default_peek_limit}")
    
    assert hasattr(TokenizerManager.validation, 'max_vocab_size')
    assert TokenizerManager.validation.max_vocab_size == 1_000_000
    print(f"  ✓ validation.max_vocab_size = {TokenizerManager.validation.max_vocab_size:,}")
    
    assert hasattr(TokenizerManager.validation, 'min_training_samples')
    assert TokenizerManager.validation.min_training_samples == 100
    print(f"  ✓ validation.min_training_samples = {TokenizerManager.validation.min_training_samples}")
    
    assert hasattr(TokenizerManager.validation, 'recommended_training_samples')
    assert TokenizerManager.validation.recommended_training_samples == 1_000
    print(f"  ✓ validation.recommended_training_samples = {TokenizerManager.validation.recommended_training_samples:,}")
    
    assert hasattr(TokenizerManager.validation, 'production_training_samples')
    assert TokenizerManager.validation.production_training_samples == 10_000
    print(f"  ✓ validation.production_training_samples = {TokenizerManager.validation.production_training_samples:,}")


def test_min_vocab_buffer_used_in_validation():
    """Test that MIN_VOCAB_BUFFER constant is actually used in validation."""
    try:
        # Should fail because vocab_size < (4 special tokens + 100 buffer)
        manager = TokenizerManager("bpe", vocab_size=50)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "too small" in error_msg.lower()
        # Check that it uses the constant (minimum should be 104)
        assert "104" in error_msg or "Minimum is 104" in error_msg
        print(f"  ✓ MIN_VOCAB_BUFFER used in validation: '{error_msg[:80]}...'")


# ============================================================================
# Error Message Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
def test_runtime_error_messages():
    """Test that RuntimeError messages are helpful."""
    manager = TokenizerManager("bpe", vocab_size=500)
    
    # Test encode error message
    try:
        manager.encode("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        error_msg = str(e)
        assert "not been loaded or trained" in error_msg
        assert "train_tokenizer()" in error_msg
        assert "from_directory()" in error_msg
        print(f"  ✓ Encode error message is helpful: '{error_msg[:60]}...'")
    
    # Test decode error message
    try:
        manager.decode([1, 2, 3])
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        error_msg = str(e)
        assert "not been loaded or trained" in error_msg
        print(f"  ✓ Decode error message is helpful")


def test_import_error_messages():
    """Test that ImportError messages are helpful when tokenizers library is missing."""
    # This test simulates the case where tokenizers is not available
    # We can't actually test this if tokenizers IS installed, so we skip
    if not HAS_TOKENIZERS:
        manager = TokenizerManager("bpe", vocab_size=500)
        try:
            manager.train_tokenizer(["test"], "/tmp/test")
            assert False, "Should have raised ImportError"
        except ImportError as e:
            assert "tokenizers" in str(e).lower()
            assert "pip install" in str(e).lower()
            print(f"  ✓ ImportError message is helpful")
    else:
        print("  ⊘ Skipping ImportError test (tokenizers is installed)")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tokenizer Training Tests")
    print("=" * 70)
    
    print("\n--- Text Extraction Tests ---")
    test_extract_text_from_record_explicit_column()
    test_extract_text_from_record_priority_columns()
    test_extract_text_from_record_fallback()
    test_extract_text_from_record_empty()
    test_extract_text_from_record_custom_priority()
    test_default_text_columns_constant()
    
    print("\n--- Iterator Utility Tests ---")
    test_validate_training_iterator_preserves_data()
    test_validate_training_iterator_with_mixed_data()
    
    print("\n--- Configuration Tests ---")
    test_tokenizer_manager_constants()
    test_min_vocab_buffer_used_in_validation()
    
    if HAS_TOKENIZERS:
        print("\n--- Error Message Tests ---")
        test_runtime_error_messages()
        test_import_error_messages()
    else:
        print("\n--- Skipping error message tests (tokenizers library not installed) ---")
    
    print("\n" + "=" * 70)
    print("All training tests passed! ✓")
    print("=" * 70)
