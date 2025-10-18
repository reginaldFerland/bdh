"""Tokenizer utilities for the BDH project."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import warnings

if TYPE_CHECKING:
    from tokenizers import Tokenizer as TokenizerType
else:
    TokenizerType = Any  # type: ignore

try:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None  # type: ignore
    models = None  # type: ignore
    normalizers = None  # type: ignore
    pre_tokenizers = None  # type: ignore
    trainers = None  # type: ignore


# Module-level constants
DEFAULT_TEXT_COLUMNS = ("text", "content", "article", "body")


def extract_text_from_record(
    record: Dict[str, Any],
    text_column: Optional[str] = None,
    priority_columns: Tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
) -> str:
    """Extract text from a dataset record with priority column detection.
    
    This function attempts to extract text from a record dictionary using
    the following priority order:
    1. Explicit text_column if specified and present
    2. Priority columns (text, content, article, body by default)
    3. Any string value in the record
    
    Args:
        record: Dataset record dictionary
        text_column: Optional specific column to extract text from
        priority_columns: Tuple of column names to check in priority order
        
    Returns:
        Extracted text string, or empty string if no valid text found
        
    Examples:
        >>> record = {"text": "Hello", "meta": "data"}
        >>> extract_text_from_record(record)
        'Hello'
        
        >>> record = {"title": "Article", "body": "Content"}
        >>> extract_text_from_record(record)
        'Content'
        
        >>> record = {"custom_field": "Text"}
        >>> extract_text_from_record(record, text_column="custom_field")
        'Text'
    """
    # Explicit column takes precedence
    if text_column and text_column in record:
        value = record[text_column]
        return value if isinstance(value, str) else ""
    
    # Try priority columns first with early return
    for col in priority_columns:
        if col in record:
            value = record[col]
            if isinstance(value, str) and value:
                return value
    
    # Fall back to any string value in the record
    for value in record.values():
        if isinstance(value, str) and value:
            return value
    
    return ""


class _CountingIterator:
    """Lightweight iterator wrapper that counts items.
    
    After initial validation confirms the iterator contains valid data,
    this wrapper simply counts items without re-validating each one.
    This improves performance by avoiding wasteful type checks.
    """
    
    def __init__(self, iterator: Iterator[str], first_item: str):
        self._iterator = iterator
        self._first_item = first_item
        self._yielded_first = False
        self.count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        # Yield the first valid item we already found
        if not self._yielded_first:
            self._yielded_first = True
            self.count += 1
            return self._first_item
        
        # Yield remaining items without re-validation (trust the data source)
        # Validation already passed during peek phase in _validate_and_prepare_training_iterator
        item = next(self._iterator)  # Will raise StopIteration when done
        self.count += 1
        return item


def _validate_and_prepare_training_iterator(
    texts: Iterable[str],
    max_peek: int = 100,
) -> _CountingIterator:
    """Validate training data and prepare iterator with sample counting.
    
    This function checks that the iterator contains at least one valid text sample
    and returns a new iterator that validates and counts samples as they're consumed.
    
    Args:
        texts: Iterable of text strings for training
        max_peek: Maximum number of items to check for first valid sample
        
    Returns:
        A counting iterator that validates items and tracks count
        
    Raises:
        ValueError: If no valid text samples found within max_peek items
    """
    iterator = iter(texts)
    
    # Try to find the first valid item
    first_valid = None
    items_checked = 0
    
    for item in iterator:
        items_checked += 1
        if isinstance(item, str) and item:
            first_valid = item
            break
        if items_checked >= max_peek:
            break
    
    if first_valid is None:
        raise ValueError(
            f"No valid text samples found in first {items_checked} items. "
            "Ensure the input contains non-empty strings."
        )
    
    return _CountingIterator(iterator, first_valid)


@dataclass
class TokenizerMetadata:
    tokenizer_type: str
    vocab_size: int
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None


class TokenizerManager:
    """Lightweight wrapper around HuggingFace tokenizers with byte fallback."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    
    # Validation and size constants
    MAX_VOCAB_SIZE = 1_000_000
    MIN_TRAINING_SAMPLES = 100
    RECOMMENDED_TRAINING_SAMPLES = 1_000
    PRODUCTION_TRAINING_SAMPLES = 10_000
    MIN_VOCAB_BUFFER = 100  # Minimum tokens beyond special tokens
    DEFAULT_PEEK_LIMIT = 1000  # Default items to check in iterator validation
    
    # Error message constants
    _TOKENIZERS_IMPORT_ERROR = (
        "The `tokenizers` library is required for non-byte tokenizers. "
        "Install it with `pip install tokenizers>=0.15.0`."
    )
    _TOKENIZER_NOT_LOADED_ERROR = (
        "Tokenizer of type '{tokenizer_type}' has not been loaded or trained. "
        "Call train_tokenizer() to train a new tokenizer, or use from_directory() "
        "to load an existing tokenizer from disk."
    )
    _TOKENIZER_NOT_TRAINED_ERROR = (
        "Cannot save untrained tokenizer of type '{tokenizer_type}'. "
        "Call train_tokenizer() first."
    )

    def __init__(self, tokenizer_type: str = "byte", vocab_size: int = 256, _skip_validation: bool = False):
        self.tokenizer_type = tokenizer_type.lower()
        self.requested_vocab_size = vocab_size
        self.tokenizer: Optional[TokenizerType] = None
        self.vocab_size = 256 if self.tokenizer_type == "byte" else vocab_size
        self.pad_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.tokenizer_path: Optional[str] = None
        
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if vocab_size > self.MAX_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size {vocab_size} is unusually large (>{self.MAX_VOCAB_SIZE:,} tokens). "
                "This may cause memory issues. If intentional, adjust this limit."
            )
        
        if not _skip_validation and self.tokenizer_type != "byte":
            min_vocab_size = len(self.SPECIAL_TOKENS) + self.MIN_VOCAB_BUFFER
            if vocab_size < min_vocab_size:
                raise ValueError(
                    f"vocab_size {vocab_size} is too small for '{self.tokenizer_type}' tokenizer. "
                    f"Minimum is {min_vocab_size} ({len(self.SPECIAL_TOKENS)} special tokens + "
                    f"{self.MIN_VOCAB_BUFFER} regular tokens). Consider using 'byte' tokenizer for small vocabularies."
                )

        if self.tokenizer_type != "byte" and Tokenizer is None:
            raise ImportError(self._TOKENIZERS_IMPORT_ERROR)

        if self.tokenizer_type == "byte":
            self.vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []

        if self.tokenizer_type == "byte":
            encoded = list(text.encode("utf-8"))
            if add_special_tokens and self.bos_token_id is not None:
                encoded = [self.bos_token_id] + encoded
            if add_special_tokens and self.eos_token_id is not None:
                encoded = encoded + [self.eos_token_id]
            return encoded

        if self.tokenizer is None:
            raise RuntimeError(
                self._TOKENIZER_NOT_LOADED_ERROR.format(tokenizer_type=self.tokenizer_type)
            )
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        return list(encoding.ids)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if not token_ids:
            return ""

        if self.tokenizer_type == "byte":
            filtered = token_ids
            if skip_special_tokens:
                special_ids = {
                    sid
                    for sid in (
                        self.pad_token_id,
                        self.bos_token_id,
                        self.eos_token_id,
                        self.unk_token_id,
                    )
                    if sid is not None
                }
                if special_ids:
                    filtered = [tid for tid in filtered if tid not in special_ids]
            return bytes(filtered).decode("utf-8", errors="backslashreplace")

        if self.tokenizer is None:
            raise RuntimeError(
                self._TOKENIZER_NOT_LOADED_ERROR.format(tokenizer_type=self.tokenizer_type)
            )
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        """Encode multiple texts efficiently in batch mode."""
        if not texts:
            return []

        if self.tokenizer_type == "byte":
            return [self.encode(text, add_special_tokens) for text in texts]

        if self.tokenizer is None:
            raise RuntimeError(
                self._TOKENIZER_NOT_LOADED_ERROR.format(tokenizer_type=self.tokenizer_type)
            )
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [list(enc.ids) for enc in encodings]

    def decode_batch(
        self, token_id_lists: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode multiple token ID sequences efficiently in batch mode."""
        if not token_id_lists:
            return []

        if self.tokenizer_type == "byte":
            return [self.decode(token_ids, skip_special_tokens) for token_ids in token_id_lists]

        if self.tokenizer is None:
            raise RuntimeError(
                self._TOKENIZER_NOT_LOADED_ERROR.format(tokenizer_type=self.tokenizer_type)
            )
        return self.tokenizer.decode_batch(token_id_lists, skip_special_tokens=skip_special_tokens)

    def train_tokenizer(
        self,
        texts: Iterable[str],
        output_dir: Union[str, Path],
    ) -> Path:
        """Train tokenizer on the provided text iterator and save to disk."""
        if self.tokenizer_type == "byte":
            raise ValueError("Byte-level tokenizers do not require training.")
        
        if Tokenizer is None:
            raise ImportError(self._TOKENIZERS_IMPORT_ERROR)
        
        # Validate and prepare training iterator
        counting_iter = _validate_and_prepare_training_iterator(texts, max_peek=100)
        
        tokenizer, trainer = self._build_tokenizer_and_trainer()
        
        tokenizer.train_from_iterator(counting_iter, trainer=trainer)
        
        # Get final count and warn if too small
        final_count = counting_iter.count
        if final_count < self.MIN_TRAINING_SAMPLES:
            warnings.warn(
                f"Only {final_count} samples provided for training. "
                f"Tokenizer quality may be poor. Recommend at least {self.RECOMMENDED_TRAINING_SAMPLES:,} samples "
                f"for reasonable quality, {self.PRODUCTION_TRAINING_SAMPLES:,}+ for production use.",
                UserWarning
            )
        
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self._populate_special_token_ids()
        self._configure_bpe_post_processor()
        
        # Validate that actual vocab size is reasonable compared to requested
        if self.vocab_size < self.requested_vocab_size * 0.8:
            warnings.warn(
                f"Trained vocab size ({self.vocab_size}) is significantly smaller than "
                f"requested ({self.requested_vocab_size}). This typically indicates insufficient "
                f"or insufficiently diverse training data. Consider providing more text samples.",
                UserWarning
            )

        save_path = self.save(output_dir)
        return save_path

    def _build_tokenizer_and_trainer(self):
        if (
            Tokenizer is None
            or models is None
            or trainers is None
            or pre_tokenizers is None
        ):
            raise ImportError(self._TOKENIZERS_IMPORT_ERROR)
        
        tok_type = self.tokenizer_type

        if tok_type == "bpe":
            from tokenizers import decoders, processors
            
            tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            tokenizer.decoder = decoders.ByteLevel()
            
            trainer = trainers.BpeTrainer(
                vocab_size=self.requested_vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                show_progress=True,
            )
        elif tok_type == "wordpiece":
            if normalizers is None:
                raise ImportError(self._TOKENIZERS_IMPORT_ERROR)
            tokenizer = Tokenizer(models.WordPiece(unk_token=self.UNK_TOKEN))
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.requested_vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                show_progress=True,
            )
        elif tok_type == "unigram":
            if normalizers is None:
                raise ImportError(self._TOKENIZERS_IMPORT_ERROR)
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.UnigramTrainer(
                vocab_size=self.requested_vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                show_progress=True,
            )
        else:
            raise ValueError(
                f"Unsupported tokenizer_type '{self.tokenizer_type}'. "
                "Expected 'byte', 'bpe', 'wordpiece', or 'unigram'."
            )
        return tokenizer, trainer

    def save(self, directory: Union[str, Path]) -> Path:
        """Save tokenizer configuration and model to directory.
        
        Uses atomic writes to prevent corruption if interrupted.
        
        Args:
            directory: Path to directory where tokenizer should be saved
            
        Returns:
            Path object pointing to the save directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Write config atomically using temp file + rename
        config_path = directory / "tokenizer_config.json"
        temp_config = directory / "tokenizer_config.json.tmp"
        
        metadata = self.metadata()
        try:
            with temp_config.open("w", encoding="utf-8") as f:
                json.dump(metadata.__dict__, f, indent=2, sort_keys=True)
                f.flush()  # Flush Python buffer
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            # Atomic rename (POSIX guarantees atomicity)
            temp_config.replace(config_path)
        except Exception:
            # Clean up temp file on error
            temp_config.unlink(missing_ok=True)
            raise

        if self.tokenizer_type != "byte":
            if self.tokenizer is None:
                raise RuntimeError(
                    self._TOKENIZER_NOT_TRAINED_ERROR.format(tokenizer_type=self.tokenizer_type)
                )
            # Atomic save for tokenizer.json using temp file + rename
            tokenizer_path = directory / "tokenizer.json"
            temp_tokenizer = directory / "tokenizer.json.tmp"
            try:
                self.tokenizer.save(str(temp_tokenizer))
                # Ensure tokenizer file is synced to disk
                with open(temp_tokenizer, 'rb') as f:
                    os.fsync(f.fileno())
                # Atomic rename (POSIX guarantees atomicity)
                temp_tokenizer.replace(tokenizer_path)
            except Exception:
                # Clean up temp file on error
                temp_tokenizer.unlink(missing_ok=True)
                raise
        
        self.tokenizer_path = str(directory)
        return directory

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "TokenizerManager":
        directory = Path(directory)
        config_path = directory / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata = TokenizerMetadata(**metadata_dict)
        manager = cls(
            tokenizer_type=metadata.tokenizer_type,
            vocab_size=metadata.vocab_size,
            _skip_validation=True,
        )
        manager.pad_token_id = metadata.pad_token_id
        manager.bos_token_id = metadata.bos_token_id
        manager.eos_token_id = metadata.eos_token_id
        manager.unk_token_id = metadata.unk_token_id
        manager.tokenizer_path = str(directory)

        if manager.tokenizer_type != "byte":
            if Tokenizer is None:
                raise ImportError(cls._TOKENIZERS_IMPORT_ERROR)
            tokenizer_file = directory / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
            manager.tokenizer = Tokenizer.from_file(str(tokenizer_file))
            manager._populate_special_token_ids()
            manager.vocab_size = manager.tokenizer.get_vocab_size()
            
            # Validate that loaded special tokens match metadata using helper method
            loaded_tokens = manager._get_special_token_map()
            metadata_tokens = {
                "PAD": metadata.pad_token_id,
                "BOS": metadata.bos_token_id,
                "EOS": metadata.eos_token_id,
                "UNK": metadata.unk_token_id,
            }
            
            token_mismatches = []
            for token_name in loaded_tokens.keys():
                if loaded_tokens[token_name] != metadata_tokens[token_name]:
                    token_mismatches.append(
                        f"{token_name}: metadata={metadata_tokens[token_name]}, "
                        f"loaded={loaded_tokens[token_name]}"
                    )
            
            if token_mismatches:
                raise ValueError(
                    f"Special token ID mismatch between metadata and loaded tokenizer:\n  " +
                    "\n  ".join(token_mismatches) +
                    "\nThe tokenizer file may be corrupted or from a different training run."
                )
            
            if manager.tokenizer_type == "bpe":
                from tokenizers import decoders
                if manager.tokenizer.decoder is None:
                    manager.tokenizer.decoder = decoders.ByteLevel()
            
            manager._configure_bpe_post_processor()
            
            if manager.vocab_size != metadata.vocab_size:
                raise ValueError(
                    f"Tokenizer vocab size mismatch: metadata says {metadata.vocab_size}, "
                    f"but loaded tokenizer has {manager.vocab_size}. "
                    f"The tokenizer file may be corrupted or from a different training run."
                )

        return manager

    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
        )

    def _get_special_token_map(self) -> Dict[str, Optional[int]]:
        """Return mapping of special token names to their IDs.
        
        Returns:
            Dictionary mapping token names (PAD, BOS, EOS, UNK) to token IDs
        """
        return {
            "PAD": self.pad_token_id,
            "BOS": self.bos_token_id,
            "EOS": self.eos_token_id,
            "UNK": self.unk_token_id,
        }

    def _populate_special_token_ids(self) -> None:
        if self.tokenizer_type == "byte":
            self.pad_token_id = None
            self.bos_token_id = None
            self.eos_token_id = None
            self.unk_token_id = None
            return

        assert self.tokenizer is not None
        
        # Populate all special token IDs from the tokenizer
        self.pad_token_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.bos_token_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(self.EOS_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
        
        # Check for missing tokens using the helper method
        special_map = self._get_special_token_map()
        missing_tokens = [name for name, token_id in special_map.items() if token_id is None]
        
        if missing_tokens:
            raise ValueError(
                f"Special tokens not found in vocabulary: {missing_tokens}. "
                f"This indicates the tokenizer was not trained correctly or the "
                f"vocab_size ({self.vocab_size}) is too small. "
                f"Minimum vocab_size should be at least {len(self.SPECIAL_TOKENS) + 100}."
            )

    def _configure_bpe_post_processor(self) -> None:
        """Configure BPE post-processor for special token handling."""
        if self.tokenizer_type != "bpe" or self.tokenizer is None:
            return
        
        from tokenizers import processors
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            pair=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN} $B:1 {self.EOS_TOKEN}:1",
            special_tokens=[
                (self.BOS_TOKEN, self.bos_token_id),
                (self.EOS_TOKEN, self.eos_token_id),
            ],
        )

    def __repr__(self) -> str:
        return (
            f"TokenizerManager(type={self.tokenizer_type}, "
            f"vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_token_id}, "
            f"bos_id={self.bos_token_id}, "
            f"eos_id={self.eos_token_id}, "
            f"unk_id={self.unk_token_id})"
        )
