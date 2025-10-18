"""Tokenizer utilities for the BDH project."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import warnings

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer
else:
    HFTokenizer = Any

try:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None
    models = None
    normalizers = None
    pre_tokenizers = None
    trainers = None


DEFAULT_TEXT_COLUMNS = ("text", "content", "article", "body")
TokenizerType = Literal["byte", "bpe", "wordpiece", "unigram"]


def _is_valid_text(value: Any) -> bool:
    """Check if a value is a non-empty string.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a non-empty string, False otherwise
    """
    return isinstance(value, str) and bool(value)


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
    if text_column and text_column in record:
        value = record[text_column]
        return value if _is_valid_text(value) else ""
    
    for col in priority_columns:
        if col in record and _is_valid_text(record[col]):
            return record[col]
    
    for value in record.values():
        if _is_valid_text(value):
            return value
    
    return ""


def iter_texts_from_dataset(
    dataset: Iterable,
    text_column: Optional[str] = None,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> Iterator[str]:
    """Extract text from dataset records with optional progress tracking.
    
    This utility function extracts text from HuggingFace dataset records,
    filtering out empty records and optionally limiting the number of texts.
    
    Args:
        dataset: HuggingFace dataset (or any iterable of dict-like records) to iterate over
        text_column: Optional specific column to extract text from
        limit: Optional maximum number of texts to extract
        show_progress: Whether to show a progress bar (requires tqdm)
        
    Yields:
        Text strings from dataset records
        
    Examples:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        >>> texts = iter_texts_from_dataset(dataset, text_column="text", limit=1000)
        >>> for text in texts:
        ...     process(text)
    """
    count = 0
    
    if show_progress:
        try:
            from tqdm import tqdm
            desc = f"Extracting texts (limit={limit})" if limit else "Extracting texts"
            dataset_iter = tqdm(dataset, desc=desc, unit=" docs")
        except ImportError:
            dataset_iter = dataset
    else:
        dataset_iter = dataset
    
    for record in dataset_iter:
        text = extract_text_from_record(record, text_column)
        if not text:
            continue
        yield text
        count += 1
        if limit is not None and count >= limit:
            break


def _validate_training_iterator(
    texts: Iterable[str],
    validate: bool = True,
    max_peek: int = 100,
) -> Iterator[str]:
    """Optionally validate training data has at least one valid sample.
    
    When validation is enabled, checks that the first valid item is valid text.
    This approach doesn't duplicate data in memory and uses a single efficient loop.
    
    Args:
        texts: Iterable of text strings for training
        validate: Whether to perform validation (default: True)
        max_peek: Maximum number of items to check for first valid sample
        
    Yields:
        Text strings from the input iterator
        
    Raises:
        ValueError: If validation enabled and no valid text samples found
    """
    if not validate:
        yield from texts
        return
    
    found_valid = False
    items_checked = 0
    
    for item in texts:
        items_checked += 1
        
        if items_checked <= max_peek:
            if not found_valid and isinstance(item, str) and item:
                found_valid = True
        
        yield item
        
        if items_checked == max_peek and not found_valid:
            raise ValueError(
                f"No valid text samples found in first {max_peek} items. "
                "Ensure the input contains non-empty strings."
            )
    
    if not found_valid:
        raise ValueError(
            f"No valid text samples found in {items_checked} items. "
            "Ensure the input contains non-empty strings."
        )


@dataclass
class TokenizerMetadata:
    tokenizer_type: str
    vocab_size: int
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None


@dataclass
class TokenizerValidationConfig:
    """Configuration for tokenizer validation thresholds and limits."""
    
    max_vocab_size: int = 1_000_000
    min_training_samples: int = 100
    recommended_training_samples: int = 1_000
    production_training_samples: int = 10_000
    min_vocab_buffer: int = 100
    min_vocab_ratio: float = 0.8
    default_peek_limit: int = 1000


class TokenizerManager:
    """Lightweight wrapper around HuggingFace tokenizers with byte fallback."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    
    validation = TokenizerValidationConfig()
    
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

    def __init__(self, tokenizer_type: TokenizerType = "byte", vocab_size: int = 256):
        self.tokenizer_type = tokenizer_type.lower()
        self.requested_vocab_size = vocab_size
        self.tokenizer: Optional[HFTokenizer] = None
        self.vocab_size = 256 if self.tokenizer_type == "byte" else vocab_size
        self.pad_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.tokenizer_path: Optional[str] = None
        
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if vocab_size > self.validation.max_vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} is unusually large (>{self.validation.max_vocab_size:,} tokens). "
                "This may cause memory issues. If intentional, adjust this limit."
            )
        
        if self.tokenizer_type != "byte":
            min_vocab_size = len(self.SPECIAL_TOKENS) + self.validation.min_vocab_buffer
            if vocab_size < min_vocab_size:
                raise ValueError(
                    f"vocab_size {vocab_size} is too small for '{self.tokenizer_type}' tokenizer. "
                    f"Minimum is {min_vocab_size} ({len(self.SPECIAL_TOKENS)} special tokens + "
                    f"{self.validation.min_vocab_buffer} regular tokens). Consider using 'byte' tokenizer for small vocabularies."
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
                special_ids = self._special_token_ids
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
            if not add_special_tokens:
                return [list(text.encode("utf-8")) for text in texts]
            
            add_bos = self.bos_token_id is not None
            add_eos = self.eos_token_id is not None
            bos_id = self.bos_token_id
            eos_id = self.eos_token_id
            
            result = []
            for text in texts:
                encoded = list(text.encode("utf-8"))
                if add_bos and add_eos:
                    encoded = [bos_id] + encoded + [eos_id]
                elif add_bos:
                    encoded = [bos_id] + encoded
                elif add_eos:
                    encoded = encoded + [eos_id]
                result.append(encoded)
            return result

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
            if not skip_special_tokens:
                return [bytes(token_ids).decode("utf-8", errors="backslashreplace") 
                        for token_ids in token_id_lists]
            
            special_ids = self._special_token_ids
            if not special_ids:
                return [bytes(token_ids).decode("utf-8", errors="backslashreplace") 
                        for token_ids in token_id_lists]
            
            return [bytes([tid for tid in token_ids if tid not in special_ids])
                    .decode("utf-8", errors="backslashreplace") 
                    for token_ids in token_id_lists]

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
        
        validated_iter = _validate_training_iterator(
            texts, 
            validate=True,
            max_peek=self.validation.default_peek_limit
        )
        
        tokenizer, trainer = self._build_tokenizer_and_trainer()
        
        tokenizer.train_from_iterator(validated_iter, trainer=trainer)
        
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self._populate_special_token_ids()
        self._configure_bpe_post_processor()
        
        if self.vocab_size < self.requested_vocab_size * self.validation.min_vocab_ratio:
            vocab_deficit = self.requested_vocab_size - self.vocab_size
            warnings.warn(
                f"Trained vocab size ({self.vocab_size:,}) is significantly smaller than "
                f"requested ({self.requested_vocab_size:,}), missing {vocab_deficit:,} tokens "
                f"({(1 - self.vocab_size/self.requested_vocab_size)*100:.1f}% deficit). "
                f"This typically indicates insufficient or insufficiently diverse training data. "
                f"Consider providing more text samples.",
                UserWarning,
                stacklevel=2
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
        
        config_path = directory / "tokenizer_config.json"
        temp_config = directory / "tokenizer_config.json.tmp"
        
        metadata = self.metadata()
        try:
            with temp_config.open("w", encoding="utf-8") as f:
                json.dump(metadata.__dict__, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            
            temp_config.replace(config_path)
        except Exception:
            temp_config.unlink(missing_ok=True)
            raise

        if self.tokenizer_type != "byte":
            if self.tokenizer is None:
                raise RuntimeError(
                    self._TOKENIZER_NOT_TRAINED_ERROR.format(tokenizer_type=self.tokenizer_type)
                )
            tokenizer_path = directory / "tokenizer.json"
            temp_tokenizer = directory / "tokenizer.json.tmp"
            try:
                self.tokenizer.save(str(temp_tokenizer))
                with open(temp_tokenizer, 'rb') as f:
                    os.fsync(f.fileno())
                temp_tokenizer.replace(tokenizer_path)
            except Exception:
                temp_tokenizer.unlink(missing_ok=True)
                raise
        
        try:
            dir_fd = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
        
        self.tokenizer_path = str(directory)
        return directory

    @classmethod
    def _from_metadata(cls, metadata: TokenizerMetadata, directory: Path) -> "TokenizerManager":
        """Internal constructor for loading tokenizer from metadata.
        
        This bypasses the normal validation since we're loading a pre-trained tokenizer.
        
        Args:
            metadata: Tokenizer metadata loaded from disk
            directory: Directory containing the tokenizer files
            
        Returns:
            TokenizerManager instance with metadata populated
        """
        manager = object.__new__(cls)
        manager.tokenizer_type = metadata.tokenizer_type.lower()
        manager.requested_vocab_size = metadata.vocab_size
        manager.vocab_size = metadata.vocab_size
        manager.tokenizer = None
        manager.pad_token_id = metadata.pad_token_id
        manager.bos_token_id = metadata.bos_token_id
        manager.eos_token_id = metadata.eos_token_id
        manager.unk_token_id = metadata.unk_token_id
        manager.tokenizer_path = str(directory)
        return manager

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "TokenizerManager":
        directory = Path(directory)
        config_path = directory / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata = TokenizerMetadata(**metadata_dict)
        manager = cls._from_metadata(metadata, directory)

        if manager.tokenizer_type != "byte":
            if Tokenizer is None:
                raise ImportError(cls._TOKENIZERS_IMPORT_ERROR)
            tokenizer_file = directory / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
            manager.tokenizer = Tokenizer.from_file(str(tokenizer_file))
            manager._populate_special_token_ids()
            manager.vocab_size = manager.tokenizer.get_vocab_size()
            
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

    def _reset_cached_properties(self) -> None:
        """Clear cached properties after special tokens change."""
        if hasattr(self, '_special_token_ids'):
            delattr(self, '_special_token_ids')

    @cached_property
    def _special_token_ids(self) -> set:
        """Lazily compute and cache the set of special token IDs.
        
        Returns:
            Set of special token IDs (excluding None values)
        """
        return {
            sid
            for sid in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            )
            if sid is not None
        }

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
            self._reset_cached_properties()
            return

        assert self.tokenizer is not None
        
        self.pad_token_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.bos_token_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(self.EOS_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
        
        self._reset_cached_properties()
        
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
