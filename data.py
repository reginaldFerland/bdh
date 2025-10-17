"""Utilities for loading datasets for the BDH training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

import numpy as np
import requests
import torch

from tokenizer_utils import TokenizerManager

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


class IterableDataset(Protocol):
    """Protocol for iterable dataset objects from HuggingFace datasets library."""
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset records."""
        ...


try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore


_DEFAULT_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class DatasetLoaderConfig:
    dataset_name: str = "shakespeare"
    dataset_config: Optional[str] = None
    streaming: bool = False
    text_column: Optional[str] = None
    tokenizer_manager: Optional[TokenizerManager] = None
    block_size: int = 512
    batch_size: int = 8
    train_split: float = 0.9
    data_dir: Optional[Path] = None
    device: Optional[torch.device] = None
    cache_dir: Optional[Path] = None
    seed: int = 42


class DatasetLoader:
    """Dataset loader supporting HuggingFace datasets and local files."""

    # Configuration constants
    MAX_CONSECUTIVE_EMPTY_RECORDS = 100
    MAX_TOTAL_STREAM_ATTEMPTS = 1000
    STREAM_BUFFER_SIZE_MULTIPLIER = 100
    DEFAULT_PRIORITY_COLUMNS = ("text", "content", "article", "body")

    def __init__(self, config: DatasetLoaderConfig):
        self.config = config
        self.data_dir = config.data_dir or Path(__file__).resolve().parent
        self.device = self._normalize_device(config.device)
        self.block_size = config.block_size
        self.train_split = max(0.0, min(config.train_split, 1.0))
        self.dataset_name = config.dataset_name.lower()
        self.dataset_config = config.dataset_config
        self.streaming = config.streaming
        self.text_column = config.text_column
        self.tokenizer = config.tokenizer_manager

        self._train_array: Optional[np.ndarray] = None
        self._val_array: Optional[np.ndarray] = None
        self._stream_iters: Dict[str, Iterator] = {}
        self._stream_buffers: Dict[str, List[int]] = {"train": [], "val": []}
        self._stream_builders: Dict[str, Callable[[], IterableDataset]] = {}

        self._separator_tokens = self._compute_separator_tokens()
        self._rng = np.random.default_rng(config.seed)
        
        # Validate configuration
        self._validate_config()

    # ------------------------------------------------------------------ Public API
    def load_dataset(self) -> None:
        if self.dataset_name in {"shakespeare", "tinyshakespeare"}:
            self._load_shakespeare()
            self._validate_loaded_data()
            self._log_dataset_info()
            return

        if load_dataset is None:
            raise ImportError(
                "datasets library is required for HuggingFace dataset support. "
                "Install it with `pip install datasets>=2.14.0`."
            )

        if self.streaming:
            self._load_hf_streaming()
            self._log_dataset_info()
        else:
            self._load_hf_in_memory()
            self._validate_loaded_data()
            self._log_dataset_info()

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")

        if self.streaming:
            batch = self._streaming_batch(split, batch_size)
        else:
            batch = self._in_memory_batch(split, batch_size)
        
        # Track batch generation for context manager statistics
        if hasattr(self, '_batches_generated'):
            self._batches_generated += 1
        
        return batch

    def __iter__(self):
        """Make the loader iterable for Pythonic usage.
        
        Allows usage like:
            for x_batch, y_batch in dataset_loader:
                # training code
        """
        self._iter_split = "train"
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch from the training split.
        
        Uses the default batch_size from config if available, otherwise defaults to 8.
        """
        batch_size = getattr(self.config, 'batch_size', 8)
        return self.get_batch(self._iter_split, batch_size)

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statements."""
        # Initialize tracking for context manager usage
        self._batches_generated = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper resource cleanup and error handling.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred  
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to not suppress exceptions
        """
        try:
            # Clear streaming resources to free memory
            self._stream_iters.clear()
            self._stream_buffers.clear()
            
            # Log statistics on successful completion
            if exc_type is None and hasattr(self, '_batches_generated'):
                if self._batches_generated > 0:
                    total_tokens = self._batches_generated * self.config.batch_size * self.block_size
                    print(f"Dataset loader closed. Generated {self._batches_generated} batches "
                          f"({total_tokens:,} tokens processed).")
        except Exception as cleanup_error:
            # Don't suppress the original exception if one exists
            if exc_type is None:
                raise cleanup_error
            # Otherwise, just log the cleanup error and continue with original exception
            print(f"Warning: Error during cleanup: {cleanup_error}")
        
        return False  # Don't suppress exceptions

    # ---------------------------------------------------------------- Shakespeare
    def _load_shakespeare(self) -> None:
        input_path = self.data_dir / "input.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.exists():
            response = requests.get(_DEFAULT_SHAKESPEARE_URL, timeout=30)
            response.raise_for_status()
            input_path.write_text(response.text, encoding="utf-8")

        text = input_path.read_text(encoding="utf-8")
        corpus_tokens = self._corpus_to_array([text])
        total = len(corpus_tokens)
        if total == 0:
            raise ValueError("Shakespeare dataset appears to be empty.")

        if total > self.block_size:
            split_idx = int(total * self.train_split)
            split_idx = max(split_idx, self.block_size + 1)
        else:
            split_idx = total
        split_idx = min(split_idx, total)

        self._train_array = corpus_tokens[:split_idx]
        val_slice = corpus_tokens[split_idx:]
        self._val_array = val_slice if len(val_slice) > self.block_size + 1 else None

    # ------------------------------------------------------------- HuggingFace IO
    def _load_hf_in_memory(self) -> None:
        dataset_dict: "DatasetDict" = load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=False,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )
        train_dataset, val_dataset = self._resolve_splits(dataset_dict)
        
        self._train_array = self._dataset_to_array(train_dataset)
        self._val_array = (
            self._dataset_to_array(val_dataset) if val_dataset is not None else None
        )

    def _load_hf_streaming(self) -> None:
        dataset_dict: "DatasetDict" = load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=True,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )
        available_splits = list(dataset_dict.keys())
        base_split = "train" if "train" in available_splits else available_splits[0]
        train_slice, val_slice = self._streaming_splits(base_split)

        self._stream_builders = {
            "train": self._build_stream_loader(train_slice),
        }
        if val_slice is not None:
            self._stream_builders["val"] = self._build_stream_loader(val_slice)
        else:
            self._stream_builders.pop("val", None)
        self._stream_iters.clear()
        self._stream_buffers = {"train": []}
        if "val" in self._stream_builders:
            self._stream_buffers["val"] = []

    # ----------------------------------------------------------- Batch generation
    def _in_memory_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._train_array if split == "train" else self._val_array
        if data is None or len(data) <= self.block_size:
            raise ValueError(
                f"No data available for split '{split}'. Did you call load_dataset() "
                "and ensure enough tokens for the configured block size?"
            )

        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset split '{split}' is too small for block_size {self.block_size}."
            )

        starts = self._rng.integers(0, max_start, size=batch_size)
        
        # Pre-allocate NumPy arrays for efficient batch construction
        x_np = np.empty((batch_size, self.block_size), dtype=np.int64)
        y_np = np.empty((batch_size, self.block_size), dtype=np.int64)
        
        for i, start in enumerate(starts):
            chunk = data[start : start + self.block_size + 1]
            x_np[i] = chunk[:-1]
            y_np[i] = chunk[1:]
        
        # Ensure contiguous arrays for optimal PyTorch performance
        x = torch.from_numpy(np.ascontiguousarray(x_np))
        y = torch.from_numpy(np.ascontiguousarray(y_np))
        
        return self._move_to_device(x, y)

    def _streaming_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        builder = self._stream_builders.get(split)
        if builder is None:
            raise ValueError(f"Streaming dataset for split '{split}' is not available.")

        buffer = self._stream_buffers.setdefault(split, [])
        tokens_needed = self.block_size + 1
        
        # Collect all samples first
        all_samples = []
        for _ in range(batch_size):
            sample_tokens = self._next_stream_sample(split, builder, buffer, tokens_needed)
            all_samples.append(sample_tokens)

        # Use NumPy for efficient batch construction, then convert to torch once
        x_np = np.array([s[:-1] for s in all_samples], dtype=np.int64)
        y_np = np.array([s[1:] for s in all_samples], dtype=np.int64)
        x_batch = torch.from_numpy(x_np)
        y_batch = torch.from_numpy(y_np)
        return self._move_to_device(x_batch, y_batch)

    def _next_stream_sample(
        self,
        split: str,
        builder: Callable[[], IterableDataset],
        buffer: List[int],
        required: int,
    ) -> List[int]:
        """Extract next sample from streaming dataset buffer.
        
        Fills the buffer from the streaming iterator until it has enough tokens,
        then extracts and returns the required number of tokens.
        
        Args:
            split: Dataset split name ('train' or 'val')
            builder: Factory function to create new iterator
            buffer: Token buffer to use
            required: Number of tokens needed
            
        Returns:
            List of token IDs for the sample
        """
        iterator = self._stream_iters.get(split)
        if iterator is None:
            iterator = iter(builder())
            self._stream_iters[split] = iterator

        self._fill_buffer_from_stream(split, iterator, builder, buffer, required)
        
        sample = buffer[:required]
        del buffer[:required]
        return sample
    
    def _fill_buffer_from_stream(
        self,
        split: str,
        iterator: Iterator[Dict[str, Any]],
        builder: Callable[[], IterableDataset],
        buffer: List[int],
        required: int,
    ) -> None:
        """Fill buffer from streaming iterator until it has required tokens.
        
        Args:
            split: Dataset split name ('train' or 'val')
            iterator: Current iterator over dataset records
            builder: Factory function to create new iterator on exhaustion
            buffer: Token buffer to fill
            required: Minimum number of tokens needed in buffer
        """
        consecutive_empty = 0
        total_attempts = 0

        while len(buffer) < required:
            # Absolute limit to prevent infinite loops on empty/corrupted datasets
            total_attempts += 1
            if total_attempts > self.MAX_TOTAL_STREAM_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to fill buffer after {self.MAX_TOTAL_STREAM_ATTEMPTS} attempts "
                    f"for split '{split}'. Dataset may be empty, corrupted, or all records "
                    "are invalid. Check your dataset configuration and text_column setting."
                )

            # Get next record, restart iterator if exhausted
            try:
                record = next(iterator)
            except StopIteration:
                iterator = iter(builder())
                self._stream_iters[split] = iterator
                consecutive_empty = 0  # Reset counter on iterator restart
                continue

            # Extract and validate text from record
            text = self._extract_text(record)
            if not text:
                consecutive_empty += 1
                if consecutive_empty >= self.MAX_CONSECUTIVE_EMPTY_RECORDS:
                    raise ValueError(
                        f"Found {self.MAX_CONSECUTIVE_EMPTY_RECORDS} consecutive empty records in "
                        f"streaming dataset split '{split}'. Data may be corrupted or the "
                        f"text_column setting ('{self.text_column}') may be incorrect."
                    )
                continue
            
            # Successfully got valid text, reset empty counter
            consecutive_empty = 0
            
            # Encode and add to buffer
            encoded = self._encode_text(text)
            if encoded:
                buffer.extend(encoded)
                if self._separator_tokens:
                    buffer.extend(self._separator_tokens)
            
            # Prevent buffer from growing too large (OOM protection)
            max_buffer_size = self.block_size * self.STREAM_BUFFER_SIZE_MULTIPLIER
            if len(buffer) > max_buffer_size:
                break

    # --------------------------------------------------------------- Util helpers
    def has_split(self, split: str) -> bool:
        if split == "train":
            has_array = self._train_array is not None and len(self._train_array) > self.block_size
            return has_array or ("train" in self._stream_builders)
        if split == "val":
            has_array = self._val_array is not None and len(self._val_array) > self.block_size
            return has_array or ("val" in self._stream_builders)
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")

    def _normalize_device(self, device: Optional[torch.device]) -> Optional[torch.device]:
        """Normalize device to torch.device or None.
        
        Args:
            device: Device specification (torch.device, string, or None)
            
        Returns:
            Normalized torch.device or None
        """
        if device is None:
            return None
        return device if isinstance(device, torch.device) else torch.device(device)

    def _validate_config(self) -> None:
        """Validate configuration parameters at initialization.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be at least 1, got {self.block_size}. "
                "Consider using a standard value like 512 or 1024."
            )
        
        if self.config.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.config.batch_size}. "
                "Typical values are 8, 16, 32, or 64."
            )
        
        if not (0.0 <= self.config.train_split <= 1.0):
            raise ValueError(
                f"train_split must be between 0.0 and 1.0, got {self.config.train_split}. "
                "Use values like 0.9 for 90% train / 10% validation split."
            )
        
        if self.config.seed < 0:
            raise ValueError(
                f"seed must be non-negative, got {self.config.seed}. "
                "Use a positive integer for reproducibility."
            )
        
        if not self.dataset_name:
            raise ValueError(
                "dataset_name cannot be empty. "
                "Use 'shakespeare' for the default dataset or specify a HuggingFace dataset."
            )

    def _validate_loaded_data(self) -> None:
        """Validate that loaded datasets have sufficient tokens for the configured block size."""
        min_size = self.block_size + 1
        
        if self._train_array is not None:
            train_size = len(self._train_array)
            if train_size < min_size:
                raise ValueError(
                    f"Training data has {train_size} tokens but requires at least "
                    f"{min_size} tokens (block_size={self.block_size} + 1). "
                    f"Consider reducing --block-size or using more training data."
                )
        
        if self._val_array is not None:
            val_size = len(self._val_array)
            if val_size < min_size:
                raise ValueError(
                    f"Validation data has {val_size} tokens but requires at least "
                    f"{min_size} tokens (block_size={self.block_size} + 1). "
                    f"Consider reducing --block-size or using more validation data."
                )

    def _log_dataset_info(self) -> None:
        """Log information about the loaded dataset for debugging."""
        mode = "streaming" if self.streaming else "in-memory"
        print(f"Loaded dataset '{self.dataset_name}' in {mode} mode")
        
        if self._train_array is not None:
            train_tokens = len(self._train_array)
            train_batches = train_tokens // self.block_size
            print(f"  Train: {train_tokens:,} tokens (~{train_batches:,} batches of size {self.block_size})")
        elif "train" in self._stream_builders:
            print(f"  Train: streaming mode (infinite batches of size {self.block_size})")
        
        if self._val_array is not None:
            val_tokens = len(self._val_array)
            val_batches = val_tokens // self.block_size
            print(f"  Val: {val_tokens:,} tokens (~{val_batches:,} batches of size {self.block_size})")
        elif "val" in self._stream_builders:
            print(f"  Val: streaming mode (infinite batches of size {self.block_size})")
        elif not self.has_split("val"):
            print("  Val: not available")

    def _move_to_device(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.device is None:
            return x, y
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def _resolve_splits(
        self,
        dataset_dict: "DatasetDict",
    ) -> Tuple["Dataset", Optional["Dataset"]]:
        """Resolve train and validation splits from a HuggingFace DatasetDict.
        
        Returns the training dataset and optional validation dataset.
        If no validation split exists and train_split < 1.0, automatically
        splits the training data.
        """
        train_ds = self._get_train_dataset(dataset_dict)
        
        # Check for existing validation split
        val_ds = None
        for split_name in ("validation", "val", "test"):
            if split_name in dataset_dict:
                val_ds = dataset_dict[split_name]
                break
        
        # Auto-split if no validation exists and train_split < 1.0
        if val_ds is None and self.train_split < 1.0:
            split = train_ds.train_test_split(
                test_size=1.0 - self.train_split,
                shuffle=True,
                seed=self.config.seed,
            )
            train_ds = split["train"]
            val_ds = split["test"]
        
        return train_ds, val_ds

    def _get_train_dataset(self, dataset_dict: "DatasetDict") -> "Dataset":
        """Extract the training dataset from a DatasetDict."""
        if "train" in dataset_dict:
            return dataset_dict["train"]
        # Fallback to first available split
        return dataset_dict[list(dataset_dict.keys())[0]]

    def _dataset_to_array(self, dataset: "Dataset") -> np.ndarray:
        texts = (self._extract_text(record) for record in dataset)
        return self._corpus_to_array(texts)

    def _streaming_splits(self, base_split: str) -> Tuple[str, Optional[str]]:
        if self.train_split >= 1.0:
            return base_split, None
        train_pct = int(self.train_split * 100)
        train_pct = min(max(train_pct, 1), 99)
        train_slice = f"{base_split}[:{train_pct}%]"
        val_slice = f"{base_split}[{train_pct}%:]"
        return train_slice, val_slice

    def _build_stream_loader(self, dataset_slice: str) -> Callable[[], IterableDataset]:
        """Build a factory function that creates streaming dataset iterators.
        
        Args:
            dataset_slice: Dataset slice specification (e.g., "train[:90%]")
            
        Returns:
            Factory function that returns an iterable dataset
        """
        if load_dataset is None:  # pragma: no cover - defensive guard
            raise ImportError(
                "datasets library is required for HuggingFace dataset support. "
                "Install it with `pip install datasets>=2.14.0`."
            )

        def _loader() -> IterableDataset:
            return load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=dataset_slice,
                streaming=True,
            )

        return _loader

    def _extract_text(self, record: Dict) -> str:
        """Extract text from a dataset record with priority column detection.
        
        If text_column is specified, use that. Otherwise, check priority columns
        first ('text', 'content', 'article', 'body'), then any string value.
        
        Args:
            record: Dataset record dictionary
            
        Returns:
            Extracted text string, or empty string if no valid text found
        """
        # Explicit column takes precedence
        if self.text_column and self.text_column in record:
            value = record[self.text_column]
            return value if isinstance(value, str) else ""
        
        # Try priority columns, then any string value
        candidates = [
            *[record.get(col) for col in self.DEFAULT_PRIORITY_COLUMNS],
            *record.values()
        ]
        
        return next((v for v in candidates if isinstance(v, str) and v), "")

    def _corpus_to_array(self, texts: Iterable[str]) -> np.ndarray:
        """Convert an iterable of texts into a single numpy array of token IDs.
        
        Uses numpy concatenation for better memory efficiency compared to 
        accumulating in Python lists.
        
        Args:
            texts: Iterable of text strings to encode
            
        Returns:
            NumPy array of token IDs (dtype=int64)
        """
        chunks: List[np.ndarray] = []
        for text in texts:
            if not text:
                continue
            encoded = self._encode_text(text)
            if not encoded:
                continue
            # Add separator tokens if configured
            if self._separator_tokens:
                encoded = encoded + self._separator_tokens
            chunks.append(np.array(encoded, dtype=np.int64))
        
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(chunks)

    def _encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs using configured tokenizer.
        
        Args:
            text: Text string to encode
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            return list(text.encode("utf-8"))
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _compute_separator_tokens(self) -> List[int]:
        """Compute separator tokens to use between documents.
        
        Returns:
            List of token IDs for newline separator, or [10] for byte-level fallback
        """
        newline_tokens = self._encode_text("\n")
        if newline_tokens:
            return newline_tokens
        return [10] if self.tokenizer is None else []
