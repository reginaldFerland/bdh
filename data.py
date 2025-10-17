"""Utilities for loading datasets for the BDH training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import requests
import torch

from tokenizer_utils import TokenizerManager

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore


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
    train_split: float = 0.9
    data_dir: Optional[Path] = None
    device: Optional[torch.device] = None
    cache_dir: Optional[Path] = None
    seed: int = 42
    show_progress: bool = True


class DatasetLoader:
    """Dataset loader supporting HuggingFace datasets and local files."""

    def __init__(self, config: DatasetLoaderConfig):
        self.config = config
        self.data_dir = config.data_dir or Path(__file__).resolve().parent
        self.device = config.device
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
        self._stream_builders: Dict[str, Callable[[], Iterable]] = {}

        self._separator_tokens = self._compute_separator_tokens()
        self._rng = np.random.default_rng()

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
            return self._streaming_batch(split, batch_size)
        return self._in_memory_batch(split, batch_size)

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
        dataset_dict: DatasetDict = load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=False,
        )
        train_dataset, val_dataset = self._resolve_splits(dataset_dict)
        self._train_array = self._dataset_to_array(train_dataset)
        self._val_array = (
            self._dataset_to_array(val_dataset) if val_dataset is not None else None
        )

    def _load_hf_streaming(self) -> None:
        dataset_dict: DatasetDict = load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=True,
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
        
        # Optimize tensor creation for CUDA by pre-allocating in pinned memory
        if self.device and self.device.type == "cuda":
            x = torch.empty((batch_size, self.block_size), dtype=torch.long, pin_memory=True)
            y = torch.empty((batch_size, self.block_size), dtype=torch.long, pin_memory=True)
            for i, start in enumerate(starts):
                chunk = data[start : start + self.block_size + 1]
                x[i] = torch.from_numpy(chunk[:-1])
                y[i] = torch.from_numpy(chunk[1:])
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        else:
            # Standard path for CPU or when device is not set
            x_list = []
            y_list = []
            for start in starts:
                chunk = data[start : start + self.block_size + 1]
                x_list.append(torch.from_numpy(chunk[:-1]))
                y_list.append(torch.from_numpy(chunk[1:]))
            
            x = torch.stack(x_list)
            y = torch.stack(y_list)
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

        # Convert to tensors in batches for better performance
        x_batch = torch.tensor([s[:-1] for s in all_samples], dtype=torch.long)
        y_batch = torch.tensor([s[1:] for s in all_samples], dtype=torch.long)
        return self._move_to_device(x_batch, y_batch)

    def _next_stream_sample(
        self,
        split: str,
        builder: Callable[[], Iterable],
        buffer: List[int],
        required: int,
    ) -> List[int]:
        iterator = self._stream_iters.get(split)
        if iterator is None:
            iterator = iter(builder())
            self._stream_iters[split] = iterator

        empty_records = 0
        max_empty = 1000  # Prevent infinite loops on bad data

        while len(buffer) < required:
            try:
                record = next(iterator)
            except StopIteration:
                iterator = iter(builder())
                self._stream_iters[split] = iterator
                continue

            text = self._extract_text(record)
            if not text:
                empty_records += 1
                if empty_records > max_empty:
                    raise ValueError(
                        f"Found {max_empty} consecutive empty records in "
                        f"streaming dataset split '{split}'. Data may be invalid."
                    )
                continue
            
            empty_records = 0  # Reset on valid record
            encoded = self._encode_text(text)
            if not encoded:
                continue
            buffer.extend(encoded)
            if self._separator_tokens:
                buffer.extend(self._separator_tokens)

        sample = buffer[:required]
        del buffer[:required]
        return sample

    # --------------------------------------------------------------- Util helpers
    def has_split(self, split: str) -> bool:
        if split == "train":
            has_array = self._train_array is not None and len(self._train_array) > self.block_size
            return has_array or ("train" in self._stream_builders)
        if split == "val":
            has_array = self._val_array is not None and len(self._val_array) > self.block_size
            return has_array or ("val" in self._stream_builders)
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")

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
        dataset_dict: DatasetDict,
    ) -> Tuple[Dataset, Optional[Dataset]]:
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

    def _get_train_dataset(self, dataset_dict: DatasetDict) -> Dataset:
        """Extract the training dataset from a DatasetDict."""
        if "train" in dataset_dict:
            return dataset_dict["train"]
        # Fallback to first available split
        return dataset_dict[list(dataset_dict.keys())[0]]

    def _dataset_to_array(self, dataset: Dataset) -> np.ndarray:
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

    def _build_stream_loader(self, dataset_slice: str) -> Callable[[], Iterable]:
        if load_dataset is None:  # pragma: no cover - defensive guard
            raise ImportError(
                "datasets library is required for HuggingFace dataset support. "
                "Install it with `pip install datasets>=2.14.0`."
            )

        def _loader() -> Iterable:
            return load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=dataset_slice,
                streaming=True,
            )

        return _loader

    def _extract_text(self, record: Dict) -> str:
        if self.text_column:
            value = record.get(self.text_column, "")
            return value if isinstance(value, str) else ""

        for key in ("text", "content", "article", "body"):
            value = record.get(key)
            if isinstance(value, str):
                return value

        for value in record.values():
            if isinstance(value, str):
                return value
        return ""

    def _corpus_to_array(self, texts: Iterable[str]) -> np.ndarray:
        """Convert an iterable of texts into a single numpy array of token IDs.
        
        Uses numpy concatenation for better memory efficiency compared to 
        accumulating in Python lists.
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
        if self.tokenizer is None:
            return list(text.encode("utf-8"))
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _compute_separator_tokens(self) -> List[int]:
        newline_tokens = self._encode_text("\n")
        if newline_tokens:
            return newline_tokens
        return [10] if self.tokenizer is None else []
