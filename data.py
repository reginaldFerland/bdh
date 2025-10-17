"""Utilities for loading datasets for the BDH training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - datasets is optional until needed
    load_dataset = None  # type: ignore
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore

import requests


_DEFAULT_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class DatasetLoaderConfig:
    dataset_name: str = "shakespeare"
    dataset_config: Optional[str] = None
    streaming: bool = False
    text_column: Optional[str] = None
    tokenizer: Optional[object] = None  # future extension
    block_size: int = 512
    train_split: float = 0.9
    data_dir: Optional[Path] = None
    device: Optional[torch.device] = None


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
        self.tokenizer = config.tokenizer

        self._train_array: Optional[np.ndarray] = None
        self._val_array: Optional[np.ndarray] = None
        self._train_stream: Optional[Iterable] = None
        self._val_stream: Optional[Iterable] = None
        self._stream_iters: Dict[str, Iterator] = {}
        self._stream_buffers: Dict[str, bytearray] = {"train": bytearray(), "val": bytearray()}

        self._rng = np.random.default_rng()

    # Public API -----------------------------------------------------------------
    def load_dataset(self) -> None:
        if self.dataset_name in {"shakespeare", "tinyshakespeare"}:
            self._load_shakespeare()
            return

        if load_dataset is None:
            raise ImportError(
                "datasets library is required for HuggingFace dataset support. "
                "Install it with `pip install datasets>=2.14.0`."
            )

        if self.streaming:
            self._load_hf_streaming()
        else:
            self._load_hf_in_memory()

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")

        if self.streaming:
            return self._streaming_batch(split, batch_size)
        return self._in_memory_batch(split, batch_size)

    # Shakespeare ----------------------------------------------------------------
    def _load_shakespeare(self) -> None:
        input_path = self.data_dir / "input.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.exists():
            response = requests.get(_DEFAULT_SHAKESPEARE_URL, timeout=30)
            response.raise_for_status()
            input_path.write_text(response.text, encoding="utf-8")

        memmap = np.memmap(input_path, dtype=np.uint8, mode="r")
        split_idx = int(len(memmap) * self.train_split)
        split_idx = max(split_idx, self.block_size + 1) if len(memmap) > self.block_size else len(memmap)
        self._train_array = np.asarray(memmap[:split_idx], dtype=np.uint8)
        self._val_array = np.asarray(memmap[split_idx:], dtype=np.uint8) if split_idx < len(memmap) else None

    # HuggingFace helpers --------------------------------------------------------
    def _load_hf_in_memory(self) -> None:
        dataset_dict: DatasetDict = load_dataset(
            self.dataset_name, self.dataset_config, streaming=False
        )
        train_dataset, val_dataset = self._resolve_splits(dataset_dict)
        self._train_array = self._dataset_to_array(train_dataset)
        self._val_array = self._dataset_to_array(val_dataset) if val_dataset is not None else None

    def _load_hf_streaming(self) -> None:
        base_split = "train"
        # We request the dataset dict to discover available splits
        dataset_dict: DatasetDict = load_dataset(
            self.dataset_name, self.dataset_config, streaming=True
        )
        available_splits = list(dataset_dict.keys())
        if base_split not in available_splits and available_splits:
            base_split = available_splits[0]

        train_slice, val_slice = self._streaming_splits(base_split)
        self._train_stream = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=train_slice,
            streaming=True,
        )
        self._val_stream = (
            load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=val_slice,
                streaming=True,
            )
            if val_slice is not None
            else None
        )

    # In-memory batching ---------------------------------------------------------
    def _in_memory_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._train_array if split == "train" else self._val_array
        if data is None or len(data) <= self.block_size:
            raise ValueError(
                f"No data available for split '{split}'. Did you call load_dataset() and "
                "choose a dataset with enough tokens?"
            )

        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset split '{split}' is too small for block_size {self.block_size}."
            )

        starts = self._rng.integers(0, max_start, size=batch_size)
        x_list = []
        y_list = []
        for start in starts:
            chunk = data[start : start + self.block_size + 1]
            x_list.append(torch.from_numpy(chunk[:-1].astype(np.int64)))
            y_list.append(torch.from_numpy(chunk[1:].astype(np.int64)))
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return self._move_to_device(x, y)

    # Streaming batching ---------------------------------------------------------
    def _streaming_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stream = self._train_stream if split == "train" else self._val_stream
        if stream is None:
            raise ValueError(f"Streaming dataset for split '{split}' is not available.")
        buffer = self._stream_buffers.setdefault(split, bytearray())
        samples = []

        tokens_needed = self.block_size + 1
        for _ in range(batch_size):
            sample_bytes = self._next_stream_sample(split, stream, buffer, tokens_needed)
            x = torch.tensor(sample_bytes[:-1], dtype=torch.long)
            y = torch.tensor(sample_bytes[1:], dtype=torch.long)
            samples.append((x, y))

        x_batch = torch.stack([s[0] for s in samples])
        y_batch = torch.stack([s[1] for s in samples])
        return self._move_to_device(x_batch, y_batch)

    def _next_stream_sample(
        self, split: str, stream: Iterable, buffer: bytearray, required: int
    ) -> bytearray:
        iterator = self._stream_iters.get(split)
        if iterator is None:
            iterator = iter(stream)
            self._stream_iters[split] = iterator

        while len(buffer) < required:
            try:
                record = next(iterator)
            except StopIteration:
                iterator = iter(stream)
                self._stream_iters[split] = iterator
                continue
            text = self._extract_text(record)
            if not text:
                continue
            buffer.extend(text.encode("utf-8"))
            buffer.append(10)  # newline separator

        sample = bytearray(buffer[:required])
        del buffer[:required]
        return sample

    # Utility methods ------------------------------------------------------------
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
        self, dataset_dict: DatasetDict
    ) -> Tuple[Dataset, Optional[Dataset]]:
        if "train" in dataset_dict:
            train_dataset = dataset_dict["train"]
            val_dataset = dataset_dict.get("validation") or dataset_dict.get("val")
            if val_dataset is None:
                val_dataset = dataset_dict.get("test")
            if val_dataset is None and self.train_split < 1.0:
                split = train_dataset.train_test_split(
                    test_size=1.0 - self.train_split, shuffle=True, seed=42
                )
                train_dataset = split["train"]
                val_dataset = split["test"]
            return train_dataset, val_dataset

        # Fallback: first split becomes train, optional second becomes val
        splits = list(dataset_dict.keys())
        train_dataset = dataset_dict[splits[0]]
        val_dataset = dataset_dict[splits[1]] if len(splits) > 1 else None
        if val_dataset is None and self.train_split < 1.0:
            split = train_dataset.train_test_split(
                test_size=1.0 - self.train_split, shuffle=True, seed=42
            )
            return split["train"], split["test"]
        return train_dataset, val_dataset

    def _dataset_to_array(self, dataset: Dataset) -> np.ndarray:
        buffer = bytearray()
        for record in dataset:
            text = self._extract_text(record)
            if not text:
                continue
            buffer.extend(text.encode("utf-8"))
            buffer.append(10)  # newline separator
        if not buffer:
            return np.zeros(0, dtype=np.uint8)
        return np.frombuffer(bytes(buffer), dtype=np.uint8)

    def _streaming_splits(self, base_split: str) -> Tuple[str, Optional[str]]:
        if self.train_split >= 1.0:
            return base_split, None
        train_pct = int(self.train_split * 100)
        train_pct = min(max(train_pct, 1), 99)
        train_slice = f"{base_split}[:{train_pct}%]"
        val_slice = f"{base_split}[{train_pct}%:]"
        return train_slice, val_slice

    def _extract_text(self, record: Dict) -> str:
        if self.text_column:
            value = record.get(self.text_column, "")
            return value if isinstance(value, str) else ""

        # Try common column names
        for key in ("text", "content", "article", "body"):
            value = record.get(key)
            if isinstance(value, str):
                return value

        # Fallback: first string field
        for value in record.values():
            if isinstance(value, str):
                return value
        return ""
