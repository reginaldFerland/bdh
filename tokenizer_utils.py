"""Tokenizer utilities for the BDH project."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

try:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    Tokenizer = None  # type: ignore
    models = None  # type: ignore
    normalizers = None  # type: ignore
    pre_tokenizers = None  # type: ignore
    trainers = None  # type: ignore


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

    def __init__(self, tokenizer_type: str = "byte", vocab_size: int = 256):
        self.tokenizer_type = tokenizer_type.lower()
        self.requested_vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None
        self.vocab_size = 256 if self.tokenizer_type == "byte" else vocab_size
        self.pad_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.tokenizer_path: Optional[str] = None

        if self.tokenizer_type != "byte" and Tokenizer is None:
            raise ImportError(
                "The `tokenizers` library is required for non-byte tokenizers. "
                "Install it with `pip install tokenizers>=0.15.0`."
            )

        if self.tokenizer_type == "byte":
            self.vocab_size = 256

    # ------------------------------------------------------------------ Encoding
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

        assert self.tokenizer is not None, "Tokenizer has not been loaded or trained."
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

        assert self.tokenizer is not None, "Tokenizer has not been loaded or trained."
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ Training
    def train_tokenizer(
        self,
        texts: Iterable[str],
        output_dir: Union[str, Path],
        limit: Optional[int] = None,
    ) -> Path:
        if self.tokenizer_type == "byte":
            raise ValueError("Byte-level tokenizers do not require training.")
        assert Tokenizer is not None
        tokenizer, trainer = self._build_tokenizer_and_trainer()

        filtered_iter = (text for text in texts if isinstance(text, str) and text)
        if limit is not None:
            filtered_iter = itertools.islice(filtered_iter, limit)
            length = limit
        else:
            length = None

        tokenizer.train_from_iterator(filtered_iter, trainer=trainer, length=length)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self._populate_special_token_ids()

        save_path = self.save(output_dir)
        return save_path

    def _build_tokenizer_and_trainer(self):
        assert (
            Tokenizer is not None
            and models is not None
            and trainers is not None
            and pre_tokenizers is not None
        )
        tok_type = self.tokenizer_type

        if tok_type == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.requested_vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                show_progress=True,
            )
        elif tok_type == "wordpiece":
            assert normalizers is not None
            tokenizer = Tokenizer(models.WordPiece(unk_token=self.UNK_TOKEN))
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.requested_vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                show_progress=True,
            )
        elif tok_type == "unigram":
            assert normalizers is not None
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

    # -------------------------------------------------------------------- Saving
    def save(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata()
        with (directory / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(metadata.__dict__, f, indent=2, sort_keys=True)

        if self.tokenizer_type != "byte":
            assert self.tokenizer is not None
            self.tokenizer.save(str(directory / "tokenizer.json"))
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
        )
        manager.pad_token_id = metadata.pad_token_id
        manager.bos_token_id = metadata.bos_token_id
        manager.eos_token_id = metadata.eos_token_id
        manager.unk_token_id = metadata.unk_token_id
        manager.tokenizer_path = str(directory)

        if manager.tokenizer_type != "byte":
            assert Tokenizer is not None
            tokenizer_file = directory / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
            manager.tokenizer = Tokenizer.from_file(str(tokenizer_file))
            manager._populate_special_token_ids()
            manager.vocab_size = manager.tokenizer.get_vocab_size()

        return manager

    # ---------------------------------------------------------------- Utilities
    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
        )

    def _populate_special_token_ids(self) -> None:
        if self.tokenizer_type == "byte":
            # Byte-level variant does not reserve explicit special tokens.
            self.pad_token_id = None
            self.bos_token_id = None
            self.eos_token_id = None
            self.unk_token_id = None
            return

        assert self.tokenizer is not None
        self.pad_token_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.bos_token_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(self.EOS_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(self.UNK_TOKEN)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"TokenizerManager(type={self.tokenizer_type}, "
            f"vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_token_id}, "
            f"bos_id={self.bos_token_id}, "
            f"eos_id={self.eos_token_id}, "
            f"unk_id={self.unk_token_id})"
        )
