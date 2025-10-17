"""Checkpoint management utilities for BDH training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)\.pt$")


@dataclass
class CheckpointState:
    state: Dict
    path: Path


class CheckpointManager:
    """Handles saving, loading, and pruning training checkpoints."""

    def __init__(self, checkpoint_dir: Path | str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = max(1, keep_last_n)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- Public
    def save_checkpoint(self, state: Dict, step: int, is_best: bool = False) -> Path:
        """Persist checkpoint for the given step and manage bookkeeping."""
        step_path = self.checkpoint_dir / f"checkpoint-{step:07d}.pt"
        self._safe_save(state, step_path)

        latest_path = self.checkpoint_dir / "checkpoint-latest.pt"
        self._safe_save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint-best.pt"
            self._safe_save(state, best_path)

        self._cleanup_old_checkpoints()
        return step_path

    def load_checkpoint(self, checkpoint_path: Optional[str | Path] = None) -> Optional[CheckpointState]:
        """Load checkpoint from a specific path or the most recent one."""
        path = Path(checkpoint_path) if checkpoint_path else self.get_latest_checkpoint()
        if path is None or not path.exists():
            return None
        try:
            state = torch.load(path, map_location="cpu")
            return CheckpointState(state=state, path=path)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"Warning: failed to load checkpoint at {path}: {exc}")
            return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Return the path to the latest checkpoint if present."""
        numbered = self._list_numbered_checkpoints()
        if numbered:
            return numbered[-1]
        latest = self.checkpoint_dir / "checkpoint-latest.pt"
        return latest if latest.exists() else None

    # -------------------------------------------------------------------- Helpers
    def _safe_save(self, state: Dict, path: Path) -> None:
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = self._list_numbered_checkpoints()
        excess = len(checkpoints) - self.keep_last_n
        for _ in range(max(0, excess)):
            oldest = checkpoints.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Warning: failed to delete old checkpoint {oldest}: {exc}")

    def _list_numbered_checkpoints(self) -> list[Path]:
        numbered = []
        for path in self.checkpoint_dir.glob("checkpoint-*.pt"):
            match = _CHECKPOINT_PATTERN.match(path.name)
            if match:
                numbered.append(path)
        numbered.sort(key=lambda p: int(_CHECKPOINT_PATTERN.match(p.name).group(1)))  # type: ignore
        return numbered
