"""Checkpoint management utilities for BDH training."""

from __future__ import annotations

import os
import re
import shutil
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

        # Use symlinks instead of copying to avoid redundant file I/O
        latest_path = self.checkpoint_dir / "checkpoint-latest.pt"
        self._update_symlink(step_path, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint-best.pt"
            self._update_symlink(step_path, best_path)

        self._cleanup_old_checkpoints()
        return step_path

    def load_checkpoint(self, checkpoint_path: Optional[str | Path] = None) -> Optional[CheckpointState]:
        """Load checkpoint from a specific path or the most recent one."""
        path = Path(checkpoint_path) if checkpoint_path else self.get_latest_checkpoint()
        if path is None or not path.exists():
            return None
        try:
            state = torch.load(path, map_location="cpu")
            self._validate_checkpoint_state(state, path)
            return CheckpointState(state=state, path=path)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
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
    def _validate_checkpoint_state(self, state: Dict, path: Path) -> None:
        """Validate that checkpoint contains required fields."""
        required_fields = ["model_state_dict", "step"]
        missing = [field for field in required_fields if field not in state]
        if missing:
            raise ValueError(
                f"Checkpoint at {path} is missing required fields: {missing}. "
                "This checkpoint may be corrupted or from an incompatible version."
            )
        
        # Validate model_state_dict is not empty
        if not state["model_state_dict"]:
            raise ValueError(
                f"Checkpoint at {path} has empty model_state_dict. "
                "This checkpoint is corrupted."
            )

    def _check_disk_space(self, estimated_size: int) -> None:
        """Verify sufficient disk space before saving checkpoint."""
        stat = shutil.disk_usage(self.checkpoint_dir)
        # Require 2x the estimated size for safety (temp + final file)
        required = estimated_size * 2
        if stat.free < required:
            raise OSError(
                f"Insufficient disk space: {stat.free / 1e9:.2f}GB available, "
                f"need ~{required / 1e9:.2f}GB for checkpoint save"
            )

    def _safe_save(self, state: Dict, path: Path) -> None:
        """Atomically save checkpoint with disk space check."""
        # Estimate checkpoint size (rough approximation)
        # Actual size will vary, but this gives us a safety check
        estimated_size = sum(
            param.nelement() * param.element_size()
            for param in state.get("model_state_dict", {}).values()
            if hasattr(param, "nelement")
        )
        # Add overhead for optimizer state, metadata, etc. (roughly 3x model size)
        estimated_size = max(estimated_size * 3, 100 * 1024 * 1024)  # Minimum 100MB
        
        self._check_disk_space(estimated_size)
        
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)

    def _update_symlink(self, target: Path, link: Path) -> None:
        """Update symlink to point to target atomically, with fallback for systems without symlink support."""
        try:
            # Use atomic replacement: create temp symlink, then rename
            # This avoids race condition where link doesn't exist between unlink and symlink_to
            tmp_link = link.with_suffix(".tmp_link")
            tmp_link.unlink(missing_ok=True)
            
            # Use relative symlink to avoid breaking if directory is moved
            tmp_link.symlink_to(target.name)
            
            # Atomic replacement - if link exists, this replaces it atomically
            tmp_link.replace(link)
        except (OSError, NotImplementedError):
            # Fallback for systems without symlink support (e.g., Windows without admin)
            import shutil
            tmp_copy = link.with_suffix(".tmp_copy")
            shutil.copy2(target, tmp_copy)
            tmp_copy.replace(link)

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
