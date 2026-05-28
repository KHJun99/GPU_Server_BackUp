"""Per-episode demo buffer + npz writer for BC pretrain.

Trajectory format: one .npz file per episode under
``runs/demos/{stage}/{ep_id}.npz`` (``runs/`` is git-ignored). Keeping
records simple npz so the BC pretrain trainer can read them without a
heavier data-loading layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class EpisodeRecord:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    success_flag: bool = False

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.obs.append(np.asarray(obs))
        self.actions.append(np.asarray(action))
        self.rewards.append(float(reward))

    def finish(self, terminated: bool, success_flag: bool) -> None:
        self.terminated = terminated
        self.success_flag = success_flag

    def __len__(self) -> int:
        return len(self.obs)


def _to_jsonable(obj):
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def hash_cfg(cfg) -> str:
    """Short stable digest of an env_cfg dataclass tree. Captures the entire
    layout the demo was collected against, so BC pretrain can refuse demos
    from a different cfg.
    """
    payload = json.dumps(_to_jsonable(cfg), sort_keys=True, default=repr)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class DemoBuffer:
    """Write/read episode trajectories."""

    def __init__(self, root: Path | str = "runs/demos") -> None:
        self.root = Path(root)

    def stage_dir(self, stage: str) -> Path:
        p = self.root / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write(
        self,
        stage: str,
        ep_id: int,
        record: EpisodeRecord,
        cfg,
        seed: int,
    ) -> Path:
        path = self.stage_dir(stage) / f"{ep_id:06d}.npz"
        np.savez_compressed(
            path,
            obs=np.stack(record.obs).astype(np.float32),
            actions=np.stack(record.actions).astype(np.float32),
            rewards=np.asarray(record.rewards, dtype=np.float32),
            terminated=np.asarray(record.terminated, dtype=np.bool_),
            success_flag=np.asarray(record.success_flag, dtype=np.bool_),
            # Explicit unicode dtypes so `np.load(..., allow_pickle=False)`
            # always succeeds — `np.array(str)` can land on object dtype.
            cfg_hash=np.array(hash_cfg(cfg), dtype="<U16"),
            seed=np.asarray(seed, dtype=np.int64),
            stage=np.array(stage, dtype="<U64"),
            ep_id=np.asarray(ep_id, dtype=np.int64),
        )
        return path

    def read(self, path: Path | str) -> dict[str, np.ndarray]:
        return dict(np.load(path, allow_pickle=False))

    def iter_stage(self, stage: str) -> Iterator[Path]:
        for path in sorted(self.stage_dir(stage).iterdir()):
            if path.suffix == ".npz":
                yield path
