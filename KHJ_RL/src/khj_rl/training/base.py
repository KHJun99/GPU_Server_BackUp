from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class EnvLike(Protocol):
    """Minimal env contract shared by all trainers (PPO, BC, Diffusion, ...)."""

    def reset(self) -> Any: ...
    def step(self, action: Any) -> Any: ...
    @property
    def observation_space(self) -> Any: ...
    @property
    def action_space(self) -> Any: ...


@dataclass
class TrainerConfig:
    run_name: str
    total_steps: int
    seed: int = 0
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    extras: dict = field(default_factory=dict)


class Trainer(ABC):
    """Algorithm-agnostic trainer interface.

    Same env contract across PPO / BC / Diffusion / Hybrid — letting Phase 4
    swap learners without touching environment, observation, or reward code.
    """

    def __init__(self, env: EnvLike, config: TrainerConfig) -> None:
        self.env = env
        self.config = config

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
