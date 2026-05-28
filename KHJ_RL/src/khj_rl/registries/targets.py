from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str  # "bin" | "pose" | "region" | ...
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None = None  # quaternion (w,x,y,z)
    tolerance: float = 0.02  # meters
    extras: dict = field(default_factory=dict)


class TargetRegistry:
    """Registry of placement targets (drop zones, target poses, regions).

    Phase 0: single target. Phase 3+: multi-bin and arbitrary target pose.
    """

    def __init__(self) -> None:
        self._specs: dict[str, TargetSpec] = {}

    def register(self, spec: TargetSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Target '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> TargetSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def __iter__(self) -> Iterator[TargetSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
