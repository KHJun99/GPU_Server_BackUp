from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str  # "cube" | "cylinder" | "mesh" | ...
    size: tuple[float, ...]
    mass: float
    friction: float
    asset_path: str | None = None
    extras: dict = field(default_factory=dict)


class ObjectRegistry:
    """Registry of manipulable objects available to the env.

    Phase 0: stores specs and allows sampling by name or weighted random.
    Phase 2+: backs object-curriculum and randomization.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ObjectSpec] = {}

    def register(self, spec: ObjectSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Object '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ObjectSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def __iter__(self) -> Iterator[ObjectSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
