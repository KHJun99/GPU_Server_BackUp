from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RangeSpec:
    low: float
    high: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class RandomizationConfig:
    """Domain randomization scopes — applied at episode reset.

    Phase 1: actuator gains, mass, friction, baseline pose jitter.
    Later phases extend visual / sensor noise scopes here.
    """

    # Dynamics
    mass_scale: RangeSpec | None = None       # e.g. RangeSpec(0.8, 1.2)
    friction: RangeSpec | None = None
    actuator_gain_scale: RangeSpec | None = None

    # Initial state
    object_xy_jitter: RangeSpec | None = None  # meters
    object_yaw_jitter: RangeSpec | None = None  # radians

    # Hooks for later phases (visual, latency, etc.)
    extras: dict = field(default_factory=dict)

    def applies_at_reset(self) -> bool:
        return any(
            v is not None
            for v in (
                self.mass_scale,
                self.friction,
                self.actuator_gain_scale,
                self.object_xy_jitter,
                self.object_yaw_jitter,
            )
        )
