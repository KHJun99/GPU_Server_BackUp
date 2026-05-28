# KHJ_RL

Manipulation RL framework with four extension axes designed in from the start:

1. **Object** — `ObjectRegistry` (shape, size, mass, friction)
2. **Target** — `TargetRegistry` (single pose, multi-bin, arbitrary target)
3. **Learner** — pluggable `Trainer` (PPO / BC / Diffusion / Hybrid) on a shared env interface
4. **Sim2Real** — `RandomizationConfig` + `NoiseModel` baked into the environment from Phase 1

## Phasing

- **Phase 0 (this commit)** — extension-point interfaces only; one stub each
- **Phase 1** — single object × single target × PPO × sim2real (noise + randomization)
- **Phase 2** — object variety + curriculum
- **Phase 3** — target variety (multi-bin, arbitrary pose)
- **Phase 4** — algorithm swappability (BC / Diffusion / Hybrid)

## Guardrails (from prior reward-hacking incident)

- Video auto-sampling inside the training loop (`eval/video_sampler.py`)
- Observation alignment check between env baseline and policy expectation (`eval/obs_alignment.py`)
- No hardcoded environment constants in sim2real wrappers — single source of truth via `env_cfg`

## Layout

```
src/khj_rl/
  envs/        # env builder
  registries/  # object & target registries
  sim2real/    # randomization + noise model
  training/    # algorithm-agnostic trainer interface
  eval/        # video sampling + obs alignment checks
```
