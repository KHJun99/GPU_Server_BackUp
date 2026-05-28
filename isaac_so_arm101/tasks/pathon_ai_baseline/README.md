---
language: en
license: mit
tags:
- reinforcement-learning
- robotics
- isaac-lab
- manipulation
- so-arm101
library_name: rsl-rl
---

# SO-ARM101 Cube Lifting Policy

This model is a reinforcement learning policy trained for the **SO-ARM101** robot arm to perform cube lifting tasks in Isaac Lab.

## Model Description

- **Task**: Lift a cube and bring it to a target position
- **Robot**: SO-ARM101 (6-DOF robotic arm)
- **Framework**: Isaac Lab 2.3.0 (on Isaac Sim 5.1.0)
- **Algorithm**: RSL-RL (Robotic Systems Lab - Reinforcement Learning)
- **Environment**: `Isaac-SO-ARM101-Lift-Cube-v0`
- **Training**: 1950 iterations with 4096 parallel environments

## Training Details

### Environment Configuration
- **Observation Space**: Joint positions, velocities, cube pose, target position
- **Action Space**: Joint position commands (6 DOF)
- **Reward Function**: Based on distance to target, cube height, and success criteria
- **Episode Length**: Variable (task-dependent)

### Training Parameters
- **Parallel Environments**: 4096
- **Total Iterations**: 1950
- **Training Time**: ~2 hours on NVIDIA RTX 4080 Super (16GB VRAM)
- **Framework**: Isaac Lab with RSL-RL runner
- **Simulator**: Isaac Sim 5.1.0

### Hardware Used
- GPU: NVIDIA RTX 4080 Super (16GB VRAM)
- OS: Ubuntu 24.04 LTS
- CUDA: 13.0

## Usage

### Prerequisites

```bash
# Install Isaac Lab (with Docker)
# See: https://isaac-sim.github.io/IsaacLab/

# Clone SO-ARM101 external project
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101
```

### Evaluation

```bash
# Inside Isaac Lab container
cd /workspace/isaaclab

# Run the trained policy
./isaaclab.sh -p /workspace/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
    --checkpoint /path/to/model_1950.pt
```

### Training From Scratch

```bash
# Train the policy
./isaaclab.sh -p /workspace/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Lift-Cube-v0 \
    --num_envs 4096 \
    --headless
```

## Performance

The trained policy successfully lifts and moves cubes to target positions with high success rate in simulation.

## Citation

If you use this model, please cite:

```bibtex
@misc{so-arm101-lift-isaaclab,
  title={SO-ARM101 Cube Lifting Policy trained with Isaac Lab},
  author={PathOn AI},
  year={2026},
  howpublished={\url{https://huggingface.co/}},
}

@software{isaaclab,
  author = {Mittal, Mayank and others},
  title = {Isaac Lab: A Unified Framework for Robot Learning},
  url = {https://isaac-sim.github.io/IsaacLab/},
  year = {2024},
}
```

## Related Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [SO-ARM101 Isaac Lab Integration](https://github.com/MuammerBay/isaac_so_arm101)
- [RSL-RL Library](https://github.com/leggedrobotics/rsl_rl)

## License

MIT License
