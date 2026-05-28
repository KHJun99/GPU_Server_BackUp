# Sim Inspection Results

`inspect_sim.py` 실행 결과로 확보한 Phase 2 ground truth.
raw 출력: `inspection_raw.txt`, 전체 로그: `inspection_full.log`.

## Joint ordering — `robot.data.joint_names` (10 joints)

| idx | name | default | 비고 |
|----:|------|--------:|------|
|  0 | shoulder_pan   | +0.0000 | arm |
|  1 | shoulder_lift  | +0.0000 | arm |
|  2 | elbow_flex     | +0.0000 | arm |
|  3 | wrist_flex     | +1.5700 | arm |
|  4 | wrist_roll     | +0.0000 | arm |
|  5 | gripper        | +0.0000 | mimic placeholder (action 안 들어감) |
|  6 | left_proximal  | +0.0000 | gripper |
|  7 | right_proximal | +0.0000 | gripper |
|  8 | left_distal    | +0.0000 | gripper |
|  9 | right_distal   | +0.0000 | gripper |

→ obs `joint_pos_rel` (idx 0:10), `joint_vel` (idx 10:20) 모두 이 ordering.

## Action mapping

### arm_action (`JointPositionActionCfg`, scale=1.5, use_default_offset=True)
- joint_names: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']`
- joint_ids (in robot.joint_names): `[0, 1, 2, 3, 4]`
- target = `1.5 * action[0:5] + default_joint_pos[0:5]`
  - 단, default_arm = [0, 0, 0, 1.57, 0] → 실질적으로 wrist_flex 만 nonzero offset

### gripper_action (`BinaryJointPositionActionCfg`)
- joint_names: `['left_proximal', 'right_proximal', 'left_distal', 'right_distal']`
- joint_ids: `[6, 7, 8, 9]`  (※ idx 5 `gripper` 는 제외)
- open_command  (action[5] > 0): `[0.0, 0.0, 0.0, 0.0]`
- close_command (action[5] ≤ 0): `[-0.8, +0.8, +0.8, -0.8]`

| joint | open | close |
|---|---:|---:|
| left_proximal  |  0.0 | -0.8 |
| right_proximal |  0.0 | +0.8 |
| left_distal    |  0.0 | +0.8 |
| right_distal   |  0.0 | -0.8 |

## Frame

- `env_origins[0]`         = `[0, 0, 0]`
- `robot.root_pos_w[0]`    ≈ `[0, 0, 0]` (≈ 1e-8 floating noise)
- `cube.root_pos_w[0]`     = `[0.3251, 0.0769, 0.0500]`
- `robot_root - env_origin` ≈ `[0, 0, 0]`  → **env_origins == robot_root ✓**
- `cube - env_origin` = `[0.3251, 0.0769, 0.0500]`

검증: `policy_obs[20:23]` = `[0.3251, 0.0769, 0.0500]` 와 정확히 일치.
→ obs 의 `cube_pos` 는 **robot base frame** (env-local) 임이 확정.

## Dimensions

- `observation_manager.compute()['policy'].shape` = `(1, 32)` ✓
- `action_manager.total_action_dim` = `6` ✓

## obs 32 dim 실측 (joint==default, cube=spawn 직후)

```
idx  0:10 (joint_pos_rel) = [0,0,0,0,0,0,0,0,0,0]
idx 10:20 (joint_vel)      = [0,0,0,0,0,0,0,0,0,0]
idx 20:23 (cube_pos)       = [0.3251, 0.0769, 0.0500]
idx 23:26 (cube_vel)       = [0, 0, 0]
idx 26:32 (last_action)    = [0, 0, 0, 0, 0, 0]
```
→ task 문서의 obs layout 가설 (`pos_rel | vel | cube_pos | cube_vel | last_action`) 정확히 확정.

## Sim timing

- `cfg.sim.dt`         = `0.008333` s (120 Hz physics)
- `cfg.decimation`     = `2`
- `step_dt` (policy/obs rate) = `0.016667` s (60 Hz)
- `env.physics_dt`     = `0.008333`
- `env.step_dt`        = `0.016667`

→ obs_builder 의 `joint_vel` finite diff dt = **0.016667 s** (= 60 Hz).

## 부차적 발견

- 부팅 중 경고: *"Not all actuators are configured! Total number of actuated joints not equal to number of joints available: 9 != 10."*
  → `gripper` joint (idx 5) 가 actuator config 없는 mimic placeholder 라서 그런 듯. 학습/실행 모두 정상 동작 했었으므로 무해.
- 초기 실행에서 stdout buffer 미flush 로 marker 가 안 나오던 문제 → `flush=True` + `PYTHONUNBUFFERED=1` 로 해결.
