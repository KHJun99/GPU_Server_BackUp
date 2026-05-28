# Phase 2 Migration Plan — vec env + ManagerBasedRLEnv

**상태**: Phase 1 stage 3 (20cm cube spawn) 통과 후 시작. 사전 계획만.

**핵심 가치**:
- Single-env (현재) → vec env (256~512 envs 병렬)
- Wall-clock 시간 ~100배 단축 (demo collection 5h → 3분, PPO 7h → 5분)
- 하루 1회 실험 → 하루 10+회 실험 가능

**비목표**:
- Phase 1 학습 결과 *재설계*. Phase 2는 같은 정책을 다른 환경 인프라에서 *재현* + 확장.

---

## 1. 현재 vs Phase 2 구조 매핑

### 현재 (`envs/cube_lift/`)

```
envs/cube_lift/
  __init__.py
  cfg.py              # CubeLiftEnvCfg (모놀리식: robot, cube, goal, success, ...)
  env.py              # CubeLiftEnv (직접 SimulationContext + InteractiveScene 다룸)
  scene.py            # InteractiveSceneCfg (num_envs=1)
  articulation.py     # SO-ARM101 ActuatorCfg
  oracle.py           # MotionPlanningOracle
  success.py          # MultiConditionSuccess
  reward_dense.py     # A 옵션 dense shaping
```

특징:
- single-env (`num_envs=1`)
- env 코드가 sim 직접 다룸 (write_data_to_sim, step, update 매 호출 명시)
- obs/action/reward 함수가 env.py 안에 mixed

### Phase 2 (`envs/cube_lift_v2/`)

```
envs/cube_lift_v2/
  __init__.py
  cfg.py              # CubeLiftEnvCfg (slim — components 참조)
  scene.py            # InteractiveSceneCfg (num_envs=256+)
  articulation.py     # SO-ARM101 (그대로 재사용)
  oracle.py           # MotionPlanningOracle (vectorized batch version)
  mdp/
    observations.py   # ObservationTerm[] — 함수 별 분리
    rewards.py        # RewardTerm[]
    terminations.py   # DoneTerm[]
    events.py         # EventTerm[] (cube randomization, NoiseModel)
    success.py        # MultiConditionSuccess (vec)
```

특징:
- `ManagerBasedRLEnv` 상속 (Isaac Lab 표준)
- vec env: num_envs=256~512 동시 sim
- ObservationManager / RewardManager / TerminationManager가 자동 vectorize
- 각 MDP 요소(obs/reward/term/event)가 별 모듈로 분리, term 단위로 조합

---

## 2. 이행 단계 (incremental, 깨지지 않게)

### Step 1: 빈 v2 스캐폴드

새 디렉토리 `envs/cube_lift_v2/` 만들고 Phase 1 코드 *복사*. 일단 single-env 그대로 작동 확인 (regression test 가능).

### Step 2: cfg 분리 + ManagerBasedRLEnv 인터페이스

`CubeLiftEnvCfg`를 ObservationCfg / RewardCfg / TerminationCfg / EventCfg / SceneCfg로 분리:

```python
@configclass
class CubeLiftEnvCfg_v2(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = ...
    observations: ObservationsCfg = ...
    rewards: RewardsCfg = ...
    terminations: TerminationsCfg = ...
    events: EventsCfg = ...
    actions: ActionsCfg = ...
```

### Step 3: MDP 모듈화

각 함수가 `(env, env_ids) -> torch.Tensor (num_envs, ...)`. 예시:

```python
# mdp/observations.py
def joint_pos(env, env_ids):
    robot = env.scene["robot"]
    return robot.data.joint_pos[env_ids, env.cfg.arm_joint_idxs]

def cube_xyz_base(env, env_ids):
    cube = env.scene["cube"]
    robot = env.scene["robot"]
    cube_world = cube.data.root_pos_w[env_ids]
    base_world = robot.data.root_pos_w[env_ids]
    return cube_world - base_world

# ObservationsCfg
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_pose_base = ObsTerm(func=mdp.ee_pose_base)
        cube_xyz_base = ObsTerm(func=mdp.cube_xyz_base)
        target_delta_base = ObsTerm(func=mdp.target_delta_base)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        last_action = ObsTerm(func=mdp.last_action)
        normalized_t = ObsTerm(func=mdp.normalized_t)
    policy: PolicyCfg = PolicyCfg()
```

### Step 4: Reward / Termination / Event 모듈화

Sparse +1은 그대로. dense_reward는 별 RewardTerm으로:

```python
# mdp/rewards.py
def sparse_success(env, env_ids):
    return env.success_signal[env_ids]  # 5조건 평가 결과

def dense_ee_cube_dist(env, env_ids):
    d = torch.linalg.norm(env.ee_xyz[env_ids] - env.cube_xyz[env_ids], dim=-1)
    return -d  # weight는 RewardsCfg에서

@configclass
class RewardsCfg:
    sparse = RewTerm(func=mdp.sparse_success, weight=1.0)
    # Phase 1에서 dense ON 으로 학습한 정책이 검증되면:
    dense_ee_cube = RewTerm(
        func=mdp.dense_ee_cube_dist,
        weight=0.005,  # α_0 * w_ee_cube_dist
        # ManagerBasedRLEnv은 anneal 자체 지원 X — wrapping 필요
    )
```

### Step 5: vec env로 단순 확장

`num_envs=1` → `num_envs=256` 변경만. ManagerBasedRLEnv이 자동 batch.

### Step 6: PPO 재학습

같은 CleanRL 단일 파일 PPO (현재 `khj_rl.training.ppo`) — 차이는 env가 `step()` 호출 시 `(B, obs_dim)` 리턴. PPOTrainer가 batch 처리하게 변경.

```python
# 현재 PPOTrainer: 1 env, T steps
# Phase 2 PPOTrainer: B envs, T/B steps per env (rollout batch B*T)
```

### Step 7: BC 재학습 (선택)

BC는 demo 분포만 학습 → vec env 영향 X. demo만 더 모으면 됨. vec env로 collect는 100배 빨라짐.

### Step 8: Curriculum 자동화

Phase 1은 stage advance 수동. Phase 2는 ManagerBasedRLEnv의 EventTerm으로 cube spawn xy range가 학습 step에 따라 자동 확대:

```python
def cube_spawn_xy_curriculum(env, env_ids):
    progress = env.common_step_counter / env.cfg.curriculum.total_steps
    range_m = lerp(0.05, 0.20, progress)
    # randomize cube xy within [-range, range]
```

---

## 3. 이행 시 주의사항 (Phase 1 사고 누적분 반영)

| 사고 | Phase 1 (troubleshooting.md) | Phase 2 적용 |
|---|---|---|
| #1 PincOpen mimic | RECURRING | 자산 그대로 재사용, mimic stamp 그대로 |
| #2 fabric pin | RECURRING | 새 설치 환경마다 수동 패치 |
| #3 Pinocchio Assimp | RECURRING | entry-point script 모두 pinocchio import 먼저 |
| #4 scene.py robot 유실 | RESOLVED | regression test로 cover |
| #7 actions vs action | RESOLVED | demo schema 통일 유지 |
| #8 BC→PPO 후퇴 | IN-PROGRESS → 해결되어야 Phase 2 진입 | dense + curriculum 자동화로 더 안정적 학습 |
| #15 video/eval 불일치 | RESOLVED | vec env에서도 video_sampler는 single env 사용 |
| #17 Isaac Sim close hang | RECURRING | watcher 패턴 그대로 |
| #18 health zombie misclassification | RESOLVED | run-scoped scope 유지 |

---

## 4. 검증 전략

각 단계마다 *Phase 1 결과 재현* 확인:
1. Step 2 끝: cfg 분리 후 same single-env 학습이 Phase 1 success rate ±5%p 안에 들어옴
2. Step 5 끝: num_envs=4로 시작, 16, 64, 256 점진 확대
3. Step 6 끝: vec env PPO가 Phase 1 single-env PPO success rate ±5%p
4. 영상 sampling: 항상 env[0]만 — single-env와 동일 분포 검증 가능

---

## 5. 시간 추정

| Step | 추정 |
|---|---|
| 1. 스캐폴드 복사 | 30분 |
| 2. cfg 분리 + ManagerBasedRLEnv 인터페이스 | 4-8h |
| 3. MDP 모듈화 (obs/reward/term/event) | 6-12h |
| 4. EventTerm으로 randomization 이행 | 2-4h |
| 5. vec env 확장 + tuning | 4-8h |
| 6. PPO batch 처리 | 2-4h |
| 7-8. BC + curriculum 자동화 | 4-6h |
| **총** | **~30-50h** (1-2주 페이스) |

각 단계마다 regression test로 Phase 1 결과 재현 확인 → 안전 ↑.

---

## 6. Phase 2의 새 가능성 (Phase 1에서 못 한 것)

1. **Goal-space curriculum 자동화** — EventTerm으로 학습 진행에 따라 cube spawn 범위 확대
2. **Domain randomization 풀 스케일** — 마찰/질량/조명/카메라 noise 모두 vec env 환경별로 다르게
3. **Vec env 차원의 robustness** — 한 env 실패해도 다른 env가 학습. 분포 다양성 ↑
4. **하루 10+ 실험** — weight tuning matrix를 vec env로 빠르게 검증 → grid search 가능
5. **Sim2Real 분석 도구** — sim2real wrapper에 별도 env 띄워 분포 비교

---

## 7. Phase 2 → Phase 3 (Real robot) 연결

Phase 2 vec env 학습 완료 후:
- 같은 정책 weight를 real robot 추론에 사용
- Sim2Real gap 분석 (obs 분포 + action 분포 비교)
- Real robot 추가 학습은 별도 phase (이미 강한 sim 정책 + few-shot fine-tune)

→ `docs/real_robot_deployment.md` 참조

---

## 8. 미해결 / 후속 결정

- **Single-env Phase 1 codebase 유지 여부**: `envs/cube_lift/`를 archive로 두고 `envs/cube_lift_v2/`만 발전시킬지, 또는 v2가 단순 single-env mode도 지원하게 합칠지
- **PPO batch size**: vec env 256 + num_steps 128 → rollout 32768 transitions. minibatch 처리 GPU 메모리 검증 필요
- **BC를 ManagerBasedRLEnv 기반으로 갈 필요?** 보통 BC는 data loader라 env 무관. demo schema만 안 깨지면 OK
