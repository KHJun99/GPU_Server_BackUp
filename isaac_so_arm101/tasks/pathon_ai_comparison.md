# PathOn-AI Baseline 평가 보고서

**일자**: 2026-05-10
**대상**: `tasks/pathon_ai_baseline/model_{1950,2100}.pt` (HuggingFace, MIT, PathOn AI)
**우리 비교**: 세션 13~15 BC v8 / PPO v14 / PPO v15

---

## TL;DR

- **평가 진행 불가 — PathOn 환경이 fundamentally different**
- obs dim 차이 (PathOn 28 vs 우리 36)
- **action dim 차이 (PathOn 6 vs 우리 8)** — gripper action 부재 (6-DOF arm only)
- 동일 task name `Isaac-SO-ARM101-Lift-Cube-v0` 사용했으나 cfg 다름
- README 의 "high success rate" 주장은 우리와 다른 task 환경에서 측정
- **외부 baseline 비교 불가** → 우리 RL setting 결함 검증 못 함
- **권고**: Path C 유지 (sim2real with BC + Oracle)

---

## 1. 평가 시도 setting

- conda env `isaaclab` 활성화 (uv 우회)
- 신규 driver `tasks/diagnose_pathon.py` 작성 (diagnose_policy.py 기반 + last_action obs strip 시도)
- 시도 1: 우리 환경 그대로 → fail (obs 36 vs 28)
- 시도 2: last_action obs strip → fail (obs 30 vs 28)

## 2. PathOn checkpoint 분석

```python
# tasks/pathon_ai_baseline/model_1950.pt
std:                 (6,)        ← action_dim
actor.0.weight:      (256, 28)   ← obs_dim
actor.6.weight:      (6, 64)     ← action_dim 6
critic.0.weight:     (256, 28)
critic.6.weight:     (1, 64)     ← value scalar
```

**핵심 dim**:
- obs_dim: **28**
- action_dim: **6** (gripper 없는 6-DOF arm only)
- network: MLP 28 → 256 → 128 → 64 → 6

## 3. 우리 환경 cfg

```python
# src/isaac_so_arm101/tasks/lift/lift_env_cfg.py
ObservationsCfg.PolicyCfg:
  joint_pos = ObsTerm(func=mdp.joint_pos_rel)        # 10 dim (10 articulated)
  joint_vel = ObsTerm(func=mdp.joint_vel_rel)        # 10 dim
  object_position = ObsTerm(...)                      # 3 dim
  target_object_position = ObsTerm(...)               # 7 dim (pose+quat)
  actions = ObsTerm(func=mdp.last_action)             # 6 dim
                                              total = 36

ActionsCfg:
  arm_action = JointPositionActionCfg(joint_names=[shoulder/elbow/wrist])  # 6 dim
  gripper_action = BinaryJointPositionActionCfg(left_proximal, right_proximal)  # 1 dim binary, mirrored
                                              total = 7~8 dim
```

(우리 학습 ckpt 도 action 8 dim — 양손 gripper 별도 학습 또는 binary expand)

## 4. PathOn vs 우리 환경 차이

| 항목 | PathOn | 우리 | 차이 |
|---|---|---|---|
| obs dim | 28 | 36 | **8 dim** (last_action 6 + 2 dim 추가) |
| action dim | 6 | 7~8 | **gripper action 부재** |
| obs 추정 spec | joint_pos(9) + joint_vel(9) + obj_pos(3) + target(7) | + last_action(6) + 2 dim (10 articulated joint) | actuated joint 9 vs articulated 10 |
| gripper control | 없음 (auto/state machine?) | binary 양손 mirrored | 본질적 차이 |
| task 이름 | Isaac-SO-ARM101-Lift-Cube-v0 | 동일 | name 만 같음 |

## 5. 의미 있는 결론 (외부 baseline 비교 자체의 가치)

### A. PathOn "high success rate" 주장 해석
PathOn README 의 "successfully lifts and moves cubes ... with high success rate" 주장은:
- **gripper control 없는 환경**에서 측정
- 가설: PathOn 환경이 cube 를 arm 만으로 push/scoop 또는 gripper 가 항상 closed (state-based)
- 우리 task (gripper binary control 학습) 와 다른 문제

### B. 우리 setting 결함 식별 시도 결과
- PathOn 1950 iter 학습 결과 우리 환경에서 재현 불가 (cfg 다름)
- 우리 BC/PPO 의 14% saddle 이 환경 결함인지 학습 결함인지 **외부 비교로 식별 불가**
- 그러나 **task spec 자체가 다르다** 는 발견은 의미 있음

### C. RL 트랙 결정에의 영향
- PathOn 의 "성공" 사례가 우리 task 에 적용 가능한지 불명
- gripper-less 환경에서 lift 학습은 우리 양손 binary control 학습보다 단순 가능성 높음
- → 우리 RL 의 14% saddle 은 우리 task spec (gripper binary 학습) 자체 어려움일 가능성

## 6. 우리 진단 metric 비교 (PathOn 빼고)

| 정책 | succ% | ee<5cm% | close% | close_dist | cube_v |
|---|---|---|---|---|---|
| s13 Oracle | 15.0 | **86.0** | 98 | 0.05 | 0.02 |
| s13 BC v8 | 14.0 | 2.0 | 9 | 0.10 | 0.0003 |
| s13 PPO v10 best | 13.0 | 1.0 | 100 | 0.14 | 0.34 |
| **s14 iter200** | 14.0 | **31.0** | 100 | 0.21 | 0.16 |
| s15 iter100 | 15.0 | 21.0 | 100 | 0.18 | 0.54 |
| **PathOn 1950** | **N/A** | N/A | N/A | N/A | N/A |
| **PathOn 2100** | **N/A** | N/A | N/A | N/A | N/A |

## 7. 시나리오 분기 판정

| 시나리오 | 결과 | 권고 |
|---|---|---|
| ≥ 80% (우리 결함 확정) | N/A | — |
| 50~80% (그들 우세) | N/A | — |
| 20~50% (부분 차이) | N/A | — |
| < 20% (환경 한계) | N/A | — |
| **평가 불가 (cfg 다름)** | **현재** | **Path C 유지, sim2real 진입** |

## 8. 권고

### 다음 단계
1. **Path C 확정** 유지 — sim2real with BC actor (s8 v1) + Oracle 백업
2. **우리 cfg 자체** 가 학습 어렵게 만든 것일 수 있음 — gripper binary 학습 자체가 BC drift 강화 가능성
3. **추가 실험** (선택, 시간 있으면):
   - 우리 cfg 에서 gripper binary 빼고 6-DOF only 환경 학습 (PathOn 와 유사 setting)
   - BC drift 가 gripper 학습 부재 시 줄어드는지 검증

### RL 트랙 결정
- 외부 baseline 비교 실패가 **Path C 결정 번복할 근거 없음**
- 세션 13~15 의 saddle 진단 + 본 세션의 task spec 차이 발견을 종합 → **RL 트랙 종료, sim2real 진입**

## 9. Cleanup Whitelist 준수 ✓

- USD/oracle/collect/train_bc/validate_oracle/demos/BC actor/PPO ckpt 모두 unchanged
- cfg 수정 0 (lift_env_cfg.py 등 보존)
- 신규 파일: `tasks/diagnose_pathon.py` (diagnose_policy.py 기반, +5줄 obs strip), `tasks/pathon_ai_comparison.md`, `tasks/diagnose_pathon_ai_*.csv` (생성 실패)
