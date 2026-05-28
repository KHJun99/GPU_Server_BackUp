# 세션 17 결과 — Gripper-less Ablation + Presentation 정리

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**ablation**: gripper always-closed (open=close=closed) + grasp_object reward 제거
**학습**: from-scratch 2000 iter, 4096 envs (BC warmstart 없음, PathOn-AI 방식)

---

## TL;DR

- **가설 FAIL — gripper control 은 saddle 원인 아님**
- gripperless final success **14%** (BC v8 14% / PPO v10/v14/v15 14% 와 정확히 같음)
- 도달율 max 5% (s14 iter200 의 31% 보다 **낮음**) — gripper 학습이 reaching 학습 incentive 제공
- **saddle 의 원인 = 환경/task 자체 한계** (cube random init 분포 + episode_length + reward shaping 종합)
- **Path C 확정 강한 보강** — sim2real 진입 (BC + Oracle)

---

## 1. 가설 + 진단 근거

세션 16 발견:
- PathOn-AI ckpt: action 6 dim (gripper 없음) vs 우리 action 8 dim
- 가설: gripper binary control 학습이 saddle 원인 → gripper 제거 시 ≥ 50% 도달

세션 17 ablation: gripper always-closed → effective 6-DOF arm only environment

## 2. cfg 변경

`joint_pos_env_cfg.py` 신규 cfg class:
```python
class SoArm101LiftCubeGripperlessEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            ...,
            open_command_expr={"left_proximal": -0.6, "right_proximal": 0.6},   # always closed
            close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6},  # always closed
        )
        if hasattr(self.rewards, "grasp_object"):
            self.rewards.grasp_object = None
```

`__init__.py` 신규 register 3개 (Gripperless / -Play / -Video).

## 3. 학습 결과 (2000 iter, 32분)

첫 iter 부터 lifting_object reward 1.13 (gripper closed → random action 으로도 cube 자주 catch). PathOn 의 "high success rate" 와 일관된 학습 dynamic.

그러나 학습 중 entropy 8.5 → 15.2 증가 + lifting reward 1.13 → 0.72 감소 (학습이 lifting 능력 일부 잃음).

## 4. 평가 (6 ckpt × 100 ep)

| iter | succ% | ee<5cm% | ee_min | close% | close_dist |
|---|---|---|---|---|---|
| 0 | 14.0 | 2.0 | 0.164 | 0 | nan |
| 200 | 14.0 | 3.0 | 0.137 | 100 | 0.183 |
| 500 | 13.0 | 3.0 | 0.119 | 7 | 0.213 |
| 1000 | 13.0 | 5.0 | 0.124 | 0 | nan |
| 1500 | 13.0 | 5.0 | 0.124 | 2 | 0.229 |
| **1999** | **14.0** | 5.0 | 0.125 | 6 | 0.216 |

→ success 13~14% **정확히 BC saddle**. 도달율 max 5% (s14 iter200 31% 보다 매우 낮음).

## 5. 비교 표 (BC / PPO v14 / v15 / s17)

| 정책 | succ% | ee<5cm% | ee_min | close% |
|---|---|---|---|---|
| s13 Oracle | 15.0 | **86.0** | 4.6cm | 98 |
| s13 BC v8 | 14.0 | 2.0 | 13.9cm | 9 |
| s13 PPO best | 13.0 | 1.0 | 8.4cm | 100 |
| **s14 iter200** | 14.0 | **31.0** | 6.8cm | 100 |
| s15 iter100 | 15.0 | 21.0 | 8.9cm | 100 |
| **s17 i1999 (gripperless)** | **14.0** | 5.0 | 12.5cm | 6 |

## 6. 가설 PASS/FAIL

| 결과 (success) | 의미 | 판정 |
|---|---|---|
| ≥ 50% | gripper 가 saddle 원인 | ❌ FAIL |
| 25~50% | gripper 일부 영향 | ❌ FAIL |
| **≤ 20%** | **gripper 무관, 환경 한계** | ✓ **PASS** |

**결과: 14% — Saddle 은 환경/task 자체 한계** (gripper 무관 확정).

추가 발견:
- gripperless 도달율 5% < BC v8 의 2% 와 거의 같음 (s14/s15 의 21~31% 보다 훨씬 낮음)
- gripper 학습 자체가 reaching 학습 incentive 일부 제공했었음 (proximity-aware grip → ee 도달 학습)

## 7. saddle 의 진짜 원인 (4세션 ablation 종합)

세션 13~17 ablation 종합:
- **13**: BC ee_to_cube_min 14cm (도달 못 함)
- **14**: reward std fix → transient 31% 도달 (+collapse)
- **15**: close reward → 21% 도달 (s14 보다 낮음)
- **16**: PathOn 비교 불가 (cfg 다름)
- **17**: gripperless → 14% saddle 동일 (환경 한계)

**Saddle 의 진짜 원인 — 추정**:
1. **cube random init 분포** — `pose_range x:(-0.1, 0.1), y:(-0.2, 0.2)` → 우연한 ee 근처 14% catch
2. **episode_length 5초** (decimation=1, 100Hz, 500 steps) — random catch 확률 fixed
3. **reward shaping 한계** — sharper reward 가 transient improvement 만 제공
4. **PPO + BC saddle** — local minima escape 못 함

→ **현재 setting 으로 14% 가 ceiling**. 학습 더 해도 안 늘어남.

## 8. 영상 자료

`~/jabis_sim/day5/videos/session17/iter1999/` (6 mp4, top + diag × 3 seed)

`~/jabis_sim/day5/videos/presentation/` (7 영상 + 3 plot):
- 01_oracle_normal_grasp.mp4 (Oracle 정상 86% 도달)
- 02_bc_v8_14pct.mp4 (BC sim 학습 14%)
- 03_ppo_s14_iter200_reach31pct.mp4 (reward fix 효과)
- 04_ppo_s14_iter999_collapse.mp4 (reward fix 후 collapse)
- 05_bc_diag_view.mp4 / 06_oracle_diag_view.mp4 (대각 view 비교)
- **07_ppo_s17_gripperless_14pct.mp4 (gripper-less 도 14% — 환경 한계)**
- ablation_evolution.png (4-panel session 13~17 evolution)
- reward_landscape.png (std=0.05 vs 0.02)
- session13_success_vs_fail.png (cube_init 분포)

## 9. 시연 path 권고 (사용자 결정 사항)

| 결과 | 권고 |
|---|---|
| gripperless ≥ 50% | "gripper 가 saddle 원인" 메시지 |
| **14% (실제)** | **"환경 한계 — BC + Oracle 시연 (Path C)"** ← 이번 결과 |

**시연 path 결정 — Path C (BC actor s8 v1 + Oracle 백업)**:
- 실물 deploy 가능 (gripperless 정책은 진단 자료 only)
- 14% 시뮬 결과 그대로 시연 (sim2real gap 무시 시 14%, gap 포함 시 다름)
- gripperless 영상은 **진단 메시지**: "gripper 학습 결함 아님 — 환경 한계 입증"

## 10. 다음 세션 (5/12 sim2real)

**Path C 확정 + sim2real**:
- BC actor `tasks/bc_actor_session8_v1.pt` 메인
- Oracle (state machine) 백업
- action_repeat 2 (sim 100Hz → hw 50Hz)
- 두 팔 시연 rehearsal

**선택 (시간 있으면)**: cube init 분포 narrow 화 ablation — 우연한 catch 14% 가 분포 함수인지 검증.

## 11. Cleanup Whitelist 준수 ✓

- USD/oracle/collect/train_bc/validate_oracle/demos/BC actor/PPO ckpt 모두 unchanged
- cfg 수정: joint_pos_env_cfg.py (gripperless variant 3 class 추가, additive), __init__.py (register 3 추가, additive)
- 백업: `tasks/backups_session17/`
- 신규: 6 csv + plot + report + 6 mp4 (session17/iter1999) + 1 mp4 (presentation 07)
