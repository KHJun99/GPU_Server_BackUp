# Sim 환경 PnP Task 진단 — 최종 보고서

날짜: 2026-05-13
이전 보고서: `tasks/oracle_lift_validation.md`, `oracle_grip_ablation.md`, `reach_map_analysis.md`, `reach_map_followups.md`
배경: 사용자가 책상/매트/trash bin/mirror_arm 추가 후 새 시연환경 매칭. Robot 자세 변경 (shoulder_pan +90° = -y forward).

## Phase 1: 환경 정비 완료
- `shoulder_pan` +1.6157 (사용자 의도, robot 정면 -y)
- `shoulder_lift` -0.8, `elbow_flex` 1.0 (work-ready 자세)
- `cube init` (0.24, 0, 0.05) → **(0.045, -0.20, 0.05)** (sweet spot 중앙)
- `pose_range` x (-0.02, +0.02), y (-0.05, +0.05) → cube xy ∈ [0.025, 0.065] × [-0.25, -0.15]

## Phase 2: NullspaceIK Ablation (5 variants)

| Variant | damping | max_dq | nullspace_gain | trash_bin best dist | reached |
|---|---|---|---|---:|---:|
| Baseline | 0.05 | 0.30 | 0.2 | **0.41m** | 0% |
| high_dq | 0.05 | 1.00 | 0.2 | 0.41m | 0% |
| **combined** ★ | **0.02** | **1.00** | **0.5** | **0.09m** | 0% |
| low_damp | 0.02 | 0.30 | 0.2 | (측정 실패) | - |
| high_ns | 0.05 | 0.30 | 0.5 | (측정 실패) | - |

**Combined IK가 압도적 개선** — trash bin 거리 0.41m → 0.09m (4.5배).

## Phase 3: Combined IK 정밀 측정

### 3a: Transport map (cube fixed, goal grid in/near trash bin)
- 모든 56 goal point에서 LIFT 100%, GOAL state 100% (combined IK로 cube grip + lift + transport 시도 가능)
- dist_median 0.19~0.45m
- reached_pct 5cm tol: 0% (모든 점)
- 12cm 도달 비율: 0~100% (점에 따라)

Best transport (가장 가까운 case):
- goal (-0.060, +0.100): dist 0.191m, 12cm 100%, hover 0.168m
- trash bin 영역 (y∈[+0.18, +0.31]) best: 9cm 차이지만 reached 0%

### 3b/3c: Drop bin test (cube → trash bin transport + gripper open)

Baseline IK (Phase 3b):
- cube가 spawn 위치에 정지 → **robot이 cube 잡지도 못함**
- in_bin: 0%

Combined IK (Phase 3c):
- Transport 후 cube xy median (0.040, +0.049, 0.025) — sweet spot에서 **+y 25cm 이동**
- Drop 후 cube xy 분포 매우 넓음 (x ∈ [-0.75, +0.27], y ∈ [-0.36, +0.41], z ∈ [-0.68, +0.16])
- 일부 cube (14.1%) 책상 아래로 튕김
- **in_bin: 0/64 (0%)** — drop 후 cube가 trash bin 벽/책상 가장자리에 부딪혀 튕김

## URDF 분석 (joint limit + reach)

| Joint | Limit (°) |
|---|---|
| shoulder_pan | ±110° (220° swing 가능) |
| shoulder_lift | ±100° (거의 ±1.745 hard limit) |
| elbow_flex | ±97° |
| wrist_flex | ±95° |
| wrist_roll | -157° ~ +163° |
| gripper | -10° ~ +100° |

**Robot total reach**: URDF link origin 누적 → **27.7cm** (직선 펴진 자세)
**Trash bin 거리**: 25cm (robot base → trash bin center)
**→ 이론적으로 reach 가능**

## 결론

### 핵심 발견 4가지

1. **Robot URDF는 PnP에 충분** — link length 27.7cm > trash bin 25cm. 실물 spec과 일치한다면 sim에서도 가능.

2. **NullspaceIK 파라미터가 sim 한계의 진짜 원인** — baseline IK는 단일 step IK라 long trajectory 못 풀음. Combined IK (damping 0.02, max_dq 1.0, ns_gain 0.5)로 transport 거리 4.5배 개선.

3. **Combined IK도 trash bin 직접 도달은 못 함** — best 9cm 차이. drop 후 책상 가장자리/벽 부딪혀 튕김. **Single-target IK의 본질적 한계**.

4. **Multi-waypoint trajectory 필요** — Oracle을 단순 IK 풀이로 쓰면 sim PnP 불가능. 다음 중 하나 필요:
   - Oracle에 intermediate waypoints 추가 (예: cube grip → lift up → side traverse → above bin → drop)
   - 또는 RL이 long-horizon trajectory 학습 (PPO scratch 어려움, BC 또는 demo 필요)

### 사용자가 돌아오면 결정 필요

| 결정 | 옵션 | 비고 |
|---|---|---|
| **A. Oracle 개선** | Multi-waypoint state machine 추가 (PRE_DROP, OVER_BIN, RELEASE) | 1~2일 작업. Oracle을 demo collector로 완성 |
| **B. RL scratch 시작** | combined IK 적용된 Oracle 만 사용 (long trajectory 학습은 RL이) | 8시간 학습. dense reward 필요 |
| **C. BC + RL fine-tune** | Oracle multi-waypoint로 demo 수집 → BC → PPO fine-tune | 2-3일 |
| **D. Task 단순화** | trash bin 제거 또는 robot reach 안에 위치 | 시연환경 매칭 깨짐 |

### 추천

**옵션 A** (Oracle multi-waypoint 구현) — 가장 확실한 sim PnP 해결.
구현 step:
1. OraclePolicy에 LIFT → APPROACH_BIN(over bin, lift_z) → DESCEND_BIN(over bin, bin_z+10cm) → RELEASE(gripper open) state 추가
2. Oracle goal_xy → goal_xyz로 확장 (z target 명시)
3. RELEASE 후 일정 step 후 episode end

이걸로 Oracle이 cube를 trash bin 위까지 옮긴 후 정확히 떨어뜨림. 그 후 BC 또는 RL 학습 진행.

## 코드 변경 사항 (env 현재 상태)

- `joint_pos_env_cfg.py`: baseline_joint_pos work-ready 자세 (shoulder_pan +1.6157, shoulder_lift -0.8, elbow_flex 1.0)
- `env_cfg.py`: cube init (0.045, -0.20, 0.05), pose_range (-0.02~+0.02, -0.05~+0.05)
- `oracle_policy.py`: OracleCfg에 `ik_damping`, `ik_max_dq`, `ik_nullspace_gain`, `finger_x_offset` 추가 (backward compat). `set_goal_xy()` 메서드 추가 (transport 측정용)
- `mdp/events.py`: `reset_cube_grid` 함수 추가 (deterministic grid spawn)
- `scripts/`: `eval_oracle_lift.py`, `eval_reach_map.py`, `eval_transport_map.py`, `eval_drop_bin.py` 모두 IK 인자 + grid 지원 추가

## 산출물

- `/tmp/ik_ablation_*.json` — Phase 2 results (baseline, high_dq, combined)
- `/tmp/transport_combined_fine.json` — Phase 3a (56 goal points)
- `/tmp/drop_combined.log` — Phase 3c (64 envs)
- `tasks/sim_env_diagnosis_final.md` — 이 보고서

---

**한 줄 결론**: URDF는 PnP 가능. NullspaceIK combined setting으로 transport 4.5배 개선. 다만 single-target IK 한계로 trash bin 직접 도달 0%. Oracle multi-waypoint 또는 RL long-horizon 학습이 다음 step.
