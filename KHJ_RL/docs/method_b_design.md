# Method B (EE-delta + env 내부 IK + IL) — Phase 1 재설계 v7

이 문서는 jabis_sim_v2 reward hacking 재발을 절대 차단하는 전제 위에 KHJ_RL Phase 1을 joint-delta + PPO/SAC/BC 경로에서 **EE-delta + env 내부 IK + 순수 BC** 경로로 전환하는 설계의 단일 진실 소스다. v1~v7 거치며 codex 6회 검토 + Claude 자가 검토 1회를 누적했다.

변경 시 이 문서를 갱신하고 CLAUDE.md "Method B pivot" 섹션의 버전 표시와 동기화한다.

용어 정리 (영어 식별자 한국어 풀이):
- end-effector (EE) — 로봇 손끝(여기선 `gripper_dummy_link` 또는 fingertip 중간점)
- inverse kinematics (IK) — 손끝 좌표를 받아 관절 각을 푸는 역기구학
- forward kinematics (FK) — 관절 각으로 손끝 좌표를 구하는 정기구학
- damped least squares (DLS) — 특이점 근처에서 IK가 발산하지 않게 정규화하는 풀이법
- behavior cloning (BC) — 시연 데이터로 정책을 지도학습하는 모방학습
- imitation learning (IL) — BC의 상위 범주
- residual oscillation — 명령한 움직임을 빼고 남는 진동
- terminal window — 에피소드 끝의 strict-eval 구간

## 1. 핵심 결정

| 축 | 결정 |
|---|---|
| Action space | **4-D** `[Δx, Δy, Δz, g]` top-down 고정. orientation은 home pose + wrist_roll clamp로 강제 |
| IK 위치 | **env 내부**. `env._ik_step()`에서 매 policy step (20 Hz, dt=0.05 s) 호출. oracle은 IK 호출 X |
| Algorithm | **Position-only DLS** (3-D task) + **null-space bias toward `q_home`**. 수식은 oracle.py:295-340 이식 |
| Jacobian 소스 | **Pinocchio** (`pin.computeJointJacobians` + `pin.getFrameJacobian(LOCAL_WORLD_ALIGNED)`). PhysX X (real path 호환 + standalone test) |
| Orientation 유지 | null-space bias (간접) + **wrist_roll hard clamp** `q_target[4] = joint_init[4] = 0.0` (직접) |
| Trainer | **BC only** (1차). 정체 시 chunked BC 또는 DAgger. RL 결합 보류 |
| Eval | **5조건 success + 4 reward hacking sub-check + 12 negative scenario test** |

## 2. 정의 (Step A 확정)

### 2.1 commanded_ee_delta_actual

```
commanded_ee_delta_actual(t) = FK(q_target(t)) - FK(q_arm(t))
```

**핵심**: post-EE-cap 단계의 raw command가 아니다. **post-IK-clamp 기대 delta**, 즉 IK가 풀어낸 q_target까지 갔을 때 FK로 예측되는 EE 변위다. DLS damping / joint-limit clamp / wrist_roll clamp 가 모두 반영된 "현실적 의도 delta" — 이 정의 없이는 tracking failure가 jitter로 잘못 잡힌다.

### 2.2 EE jitter (residual)

```
EE_jitter_residual(t) = || EE_pos(t) - (EE_pos(t-1) + commanded_ee_delta_actual(t-1)) ||
```

명령한 움직임을 빼고 남은 부분만 측정 → 진짜 oscillation만 검출. tracking failure는 별도로 잡힌다:

```
tracking_error(t) = || EE_pos(t) - expected_EE_pos(t) ||
                  where expected_EE_pos(t) = FK(q_target(t-1))
```

### 2.3 q_dot, q_jerk 단위

```
q_dot(t)  = (q_arm(t) - q_arm(t-1)) / dt    # rad/s, dt = 0.05 s
q_jerk(t) = (q_dot(t)  - q_dot(t-1)) / dt   # rad/s^2
```

모든 임계값 cfg 필드명에 단위 suffix (`_rad_s`, `_rad_s2`, `_m`, `_m_s`).

## 3. env._ik_step 흐름

1. 현재 EE pose: Pinocchio FK from `q_arm`
2. `target_ee = current_ee + ee_delta * ee_delta_max_m`
3. **EE displacement cap (1차 클램프)**: `||target_ee - current_ee|| ≤ ee_delta_max_m`
4. Pinocchio jacobian `[:3, arm_qidx]` (3-D position only)
5. DLS + null-space bias → `q_dot` → `q_target_raw = q_arm + q_dot`
6. **wrist_roll hard clamp**: `q_target_raw[4] = cfg.robot.joint_init[4] = 0.0`
7. **Joint target delta cap (2차 클램프)**: `||q_target_raw - q_arm||_∞ ≤ ik_max_joint_delta_rad_per_step`
8. **Joint soft limit clamp**
9. 다층 발산 가드 (4.1 참조)
10. `set_joint_position_target(q_target_final + 4-bar mimic + gripper)`
11. `commanded_ee_delta_actual = FK(q_target_final) - FK(q_arm)` 를 다음 step jitter 계산용으로 캐시

## 4. 가드

### 4.1 IK 다층 발산 가드

| 카테고리 | 조건 | 동작 |
|---|---|---|
| 즉시 AND | `q_dot_step.abs().max() > 0.3 rad` AND `||x_err|| > 0.05 m` | fallback `q_target = last_valid`, `_ik_fail_count += 1` |
| 즉시 OR | `EE_jitter_residual_p95 > T_jitter` (10-step window) OR `q_jerk_p95 > T_jerk` | 동일 fallback |
| 누적 | `consecutive_ik_fails ≥ 10` (0.5 s 연속) | **episode truncate** (success=False) |
| 느린 발산 | `||x_err||`가 rolling 20-step 단조 증가 | warning 로깅 (success 차단 X) |

**T_jitter, T_jerk**: M0.5에서 oracle traces로 calibrate 후 cfg에 박는다. **calibration set ≠ validation set** (서로 다른 100 ep) — 같은 데이터로 산출+검증 금지.

**Fail reason 5종 분류 로깅**: `joint_limit` / `singular` (`det(JJT) < 1e-4`) / `unreachable` (`||x_err|| > 0.1 m`) / `large_qdot` / `jitter`. 모든 fallback 발동 시 카테고리 기록.

### 4.2 Reward hacking sub-check (success.py 보강)

기존 5조건 안에 sub-check 추가. transit phase는 rolling tolerance, **terminal window (마지막 20 step = 1 s) 안은 매 step strict**.

| 5조건 # | Sub-check | Transit | Terminal (last 20) |
|---|---|---|---|
| #1 lift_history latch + post-lift consistency | `cube_z ≥ 0.05` AND `cube_visible` | 3-step rolling (2/3) | 매 step strict |
| #2 stable placement | `std(cube_xy) < 5 mm` over 20-step window | (이미 strict) | (이미 strict) |
| #3 release+retreat | `gripper_cmd == open` AND `gripper_joint_pos ≥ 0.8 * open_max` AND **contact separation** | 5-step rolling (4/5) | 매 step strict |
| (전체) IK fail masking | `consecutive_ik_fails ≥ 5` OR `ik_fail_count/total > 5%` | success=False | **last 20-step 안 1 fail로도 success=False** |
| (신규 v7) cube orientation | cube up-axis vs world up `< 30°` (tipped/edge 차단) | rolling 3 | 매 step strict |

contact separation = fingertip ↔ cube 거리 ≥ 1 cm. 4-bar mimic 지연으로 가짜 release 차단.

### 4.3 Audit 카테고리 (zero guard trip 정의 분리)

| 카테고리 | 정의 | 요구 |
|---|---|---|
| Hard exploit trip | success=False로 차단된 가드 발동 | **100 ep 전체 0건** |
| Audited warning trip | 단발 IK fail, 단발 occlusion, transit-only sub-check 미스 | **평균 ≤ 1/ep, p95 ≤ 2/ep** |
| Terminal-adjacent warning | success=True ep의 마지막 40 step 안 warning | **0건** (절대) |
| Category rate | 단일 카테고리 발동률 | `> 1%`이면 자동 audit 플래그 |

## 5. cfg dataclass

`src/khj_rl/envs/cube_lift/cfg.py`에 다음 추가:

```python
@dataclass
class EEControlCfg:
    ee_delta_max_m: float = 0.015                  # M0.5 측정
    dls_lambda: float = 0.1
    null_bias_gain: float = 0.15
    wrist_roll_clamp_to_home: bool = True
    ik_max_joint_delta_rad_per_step: float = 0.3
    ik_max_position_error_m: float = 0.05
    ik_jitter_window: int = 10
    ik_jitter_residual_p95_max_m: float = 0.0      # M0.5 calibrate
    ik_jerk_p95_max_rad_s2: float = 0.0            # M0.5 calibrate
    ik_consecutive_fail_truncate: int = 10
    ik_slow_diverge_window: int = 20
    orientation_drift_warn_rad: float = 0.2

@dataclass
class SuccessGuardCfg:
    visibility_transit_tolerance: int = 3
    gripper_match_transit_tolerance: int = 5
    cube_orientation_transit_tolerance: int = 3
    cube_max_tip_angle_rad: float = math.radians(30.0)
    terminal_window_steps: int = 20
    contact_separation_min_m: float = 0.01
    ik_fail_mask_consecutive: int = 5
    ik_fail_mask_ratio: float = 0.05
    ik_fail_mask_terminal_zero: bool = True

@dataclass
class AuditCfg:
    warning_mean_per_episode_max: float = 1.0
    warning_p95_per_episode_max: int = 2
    warning_terminal_adjacent_window: int = 40
    warning_terminal_adjacent_max_in_success: int = 0
    warning_category_rate_audit_threshold: float = 0.01
```

세 cfg는 `CubeLiftEnvCfg`에 `field(default_factory=...)`로 wire. M0.5/M1 동안은 dormant (env가 아직 안 읽음). M2에서 활성화.

## 6. 마일스톤

### M0.5. 정의 박기 + Calibration + 회귀 자산 만들기 (2 d)

**Step A — 정의 박기 (지금 진행 중)**:
- 본 문서 작성
- cfg.py에 EEControlCfg / SuccessGuardCfg / AuditCfg 추가 (dormant wire)
- CLAUDE.md에 Method B pivot 섹션
- 메모리 갱신

**Step B — Jacobian 검증 회귀 자산**:
- `tests/test_ik_jacobian_regression.py`
- 9 pose (home + 4 limit-near + 1 singular + 3 task-critical: grasp 직전 / lift 직후 / release+retreat 직전) + 10 random + finite-diff 3-way 비교
- `arm_qidx`, `frame_id`, `joint name/order` assert 매 test 시작 시

**Step C — Negative scenarios 12개 합성**:
- `tests/test_reward_hacking_negative.py`
- 시나리오 목록:
  1. cube 수평 슬라이드 → goal
  2. cube가 lift 없이 goal 도달 (slide/nudge)
  3. cube 짧게 goal 진입 후 튕겨 나옴
  4. cube lift 후 carrying 중 visibility 소실 (terminal 직전)
  5. gripper open 명령 발행했으나 cube 끼어 따라옴
  6. Terminal pose에서 IK fail/clamp + cube 위치 정상
  7. Placement 위치 맞지만 vel_stab 초과 (jabis_sim_v2 핵심)
  8. Lift 후 cube drop → slide/roll로 goal 도달
  9. Gripper closed 상태로 cube 들고만 있음 (release 없음)
  10. Release command + joint open 일치하지만 contact separation 없음
  11. Terminal window 밖 IK fail 후 마지막 20 step만 clean
  12. Cube tipped/edge contact 상태로 goal, visual만 맞음
- 각 시나리오는 forced state 합성. 5조건 + sub-check 가 success=False로 거부하면 통과.

**Step D — Oracle EE 자연 속도 측정**:
- 기존 joint-delta oracle 100 ep
- phase별 step당 EE 이동량 median / p90 / p95 / max
- `ee_delta_max_m = global p95 max` (phase별 진단도 보유, ratio > 3x면 v8에서 phase별 threshold 도입 고려)
- 측정 EE frame이 IK가 쓸 frame과 동일 (`gripper_dummy_link`) assert

**Step E — Threshold Calibration / Validation split**:
- **Calibration**: 새 EE-delta oracle 100 ep, seed 0~99 → jitter_residual / q_jerk p95 산출 → `T_jitter = p95 × 2.5`, `T_jerk = p95 × 2.5`
- **Validation**: 별도 100 ep, seed 100~199 → OR-guard trip rate < 1% 확인
- 1% 초과 시 임계값 0.5× 완화 후 재검증
- 결과를 `runs/calibration/ik_thresholds.json` 영구 저장

**Step F — Home pose 검증**:
- `_fk_ee_pose(joint_init)` 결과 quat → 좌 distal link 로컬 +X축 vs world -Z 각도 < 10° 확인

### M1. IK step 단위 검증 (1.5 d, AppLauncher 없이)

- env에 `_ik_step` 추가 (기존 step()과 병렬, 둘 다 호출 가능)
- 단위 테스트:
  - home pose ±10 cm xyz delta 50 step → EE 추적
  - 특이점 강제 → 다층 발산 가드 (즉시/누적/느린) 발동
  - wrist_roll clamp (큰 delta에서 변동 < 0.05 rad)
  - 발산 오탐 회귀 (정상 큰 delta는 통과)
  - **Case 분리**: `reachable` / `unreachable` / `limit-near` / `singular` 4 유형, 각 fail reason 분류 + truncate 동작 검증 + limit-near 회복 가능성
  - **Jitter 정의 회귀**: commanded delta = EE displacement일 때 residual ≈ 0
  - **Terminal 20-step strict 회귀**: window 안 1 step IK fail로도 success=False
- 합격 기준:
  - 100 random target, EE 추적 오차 평균 < 5 mm AND p95 < 1 cm
  - IK jitter 평균 < 1 mm/step AND p95 < 2 mm/step
  - 4 case fail reason 정확 분류
  - jitter OR-guard 합성 시나리오 100% 검출

### M2. Action space 전환 (0.5 d)

- env.step EE-delta 교체, action_space (4,)
- obs `last_action` 6→4, 총 obs 31→29
- 기존 ckpt 로드 시 shape mismatch fail-fast
- launch_viewer manual jog (EE delta UI 입력 → arm 추종)
- 기존 joint-delta path는 dormant 유지 (즉시 삭제 X)

### M3. Oracle 재작성 (0.5 d)

- oracle.py IK 블록 제거, EE-delta 출력
- 6 phase + target_fn + exit 로직 유지
- 100 ep 평가의 필수 산출물:
  - Hard exploit / Audited warning 분리 표
  - Fail reason 5종 분포
  - Terminal window IK fail count (전체 0)
  - Warning category별 rate (> 1% 자동 audit)
- 합격:
  - success ≥ joint-delta oracle baseline - 5%
  - vel_stab violation = 0
  - IK fail rate < 1%
  - episode 길이 +10% 이내
  - Hard exploit = 0
  - Warning 평균 ≤ 1/ep, p95 ≤ 2/ep
  - success=True ep terminal-adjacent warning = 0
  - 신규 가드 false-positive (oracle 정상 동작 차단) < 1%

### M4. Demo 재수집 (0.5 d 자동)

- stage 0, 1000 episode, success_only
- cube_init seed 재사용 (직접 비교 가능)
- 수집 중 Hard exploit trip = 0, terminal IK fail = 0 확인
- 수집된 demo의 EE displacement p95가 M0.5 측정과 일치 사후 검증

### M5. BC 학습 + eval (0.5 d → stretch 1~2 d)

- bc.py 그대로, network action_dim 4
- ObsNormalizer 새 demo로 재fit
- video_sampler 학습 중 자동 영상

평가 프로토콜:
- **1차 (Pass)**: stage 0 / stage 1 각 100 ep (seed 0~99 고정)
- **2차 (Stretch)**: **3 seed × 100 ep** (seed-to-seed variance 측정)

합격:
- Pass: success ≥ 70% + Hard exploit = 0 + Warning v7 기준 만족 + phase failure breakdown 안정 → M6 진입
- Stretch: 3 seed에서 success ≥ 80% → 방식 전환 입증

### M6. Curriculum / 정체 시 대응

- stage 1 (10 cm) demo 추가 수집 + BC 재학습
- 정체 시:
  - ChunkedActorCritic (chunk_size=12) + chunk 평균 delta 제한 가드
  - 또는 DAgger (oracle relabel)
- RL 결합 보류

## 7. 폐기 / 보관 / Dormant

**폐기**: 기존 BC v0~v3 ckpt, F15 PnP ckpt, `runs/demos/stage*` 전부 (action space 호환 X)

**보관**: oracle DLS + null-space IK 수식 (env로 이식), Pinocchio FK setup, `success.py` (sub-check 4종 + tipped 차단 추가 형태), `video_sampler.py`, `obs_alignment.py`, `demo_buffer.py` 포맷

**Dormant**: `training/ppo.py`, `sac.py`, `her.py`, `envs/cube_lift/reward_dense.py` — 삭제 X, 자리 유지

## 8. Open question (M0.5 진행 중 결정)

1. Phase별 threshold 도입 트리거: M0.5 Step D 결과 phase별 EE 속도 p95 max/min ratio > 3x이면 v8 patch로 phase별 threshold 도입 고려. ratio ≤ 3x이면 단일 threshold 유지.
2. Negative scenario #10 (contact separation 없는 release)의 contact distance 측정: sim에서 fingertip↔cube 거리 ≥ 1 cm 사용. Real에서는 RGB-D depth 차이 또는 force sensor — Phase 1은 sim 검증만.

## 변경 이력

- v1~v6: 폐기 (codex 검토 누적). v7부터 단일 진실 소스로 운영.
- v7 (2026-05-23): 본 문서 신설. codex v6 리뷰 반영 (jitter 정의 명확화, calibration split, terminal window 20-step, warning 강화, negative scenario 12개).

## 구현 진행 상태 (2026-05-23)

- [x] **M0.5 Step A** — 정의 박기 (cfg dataclass 3종 dormant wire + CLAUDE.md + 메모리 + 본 문서)
- [x] **M0.5 Step B** — Jacobian regression test (`tests/test_ik_jacobian_regression.py`, 22 standalone + 1 opt-in PhysX parity)
- [x] **M0.5 Step C (부분)** — Negative scenario 6개 (`tests/test_reward_hacking_negative.py`, baseline sanity + 6 negative). 잔여 6개는 M2 sub-check 도입 후
- [ ] **M0.5 Step D** — Oracle EE 자연 속도 측정 → `ee_delta_max_m` 확정 (M3 의존)
- [ ] **M0.5 Step E** — Threshold calibration / validation split → `T_jitter, T_jerk` 확정 (M3 의존)
- [x] **M0.5 Step F** — Home pose top-down 검증 (`tests/test_home_pose.py`, 3 test, down_align=0.964)
- [x] **M1** — Standalone IK pure function 단위 테스트 (`tests/test_ik_step.py`, 12 test)
- [x] **M2** — `src/khj_rl/envs/cube_lift/ik.py` 신설 + `cfg.py` action_size 4 hard switch + `env.py` step() EE-delta + IK 통합 (info dict에 ik audit 노출, consecutive_ik_fails truncate, P0 action shape ValueError assert). codex M2 review CONDITIONAL GO 통과
- [x] **M3.1** — `oracle.py` EE-delta 출력 재작성 (IK 블록 제거, 6-phase MP 유지, 4-D action 반환)
- [x] **M3.2** — P1 test 보강 (tests/test_ik_step.py 16 test: jitter strict + first-step fallback + slow_diverge + reset assertions + 100 random target 통계)
- [x] **M3.3 script** — `scripts/run_ik_calibration.py` (M0.5 D+E + M3 oracle eval 한 script). 산출물 `runs/calibration/ik_thresholds.json` + report
- [x] **M3.3 실행 + cfg 갱신** — 사용자가 calibration 돌리고 EEControlCfg 갱신 완료. ee_delta_max_m=0.14, ik_jitter_residual_p95_max_m=0.014493
- [x] **BC pipeline 4-D 정비** — network.py / bc.py / demo_buffer.py / train_bc.py 호환 확인, docstring 갱신, ChunkedActorCritic 4-D forward pass smoke test 통과
- [x] **Test 격리** — `cfg` fixture가 placeholder ee_control 사용해서 calibration 변경에 영향 없도록. 48 passed 유지
- [ ] **M4** — stage 0, 1000 episode demo 재수집
- [ ] **M5** — BC 학습 + eval (Pass 70%, Stretch 3 seed 80%)
- [ ] **M6** — stage 1 / 정체 시 chunked BC + chunk delta cap 또는 DAgger
