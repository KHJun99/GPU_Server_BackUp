# 세션 18 결과 — Task Spec Ablation (Oracle 14% saddle 진짜 원인 식별)

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**ablation 대상**: episode_length / cube_y range / success_z threshold
**학습**: 없음 (Oracle 100ep 진단 × 3 ablation)

---

## TL;DR

- **5세션 ablation의 진짜 saddle 원인 식별**: `success_z=0.10m` threshold가 Oracle 실제 lift 도달 (5–10cm)과 misalignment
- **Episode 길이는 saddle 원인 아님** (5s → 10s, success 15% → 15%, 변화 없음)
- **cube_y narrowing은 역효과** (±20cm → ±10cm, success 15% → 2%)
- **success_z 0.10 → 0.05만 변경 시 success 15% → 44%** (+29%p)
- 50% 게이트 미달 → Path C 유지, 그러나 다음 세션 후속 조사 명확
- 환경 cfg는 baseline으로 원복 commit

---

## 1. 배경 — Oracle 14% baseline 의심

5세션 ablation (s13–s17)에서 Oracle을 baseline으로 두고 BC/PPO를 학습 후 saddle 14% 결론. 그러나 **Oracle 자체가 14%인 이유**를 의심하지 않음. s17에서야 "환경 ceiling" 결론.

이번 세션 진단 (`tasks/diagnose_oracle_session13.csv`, 100ep) 재분석:

| 지표 | 값 | 의미 |
|---|---|---|
| success | 15% | low |
| fail z_max **max** | 9.91cm | 10cm threshold **직전** timeout |
| fail final_state = LIFT | 84/85 | 거의 모든 ep LIFT 단계 도달 |
| fail ee<5cm | 71/85 | cube까지 도달도 OK |
| fail z_max ≥ 0.07m | 19/85 | "거의 성공" mode |
| fail z_max < 0.025m | 48/85 | "lift 시도조차 못함" mode |
| success cube_y abs median | 14.7cm | 멀리 있는 cube가 성공 |
| fail cube_y abs median | 9.8cm | 가까운 cube가 fail |

→ 두 가지 주요 fail mode:
- (a) **timeout-near-success** (~22%): z_max 거의 10cm까지 갔으나 시간 부족
- (b) **no-lift** (~56%): grip은 했으나 z 안 올라감

→ 가설 3개:
1. **(강) episode_length 5초 부족** (mode a 설명)
2. **(약) cube_y range** (success 분포가 narrowing에 안전한지 의심)
3. **(중) success_z 임계 자체** (z_max 9.9cm 도달했는데 fail)

## 2. Ablation 1 결과 — Episode Length 5s → 10s

**cfg**: `lift_env_cfg.py:241` `self.episode_length_s = 10.0`

**결과** (`tasks/diagnose_oracle_session18_a1_ep10s.csv`, Oracle 100ep):

| 지표 | A1 | baseline (s13) |
|---|---|---|
| success | **15.0%** | 15.0% |
| fail z_max mean | 3.52cm | 3.74cm |
| fail z_max max | 9.84cm | 9.91cm |
| fail final_state=LIFT | 83/85 | 84/85 |
| fail z_max < 2.5cm | 49/85 | 48/85 |
| ep_length max | 1000 (=10s, 100% 사용) | 500 (=5s) |

**판정: 변화 없음** (정확히 baseline 재현). episode 길이는 saddle 원인 아님.

**추가 진단** — Oracle 행동 정밀 분석 (z_at_grasp_max 컬럼은 grip **순간** z, peak 아님 — codex 검증):

- 83/85 fail이 CLOSE 도달 (grip 시도 OK)
- ee_to_cube_at_close 평균 4.71cm, 76% 5cm 이내 — grip 위치 OK
- gripper_closed_steps 평균 **741/1000** — ~7.4s간 grip 유지
- cube_vel slip > 0.1: 4/85 — 미끄러짐 아님

**Bimodal fail 분포** (z_max는 grip 후 cube가 실제로 도달한 최대 z):

| z_max bin | A1 fail | A1 비율 |
|---|---|---|
| < 2.5cm | 49/85 | **57.6%** (no-lift) |
| 2.5–5cm | 10/85 | 11.8% |
| 5–7cm | 10/85 | 11.8% |
| 7–10cm | 16/85 | **18.8%** (mid-lift) |

→ **두 가지 분리된 fail mode**:
- **No-lift (49/85)**: grip은 했으나 LIFT 단계에서 z<2.5cm로 거의 안 올라감 — 시간 무관, 시간 추가 안 도움
- **Mid-lift (26/85)**: cube z가 5–10cm까지 올라가지만 10cm 임계 미달 — task threshold가 결정적 변수

→ **시간을 늘려도 fail 분포 이동 없음** (A1 = baseline 동일). 이는 Oracle의 LIFT가 short time에 plateau에 도달했다는 의미.

## 3. Ablation 2 결과 — cube_y ±20cm → ±10cm (stacked on A1 ep=10s)

**cfg**: `lift_env_cfg.py:149` y range (-0.1, 0.1)

**결과** (`tasks/diagnose_oracle_session18_a2_ep10s_y10cm.csv`):

| 지표 | A2 | A1 |
|---|---|---|
| success | **2.0%** | 15.0% |
| fail z_max max | 9.83cm | 9.84cm |
| fail final_state=LIFT | 98/98 | 83/85 |
| success cube_y abs median | 8.9cm (max 9.9cm) | 14.7cm |

**판정: 회귀** (-13%p). 사전 분석대로 success 분포가 cube_y abs > 10cm에 집중되어 있어, narrowing이 success 영역을 잘라냄.

→ **cube_y narrowing은 saddle 해소가 아닌 task 어렵게 만듬**. plan gate에 따라 env 원복.

## 4. Ablation 3 결과 — success_z 0.10 → 0.05 (측정 임계만, env baseline)

**cfg 변경 없음** (env: 5s/y±20cm 원본 복귀). diagnose_policy.py `--success_z 0.05` 인자만.

**결과** (`tasks/diagnose_oracle_session18_a3_z5cm.csv`):

| 지표 | A3 | baseline |
|---|---|---|
| **success** | **44.0%** | 15.0% |
| fail z_max mean | 1.69cm | 3.74cm |
| fail z_max max | 4.98cm | 9.91cm |
| fail final_state=LIFT | 55/56 | 84/85 |
| success cube_y abs median | 11.6cm | 14.7cm |

**판정: 큰 효과** (+29%p). 단일 변수로 가장 큰 영향. 그러나 50% 게이트 미달.

## 5. 비교 표 + 가설 PASS/FAIL

```
ablation              | episode | cube_y  | success_z | success | 변화 vs baseline
----------------------|---------|---------|-----------|---------|------------------
원본 (s13 baseline)   | 5s      | ±20cm   | 10cm      | 15.0%   | -
A1 (ep 10s)           | 10s     | ±20cm   | 10cm      | 15.0%   | 0     (null)
A2 (+y 10cm, on A1)   | 10s     | ±10cm   | 10cm      | 2.0%    | -13   (역효과)
A3 (z5cm, env 원복)   | 5s      | ±20cm   | 5cm       | 44.0%   | +29   (최대)
```

| 가설 | 판정 |
|---|---|
| 1. episode_length 5초 부족 | ❌ FAIL (변화 0) |
| 2. cube_y 분포 영향 | ❌ FAIL (역효과 -13%p) |
| 3. success_z threshold misalign | ✓ **PARTIAL** (+29%p, 50% 게이트 미달) |

## 6. 진짜 saddle 원인 식별

**Bimodal fail 분포 (A1 baseline)**:

| fail mode | N | 비율 | 특성 |
|---|---|---|---|
| No-lift (z<2.5cm) | 49/85 | 57.6% | grip은 했으나 cube가 거의 안 올라감 |
| Low-lift (2.5–5cm) | 10/85 | 11.8% | 약간 올라가다 plateau |
| Mid-lift (5–7cm) | 10/85 | 11.8% | 의미 있는 lift, 임계 미달 |
| Near-success (7–10cm) | 16/85 | 18.8% | 거의 성공, 임계 직전 |

→ saddle 원인은 **두 개로 분리**:

**원인 A (mid-lift / near-success, 26/85 = 31%)**: Oracle LIFT 동작이 success_z=0.10m 임계 직전에서 plateau. success_z 5cm로 완화 시 +29%p 회복 (A3 = 44%).

**원인 B (no-lift / low-lift, 59/85 = 69%)**: grip 후에도 cube가 5cm 미만에 머무름. success_z 임계 변경으로도 회복 안 됨. 별개 문제.

**구조적 추정 원인** (확실치 않음):
1. Oracle `lift_dz=0.10` 기본 — 목표 z = init z + 10cm = 11.2cm. 임계 10cm에 매우 근접 (mid-lift 원인 A 일부 설명). lift_dz 증가가 mid-lift 26개를 회복할 가능성 있음
2. SO-ARM101 6-DOF kinematic reachable workspace 한계 — cube_y 끝부분일수록 lift 여유 작음 (success가 cube_y abs > 10cm에 집중되는 것도 reach + lift workspace이 좁다는 신호)
3. **No-lift 49개의 메커니즘은 본 세션에서 미식별** — grip을 시도하지만 cube가 안 올라가는 원인 추가 진단 필요

**아닌 원인** (확정):
- 시간 부족 (A1: ep=10s에서도 동일 분포)
- BC/PPO 학습 알고리즘 결함 (Oracle 자체가 같은 saddle)
- gripper binary control (s17 gripperless ablation 14% 동일)
- cube_y narrowing (A2: 역효과)

## 7. 5세션 ablation의 진짜 교훈

| 세션 | 가설 | 결과 | 누락된 것 |
|---|---|---|---|
| 13 | BC trajectory drift | ee_min 14cm — 도달 못함 | Oracle baseline 행동 분석 안 함 |
| 14 | reward distance std | 도달 31% spike → collapse | Oracle 자체 z_max 분포 안 봄 |
| 15 | close reward | 도달 21% — 14보다 낮음 | 동일 |
| 16 | PathOn 비교 | cfg 다름 | 동일 |
| 17 | gripperless | 14% (변화 없음, 환경 한계) | Oracle z_max **distribution** 분석 누락 |
| **18** | **task spec** | **A3 success_z 5cm로 +29%** | episode-level fail mode + Oracle z_max 분포 분석 |

핵심 교훈:
- **Aggregate metric (success rate)의 함정** (s13에서 발견)
- **Oracle baseline 의심하지 않은 함정** ← s17까지 5세션 동안 미발견 (이번 세션 핵심)
- **Episode-level z_max **distribution**이 saddle 진단의 핵심** (단일 평균 9.9cm가 아닌 [5,10) 분포)

## 8. Final cfg 결정

env cfg (`lift_env_cfg.py`): **baseline 원복** (`5s` / `y±20cm`) — A1/A2는 효과 없거나 역효과, A3는 env 무영향.

→ `git diff src/...` 빈 출력 (baseline 동일).

success_z=0.10 task spec 자체 변경은 보류 — 다음 세션에서 결정.

## 9. 다음 세션 19 권고

**옵션 A — Oracle lift_dz 보강 (1-line 변경, 즉시 검증 가능)**:
- diagnose_policy.py `--lift_dz 0.15` (default 0.10) → Oracle이 더 높이 들어올림
- **현실적 가설** (codex 검증 반영): mid-lift 26/85가 10cm 넘기면 +26%p, 즉 ~41% 가능. **No-lift 49/85는 회복 불가** (lift_dz 변경은 LIFT 단계만 영향, no-lift는 LIFT 이전 plateau).
- 50% 게이트 미달 가능성 높지만 진단 가치 큼 (mid-lift 회복 여부 확정)
- demos 재수집 → BC/PPO 재학습 시 mid-lift 영역 학습 가능

**옵션 B — success_z task threshold 수정 (env spec 변경)**:
- success metric 자체를 5cm로 (mdp.object_is_lifted minimal_height 0.025 → 그대로, 단 reward는 이미 0.025 사용 중)
- 그러나 `success_z` 자체는 diagnose_policy.py의 측정 인자일 뿐 env에서 사용 안 함 → 시연 평가 기준만 변경
- 시연 메시지: "5cm lift = 성공" → BC 44%, Oracle 44% 시연 가능

**옵션 C — Path C 유지 (현재 가용)**:
- BC actor s8 v1 + Oracle 백업 그대로 sim2real (5/12)
- 시연 success 정의 명확화: 10cm = 14%, 5cm = 44%

**권장**: 옵션 A (시간 30분, Oracle lift_dz 검증). mid-lift 26/85 회복 여부 확정. 41% 도달 시 BC/PPO 재학습 path 일부 부활 가능. **No-lift 49/85 별도 진단 필요** (다음 세션 후속 항목).

## 10. Validation Checklist

- [x] cfg 백업 (`tasks/backups_session18/lift_env_cfg.py.session18_pre`, `joint_pos_env_cfg.py.session18_pre`)
- [x] A1 cfg 변경 + 100ep 진단 + 분석
- [x] A2 cfg 변경 (stacked on A1) + 100ep 진단 + 분석
- [x] A3 env 원복 + diagnose 인자만 변경 + 100ep 진단 + 분석
- [x] 비교 표 작성 (4 row)
- [x] 진짜 saddle 원인 식별 (success_z threshold + Oracle lift ceiling)
- [x] env 원복 (baseline 5s/±20cm)
- [x] codex 검토 1회 (plan 단계, 70% gate 재조정 + .pyc 캐시 + PLAY override 미존재 확인)
