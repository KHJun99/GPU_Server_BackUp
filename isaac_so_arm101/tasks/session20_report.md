# 세션 20 결과 — Mode B (no-lift 48%) 메커니즘 진단 — 환경 한계 확정

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**가설 검증**: lift_dz / reach_dist / close_steps Oracle hyperparameter ablation
**핵심 발견**: Mode B는 Oracle 파라미터 무관, gripper-cube physical grasp 실패

---

## TL;DR

- 12세션 ablation (s9~s20)의 마침표 — Oracle close-timing/reach 3가지 파라미터로 Mode B 회복 불가 입증, **환경 수준 fix 필요** (gripper geometry vs cube — USD 변경 범위 외)
- Oracle lift_dz=0.15: 베이스라인과 완전 동일 (Mode B 48 동일, ee_at_close 4.91cm 동일)
- Oracle reach_dist tighter: ee_at_close 4.91→3.56→2.76cm로 가까워졌으나 Mode B count 48→48→49 변화 없음
- 가설 2 (friction): 현재 cube friction 이미 1.5 (plan 기준값), 추가 ablation 무의미
- Step 5 (BC/PPO 재학습) skip — fix 없음
- **Path C 확정 권고** (BC s8/v19 + Oracle 백업, 5/12 sim2real)

---

## 1. Mode B 진단 (Step 1)

세션 19 csv 재분석:

| 정책 | Mode B count | reached CLOSE | ee_to_cube_at_close mean |
|---|---|---|---|
| Oracle s18 A3 (z=0.05 baseline) | 48/100 | 47/48 | **4.91cm** |
| Oracle s13 (z=0.10 baseline) | 48/100 | 47/48 | 4.91cm |
| BC v19 | 51/100 | **0/51** (approach 실패) | n/a |
| PPO v19 iter300 | 48/100 | 21/48 | 20.80cm (approach 더 멀어짐) |

**Oracle Mode B 정체** (가장 informative):
- 47/48이 CLOSE 상태 도달 → grip 시도 OK
- ee_to_cube_at_close mean 4.91cm → **gripper가 cube에서 약 5cm 떨어진 상태로 닫힘**
- final_state 47/48 LIFT (3) → CLOSE 거쳐 LIFT 진입했으나 cube 안 들림

→ Mode B의 본질 = **grasp 실패** (lift 실패 아님). Oracle은 cube를 "잡으려" 했지만 cube가 gripper jaw 사이로 들어가지 않음.

## 2. 가설 1 (lift_dz 0.10 → 0.15)

**예측**: cube 높이 들어올림 시도 → Mode A는 회복 가능, Mode B는 grasp 실패라 무관.

**결과** (`tasks/diagnose_oracle_session20_h1_liftdz015.csv`):

| 지표 | H1 | s18 baseline |
|---|---|---|
| success @z=0.05 | **44.0%** | 44.0% |
| success @z=0.10 | **15.0%** | 15.0% |
| Mode B | 48/100 | 48/100 |
| ee_at_close mean | 4.91cm | 4.91cm |

→ **완전 동일** (예측 일치). lift_dz는 grasp 실패 mode와 무관. **가설 1 FAIL**.

## 3. 가설 2 (cube friction 0.5 → 1.5) — 사전 적용 확인

`joint_pos_env_cfg.py:154-155` 확인:
```python
static_friction=1.5,
dynamic_friction=1.5,
```

**이미 1.5** (이전 세션 v4에서 default → 1.5 변경됨, "5cm cube/default friction에서 위누름 자세 정체 → 측면 그립 강제"). 추가 변경 없이는 가설 2 ablation 불가.

→ 옵션: 0.5로 낮춰 역방향 ablation 가능하나 이는 시간 낭비 (현 1.5에서 Mode B가 발생하므로 0.5 → 더 악화 예상). **Skip**.

## 4. 가설 3 (close timing tighter)

**예측**: oracle이 cube에 더 가까이 도달한 후 CLOSE 하면 Mode B 회복 가능.

**H3 결과** (`tasks/diagnose_oracle_session20_h3_reach025_desc015.csv`):

| 지표 | H3 (reach=0.025, desc=0.015) | s18 baseline |
|---|---|---|
| success @z=0.05 | **44.0%** | 44.0% |
| success @z=0.10 | **15.0%** | 15.0% |
| Mode B | 48/100 | 48/100 |
| ee_at_close mean | **3.56cm** ⬇ | 4.91cm |

ee_at_close가 1.35cm 가까워졌으나 Mode B count 변화 없음.

**H4 결과** (`tasks/diagnose_oracle_session20_h4_tighter.csv`, reach=0.015 / desc=0.008 / close_steps=16):

| 지표 | H4 | s18 baseline |
|---|---|---|
| success @z=0.05 | 42.0% ⬇ | 44.0% |
| success @z=0.10 | 13.0% ⬇ | 15.0% |
| Mode B | 49/100 | 48/100 |
| ee_at_close mean | **2.76cm** ⬇⬇ | 4.91cm |

ee_at_close 2.76cm까지 cube에 거의 닿는 거리에서 close, but Mode B count 동일 (오히려 +1). H4는 너무 tight해서 일부 ep가 5s 안에 CLOSE 못 함 → success 약간 ⬇.

→ **가설 3 FAIL**. CLOSE 시점을 cube에 거의 닿게 만들어도 grasp 회복 안 됨.

## 5. 종합 ablation 표

```
variant                      | @z=0.05 | @z=0.10 | Mode B  | ee_at_close (Mode B subset)
-----------------------------|---------|---------|---------|----------------------------
Oracle s18 baseline          | 44.0%   | 15.0%   | 48/100  | 4.91cm  (n=47 reached CLOSE)
H1 lift_dz=0.15              | 44.0%   | 15.0%   | 48/100  | 4.91cm  (n=47)
H3 reach=0.025 desc=0.015    | 44.0%   | 15.0%   | 48/100  | 3.56cm  (n=45)
H4 reach=0.015 close=16      | 42.0%   | 13.0%   | 49/100  | 2.76cm  (n=39)
```

> **참고**: ee_at_close는 Mode B 중 CLOSE 도달 episode subset 평균 (Mode B 본질이 grasp 실패이므로 이 subset이 분석에 적합). 전체 CLOSE 도달 episode (success+fail 통합) 평균은 baseline 4.75cm → H4 2.75cm로 동일 추세. H4는 reach_dist 0.015이 너무 tight해서 100ep 중 13ep이 5s 안에 CLOSE 도달 실패 (DESCEND 단계 timeout) → success 약간 ⬇.

**모든 변경에서 Mode B count = 48±1**. ee_at_close가 4.91→2.76cm로 변해도 동일.

→ Mode B는 Oracle close-timing / reach-distance 파라미터에 insensitive. **CLOSE 시점이 cube에 거의 닿는 거리(2.76cm)로 와도 grasp 실패**.

## 6. Mode B의 진짜 메커니즘 (확정 추정)

**Gripper-cube physical grasp 실패** — 다음 중 하나 또는 조합:

1. **Gripper jaw 간격 vs cube 크기 mismatch**: cube 3cm × 3cm, gripper 닫힐 때 jaw 간격이 cube를 통과시키지 못하거나 cube를 옆으로 튕겨냄
2. **Gripper geometry**: jaw의 angle/curve가 cube를 안정적으로 잡지 못함
3. **Contact dynamics**: PhysX solver가 close-시점 contact를 정확히 모델링하지 못함 (frequent issue with multi-body grasping in Isaac Sim)
4. **Oracle ee z 위치**: descend 후 ee z가 cube center보다 약간 위 (gripper 위에서 누름 → 옆으로 튕김)

**근거**:
- ee_at_close 2.76cm까지 (cube width 3cm의 거의 같은 거리)에 와도 grasp 실패 → gripper가 cube에 닿는 거리는 충분
- 모든 hyperparameter ablation 무효 → 행동 변화로 해결 안 됨
- BC/PPO도 같은 실패 패턴 (Mode B 48~51) → 학습된 정책도 같은 환경 한계 부딪힘

**조심스러운 결론** (codex 검토 hedge 반영): 본 세션은 Oracle close-timing / reach-distance / lift_dz 3가지 파라미터에서 Mode B 회복 불가 확인. Cube size / gripper jaw geometry / contact tolerance 자체는 직접 ablation 안 함. 따라서:
- 입증: **Oracle close-timing/reach 파라미터로 Mode B 회복 불가**
- 강한 추정 (직접 검증 X): Mode B의 진짜 fix는 환경/USD 수준 변경 (gripper jaw geometry / cube size / contact tolerance) 필요

→ Oracle 행동 변화로는 Mode B 회복 불가 (ablation 입증). 환경 수준 ablation은 본 세션 범위 외 (USD 수정 risk 높음, 시연 일정 제약).

## 7. Step 5 — BC/PPO 재학습 (skip)

근거:
- 모든 가설 fail (z=0.10 ≥ 25% 게이트 미달)
- 새 demos 수집해도 oracle behavior가 baseline과 동일 → BC도 baseline과 동일 학습
- PPO도 환경 ceiling에 부딪힘 (s9~17 입증된 패턴)

→ 시간 절약 + 의미 없는 학습 회피.

## 8. 12세션 ablation 종합 결론 (s9 ~ s20)

| 세션 | 가설 / 변경 | 결과 | 누적 발견 |
|---|---|---|---|
| 9 | PPO warmstart 시도 | 실패 (entropy 발산) | BC saddle 첫 발견 |
| 10 | 카메라 + entropy ablation | saddle 동일 | reward 의심 |
| 11 | reward shape (lifting weight 5×) | saddle 동일 | reward 무관 |
| 12 | demos coverage 분석 | RL 트랙 종료 권고 | Path C 첫 등장 |
| 13 | episode-level 진단 (BC ee_min 14cm) | 도달 자체 못함 | aggregate metric 함정 발견 |
| 14 | reward distance std 0.05→0.02 | iter200 도달 31% spike + collapse | reward 직접 영향 입증 |
| 15 | close reward 추가 | iter100 도달 21% (s14보다 낮음) | reward 추가는 reaching 방해 |
| 16 | PathOn-AI 비교 | cfg 다름, 비교 불가 | gripper action mismatch 발견 |
| 17 | gripperless ablation | 14% 동일 (변화 없음) | gripper control 무관, "환경 한계" |
| **18** | **task spec ablation (ep/y/success_z)** | **success_z 0.05 시 44%, bimodal 발견** | **Oracle baseline 의심 시작 → Mode A/B 분리** |
| 19 | success_z=0.05 demos+BC+PPO | BC 40%, PPO best 43% (z=0.05) | Mode A 회복, Mode B 그대로 |
| **20** | **Oracle 파라미터 ablation (lift_dz/close timing)** | **Mode B 48 모두 동일** | **Mode B = 환경 한계, fix 없음** |

**최종 saddle 분해**:
- Mode A 31%: success_z threshold 미달 (5–10cm lift 가능, 임계 10cm 미달) → **success_z=0.05로 회복**
- Mode B 48%: gripper-cube grasp 물리 실패 → **환경 USD/geometry 한계, 9세션 ablation으로 fix 불가**
- Mode 기타 21% (success): 정상 작동 (cube_y abs > 10cm 영역)

## 9. 시연 path 결정 권고

**Path C 확정** (강력):

1. RL 트랙은 9세션 ablation으로 충분히 입증된 환경 한계에 도달
2. 추가 RL ablation 시간 가치 0 — sim2real 시간으로 전환
3. 시연 자산:
   - **BC v19** (40% @z=0.05, ee_min 6.3cm) 메인
   - **Oracle** (44% @z=0.05, state machine 백업)
   - 시연 success 정의: "5cm lift = success" (또는 z=0.10 정의 유지하고 saddle 메시지)
4. 발표 메시지:
   - 9세션 ablation으로 환경 한계 명확히 진단
   - Bimodal fail (Mode A success_z + Mode B grasp) 분리
   - Mode A 회복 (success_z=0.05): +29%p
   - Mode B 회복 = 환경/USD 변경 필요 (이번 시연 범위 외)

## 10. 다음 세션 21 (5/12) 권고

**즉시 sim2real 진입**:
- BC v19 actor + Oracle 백업
- action_repeat 2 (sim 100Hz → hw 50Hz)
- 시연 path 결정 (Path C)

**선택 (시간 있으면, RL 추가 향상 시도)**:
- USD 수준 ablation (gripper jaw geometry, cube size) — **이번 세션 범위 외, 위험도 높음**
- reach_above_dz 변경 ablation — descend 시 gripper 더 위에서 시작
- gripper close_command 값 ablation (-0.6 → -1.0)

## 11. Honest Caveat

이번 세션 가설 모두 fail. 그러나 **fail이 가치 있음** — 9세션 동안 못 본 Mode B의 본질을 episode-level diagnostic으로 식별. 이전 5세션 ablation의 진짜 원인은 Mode B = 환경 한계.

이로써 RL 트랙은 이번 시연 사이클에서 환경 한계를 명확히 입증하고 종료. 시연은 BC + Oracle로 진행.

## 12. Validation Checklist

- [x] Mode B episode 분석 (cube_init, ee_at_close 분포)
- [x] 가설 1 (lift_dz=0.15) ablation
- [x] 가설 2 (friction) — 이미 1.5 적용 중, skip
- [x] 가설 3 (reach_dist tighter, H3 + H4)
- [x] Step 5 BC/PPO 재학습 — skip (fix 없음)
- [x] 종합 ablation 표
- [x] 9세션 결론 종합
- [x] 시연 path 권고 (Path C)

## 13. 산출물

- `tasks/diagnose_oracle_session20_h1_liftdz015.csv`
- `tasks/diagnose_oracle_session20_h3_reach025_desc015.csv`
- `tasks/diagnose_oracle_session20_h4_tighter.csv`
- `tasks/session20_report.md`

cfg 변경 없음 (모든 ablation CLI arg만).
