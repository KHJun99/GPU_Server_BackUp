# 세션 12 결과 요약 보고서 — Demos Coverage Analysis + RL 트랙 종료 확정

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**대상**: 양팔 시연 RL 트랙 (한 팔 담당)

---

## TL;DR

- **Demos coverage 가설 (codex caveat 4) 무효 확인**: 402 demos 의 cube init 이미 광범위 randomize (cube_x ±10cm, cube_y ±20cm)
- **RL 트랙 5세션 누적 ablation 의 모든 가설 exhausted** — 15% ceiling 의 진짜 원인 미해결
- **사용자 결정 — Path C 확정**: RL 트랙 종료, sim2real 즉시 전환
- 시연 자료: Oracle 13% + BC 14% + PPO ablation 충분

---

## 1. 사전 점검 — Plan 가정 invalid

세션 12 plan 의 가정: "demos 가 cube init 의 좁은 영역 (±2cm) 만 cover → 분포 확장 + 재수집으로 saddle 탈출".

실제 분석 결과 (`tasks/demos_session8_coverage.png`):

| 축 | mean | std | range |
|---|---|---|---|
| cube_x (obs idx 20) | 0.1994 | **6.1cm** | [0.10, 0.30] — ±10cm |
| cube_y (obs idx 21) | -0.0086 | **13.4cm** | [-0.20, +0.20] — ±20cm |
| cube_z | 0.012 | 0 | constant |

→ **이미 매우 광범위 randomize**. demos coverage 가 BC 14% saddle 의 원인 **아님** 확정.

## 2. RL 트랙 5세션 ablation 종합 — 모든 가설 exhausted

| Hypothesis | 세션 | 결과 |
|---|---|---|
| H1 mimic dead (ref="rotX") | 2 | PASS |
| H2 35× squash | 2 | PASS (cause: decimation) |
| H3 actuator 6 vs 10 | 2 | PASS |
| H4 cfg 미반영 | 4 | FAIL (cfg 반영됨) |
| H5/H6 distal/effort_limit | 4-5 | partial (decimation 이 진짜) |
| **H_decimation 2→1** | 7 | **TRUE FIX** (0% → 15%) ⭐ |
| H7 entropy 발산 (0.02) | 9 | 10% (악화) |
| H7 entropy 보수 (0.005, 0.001) | 10 | 13% (안정 but plateau) |
| H8 reward shape (lift 75×, reach 0.5×) | 11 | 14% (BC 동등) |
| **H9 demos coverage** | **12** | **무효 (이미 광범위)** |

**진전**: 환경 fix 14pp (0%→14%).
**미해결**: 14% → ≥25% 의 ceiling 원인.

## 3. 사용자 결정 — Path C 확정

세션 12 사용자 명시 승인: **RL 트랙 환경 학습 종료 + sim2real 즉시 전환**.

근거:
- 5세션 ablation 의 모든 가설 진단 완료
- demos coverage 도 광범위 confirmed
- 추가 RL 시도의 효과 보장 X
- Oracle / BC / PPO 영상 자료 충분

작업 2~5 (cube init 확장 + demos 재수집 + BC 재학습 + PPO) **skip**.

## 4. 시연 Path C — Honest narrative

| 자료 | success | 메시지 |
|---|---|---|
| Oracle 4.C + decimation=1 | 13% | task 가능성 입증 (state machine + 환경 fix) |
| **BC 14% (메인)** | **14%** | 재현 가능한 부분 lift, z_max 0.10~0.19m |
| PPO v9-v11 ablation | 10-14% | entropy/reward 진단 분리 (5세션 누적) |

**Narrative**:
1. 환경 5세션 진단 (mimic / actuator / decimation 식별)
2. decimation=1 TRUE FIX (0% → 15%, 의외 발견)
3. entropy / reward / coverage 가설 진단 분리 (각각 invalid 확정)
4. BC 14% 안정 lift 자료 확보
5. **sim2real 진입** — BC actor + Oracle policy hw deploy

## 5. 산출물

### 코드
- `tasks/analyze_demos_coverage.py` (cube init 분포 분석)
- `tasks/demos_session8_coverage.png` (산포도 + 히스토그램, 발표 자료)

### 분석 결과
- cube_x mean 0.1994, std 6.1cm, range [0.10, 0.30]
- cube_y mean -0.0086, std 13.4cm, range [-0.20, +0.20]
- z_max range [0.10, 0.19], mean 0.13
- final_state: 399 LIFT + 3 APPROACH (99.3% LIFT)

### Cleanup Whitelist 준수 ✓
USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged. cfg 수정 없음.

## 6. 다음 세션 13 — Sim2Real 진입

### 우선
1. **BC actor 실 hw 배포 준비** — `bc_actor_session8_v1.pt` (102 demos 학습) 또는 `bc_actor_session12_v1` (없음)
2. **Oracle policy 실 hw 배포** (state machine, 50Hz control)
3. **action_repeat 2 검증** (sim 100Hz dataset → hw 50Hz inference)

### 시연 리허설
- 양팔 시연 한 팔 담당 (RL 트랙)
- Oracle 메인 + BC 보조 + PPO ablation 발표 자료
- 1주 안 시연 day 준비

### RL 트랙 status
**환경 학습 종료 ✓** (decimation=1 fix + ablation 5세션 진단 완료, 15% ceiling 의 진짜 원인 미해결).
**시연용 자료 충분 ✓** (Oracle/BC/PPO 영상 + 5세션 진단 보고서).
