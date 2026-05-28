# 세션 11 결과 요약 보고서 — Reward Shape Ablation

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**대상**: 양팔 시연 RL 트랙 (한 팔 담당)

---

## TL;DR

- **Reward shape 가설 무효** 확정 — lifting reward 가 reaching 의 **800×** 였음에도 success 14% (BC 정확 동등)
- **3 가설 (entropy / reward / fixed_std) 모두 saddle exit 못 함**
- 남은 가설: **codex caveat 4 — demos coverage**
- 시연 strategy: **Path C (Demo path)** 권고

---

## 1. 배경

세션 9~10 진단:
- entropy 발산 (54×) → entropy 보수화 → success 13~14% plateau
- reward decomp 확인: reaching 0.70 vs lifting 0.16 → **reward misalignment 의심** (codex caveat 1)

세션 11 가설:
- reward weight 재조정 (lifting ↑, joint_vel ↓, reaching ↓) → BC saddle 탈출
- 게이트 ≥18% (BC + 4pp) 미통과 시 옵션 1 (Demo path) 즉시 전환

---

## 2. Ablation 결과

### Reward weight cfg

| Variant | reaching | lifting | joint_vel (curr) | iter 999 success |
|---|---|---|---|---|
| v0 (sess 9) | 1.0 | 15 | -1e-1 | 10% (entropy explode) |
| v1 (sess 10) | 1.0 | 15 | -1e-1 | 13% (entropy 0.005) |
| v2 (sess 10) | 1.0 | 15 | -1e-1 | 13% (entropy 0.001) |
| **sess 11 v1** | 1.0 | **45** | **-3e-2** | **13%** |
| **sess 11 v2** | **0.5** | **75** | -3e-2 | **14%** |

### Reward decomposition (학습 final)

| metric | sess 10 v2 | sess 11 v1 | sess 11 v2 |
|---|---|---|---|
| reaching | 0.70 | 0.020 | **0.001** |
| lifting | 0.16 | 0.470 | **0.817** |
| joint_vel | -0.39 | -0.036 | -0.07 |
| mean reward | 4.54 | 2.02 | **3.38** |
| action_std | 0.15 | 0.39 | 0.48 |

→ reward shape 의도대로 **매우 정확히 변경됨** (lifting/reaching 비율: v0 4× → v2 800×). 그러나 **success rate 무변화**.

---

## 3. 핵심 진단

### 5세션 누적 ablation

| 가설 | 결과 |
|---|---|
| Entropy 발산 (v0) | 10% (악화) |
| Entropy 보수 (v1, v2) | 13~14% (안정 but 무변화) |
| Reward shape v1 (lift 3×) | 13% |
| **Reward shape v2 (lift 5× + reach 0.5×)** | **14%** |

**3 가설 모두 saddle exit 못 함**. 환경 fix(decimation) + entropy 안정 + reward 정렬 적용했어도 plateau.

### 남은 가설 (Codex caveat 4)

**Demos coverage**: 402 demos 가 cube init 의 특정 영역만 cover. BC saddle 이 그 영역에 deeply 수렴. PPO 가 BC trajectory 따라가도록 학습됐지만 새 영역 explore 못 함.

검증 방법:
- demos stratify (z_max + env_idx 분포 분석)
- demos 재수집 with `cube_pos` 메타 추가 (collect_demos.py wrapper)
- cube init 다양화 (yaw randomize, xy 범위 확대)

---

## 4. 시연 Strategy 권고 — Path C (Demo Path)

| 자료 | success | 시연 메시지 |
|---|---|---|
| Oracle 4.C | 13% | "환경 fix + state machine 으로 task 가능성 입증" |
| BC | 14% | "재현 가능한 부분 lift (z_max 0.04~0.19m 분포)" |
| PPO ablation | 13~14% | "5세션 진단 (entropy/reward 분리 정확)" |

**Honest narrative**: 환경 5세션 진단 + entropy/reward 정확 분리 + BC 14% 안정 lift = 양팔 시연 한 팔 RL 트랙 자료.

---

## 5. 산출물

### 코드
- `src/.../tasks/lift/lift_env_cfg.py`: RewardsCfg lifting weight 75, reaching 0.5, joint_vel curriculum -3e-2
- `tasks/backups_session11/lift_env_cfg.py.session11_pre`

### 데이터
- `tasks/ppo_train_session11_v{1,2}.log`
- `tasks/eval_session11_v{1,2}_iter*.log` (8 logs)

### 영상 (`~/jabis_sim/day5/videos/session11/`)
- ppo_v2_reward2_{top,diag}_seed{0,1,2}.mp4 (6 mp4)
- 메시지: "reward 압도적으로 lift 보상 → PPO 가 lift 시도 적극 → 그러나 saddle 못 넘음"

### Cleanup Whitelist 준수 ✓
USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged.

---

## 6. 다음 세션 12 권고 (saddle exit 의 진짜 원인 검증)

### 우선
1. **Demos stratify analysis** (codex caveat 4) — z_max + env_idx 패턴
2. **collect_demos.py wrapper** with cube_pos meta (사용자 승인 필요)
3. **Cube init 다양화** demos 추가

### 사용자 결정 사항
- stratify 결과 후 RL 트랙 보존 vs Demo path 확정
- Demo path 확정 시 RL 환경 학습 종료, sim2real bridging 으로 시연

### RL 트랙 종료 보류 사유
세션 12 의 stratify 결과 보고 결정. 양팔 시연 한 팔 RL 트랙 다운그레이드는 김현준 명시 승인.
