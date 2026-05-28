# 세션 14 결과 — Reward distance std fix (saddle 진짜 원인)

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**Fix 1 ablation**: `lift_env_cfg.py:160` `reaching_object std=0.05 → 0.02`
**기타 cfg**: 모두 동결 (Fix 한번에 하나)

---

## TL;DR

- **부분 효과 — 도달 능력 일시 회복**: iter 200 에서 ee<5cm 도달율 **2% → 31%** (15× 개선)
- **그러나 transient & unstable**: iter 500 이후 close 시도 collapse, iter 999 close 0%
- **Success rate 변화 없음**: 14% (s13 동일) — 도달했지만 close timing 무너짐
- **핵심 게이트 미달**: 도달율 < 50% → fix 1 단독 부족 확정
- **시연 path B/C 보류** — close 학습 무너지기 전 early stop (iter 200) 또는 fix 2 (DAgger) 추가 필요

---

## 1. Fix 1 가설 + 진단 근거

세션 13 데이터:
- BC ee_to_cube_min mean **13.9cm** (Oracle 4.6cm 대비 3배 멀음)
- ee<5cm 도달율 BC 2% / PPO 1% / Oracle 86%

→ 가설: `reaching_object` reward `std=0.05` 가 너무 wide — ee 5cm 떨어진 채 reward 받아서 도달 incentive 부족. std=0.02 로 sharper landscape 만들면 saddle 도달 학습 강제.

## 2. Reward landscape 비교

| distance | reward std=0.05 | reward std=0.02 |
|---|---|---|
| 1cm | 0.80 | 0.54 |
| 2cm | 0.62 | 0.24 |
| 3cm | 0.46 | 0.10 |
| 5cm | 0.24 | **0.013** |
| 10cm | 0.04 | 0.0001 |
| 14cm (BC drift) | 0.007 | ~0 |

→ std=0.02 로 ee 5cm 안 와야 reward 회수. 14cm 영역 reward 사실상 0 (BC saddle drift 정확히 punish).

`tasks/reward_landscape.png` 시각화.

## 3. 학습 progression (full iter sweep)

| iter | succ% | **ee<5cm%** | ee_min | close% | close_dist |
|---|---|---|---|---|---|
| 0 (BC init) | 14.0 | 2.0 | 0.123 | 86 | 0.144 |
| 100 | 15.0 | 1.0 | 0.159 | 100 | 0.170 |
| 150 | 14.0 | 1.0 | 0.154 | 100 | 0.196 |
| **200** | 14.0 | **31.0** ⭐ | **0.068** | 100 | 0.212 |
| 250 | 14.0 | 3.0 | 0.152 | 100 | 0.199 |
| 300 | 14.0 | 2.0 | 0.161 | 100 | 0.187 |
| **400** | 12.0 | **26.0** | 0.085 | 100 | 0.193 |
| 500 | 14.0 | 3.0 | 0.142 | **2** ⚠️ | 0.279 |
| 999 | 13.0 | 2.0 | 0.144 | **0** ⚠️ | nan |

**진단**:
1. **iter 200/400 sweet spot** — 도달율 spike (31%/26%). reward fix 효과 명확
2. **그러나 iter 250/500/999 다시 떨어짐** — 학습이 도달 능력 재차 잃음 → sharper landscape 가 PPO exploration 망가뜨림 (gradient 가파른 영역에 갇혔다 빠져나옴)
3. **iter 500 부터 close 시도 collapse (100% → 2% → 0%)** — gripper 정책이 학습 중 무너짐. lifting_object reward (weight 75) 가 close 직접 보상 안 함 + entropy 가 close 행동 잊게 함
4. **success rate 14% 일정** — 도달 spike 시점에도 close timing 잘못 (close_dist 0.21cm 라는 비정상 ─ 도달 후에도 잘못 시점 close)

## 4. Policy Behavior 비교 표

| metric | s13 BC | s13 PPO | s14 iter200 | s14 iter999 |
|---|---|---|---|---|
| success rate | 14% | 13% | 14% | 13% |
| ee_to_cube_min mean | 0.139 | 0.084 | **0.068** | 0.144 |
| ee<5cm 도달율 | 2% | 1% | **31%** | 2% |
| ee<2cm 도달율 | 0% | 0% | 0% | 0% |
| close 시도 % | 9% | 100% | 100% | **0%** |
| ee_dist_at_close | 0.098 | 0.144 | 0.212 | nan |
| cube_vel after close | 0.0003 | 0.337 | 0.156 | nan |

해석:
- **iter 200**: 도달은 하나 close 너무 일찍 또는 close 시 ee 다시 멀어짐 (close_dist 0.21cm)
- **iter 999**: 도달도 못 함 + close 도 안 함 — 정책 완전 collapse

## 5. 핵심 게이트 — ee<5cm 도달율

**Plan 의 핵심 게이트**:
- ≥ 50% → reward fix 효과 확정, Path A
- 20~50% → 부분 효과, fix 추가
- < 20% → reward 만으론 부족

**실제 결과**:
- max 31% (iter 200) → **부분 효과 (20~50%)** 영역
- final 2% (iter 999) → **< 20%**

→ **fix 1 단독 부족 확정**. 그러나 transient improvement 명확히 관찰 → reward 방향 옳음, 다른 fix 와 결합 필요.

## 6. 시연 path 결정 (사용자)

| 결과 (success rate) | 시연 path | 다음 세션 |
|---------------------|-----------|-----------|
| ≥ 25% | Path A | Sim2Real |
| 18~24% | Path B | fix 2 (DAgger) |
| **14~17%** | **Path B (BC 메인, PPO 보조)** | **fix 2/3** |
| < 14% | Path C | Sim2Real |

success 14% → **14~17% 영역 — Path B (BC 메인, PPO 보조)** 또는 Path C 사용자 결정.

PPO 의 도달율 31% 가 시연 자료로 일부 가치 있음 (도달 능력은 입증). 단, success 14% 면 BC 와 동등.

## 7. 영상 자료

`~/jabis_sim/day5/videos/session14/`:
- `iter200/` — sweet spot 영상 (도달 시도 패턴)
- `iter999/` — collapse 영상 (close 안 함)
- 12 mp4 (top + diag × 3 seed × 2 iter)

시연 메시지: "iter 200 일시적 도달 회복 → iter 999 collapse" — sharper reward landscape 가 PPO exploration 와 gripper 학습 함께 망가뜨린 시각적 증거.

## 8. 새 가설 (세션 14 결과)

| 가설 | 증거 | 검증 |
|---|---|---|
| **6 reward sharper → exploration 망가짐** | iter 200 spike 후 collapse | std=0.03 (중간) ablation |
| **7 lifting reward 가 close 직접 보상 안 함** | iter 500 후 close 0% | close 별도 reward (object_grasped) 추가 |
| **8 PPO entropy 가 close 행동 잊음** | iter 500 → 0% close | KL/entropy bonus 추가 학습 |

**가장 promising**: 가설 7 — close 자체에 직접 reward (cube 가 ee 가까이 + gripper closed 둘 다 만족 시).

## 9. 다음 세션 권고

**옵션 A**: **Fix 7 (close reward 추가)** + std=0.02 유지 — 1 세션. close 학습 collapse 방지
**옵션 B**: **DAgger** — BC rollout fail 에 oracle relabel. 도달 + close 둘 다 학습. 1~2 세션
**옵션 C**: **Path B 확정** + iter 200 PPO ckpt 시연용으로 활용, sim2real 진입
**옵션 D**: **Path C 확정** (BC + Oracle) + sim2real 진입 — 시간 우선

iter 200 의 31% 도달이 fix 방향 옳음을 입증 → **옵션 A 또는 B 추가 가치 있음**.

## 10. Cleanup Whitelist 준수 ✓

- USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged
- cfg 1줄 수정 (`reaching_object std=0.02`) — `tasks/backups_session14/lift_env_cfg.py.session14_pre` 백업
- 신규 파일: `tasks/diagnose_session14_iter*.csv`, `tasks/session14_*.txt`, `tasks/reward_landscape.png`, `tasks/analyze_session14.py`, 12 mp4 영상
