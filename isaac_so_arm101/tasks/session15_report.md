# 세션 15 결과 — Close Reward 추가 + Early Stop (iter 200 stabilization 시도)

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**Fix 7 ablation**: `lift_env_cfg.py:161` `grasp_object` RewTerm 추가 (`gripper_closure_near_object`, std=0.03, closure_target=-0.6, weight=2.0)
**기타 cfg**: 세션 14 (reaching std=0.02) 그대로
**학습**: max_iterations 200 (collapse 방지)

---

## TL;DR

- **Close reward 효과 미흡**: 도달율 31% (s14 iter200) → 21% (s15 iter100, **감소**)
- **Close% 100% 활성** (close 행동 학습됨) 그러나 **timing 잘못** (close_dist 18cm)
- **Success rate 15% (s14 iter200 14% 와 사실상 동일)**
- **모든 핵심 게이트 미달** (success ≥18%, 도달 ≥40%, close_dist 1~5cm)
- **Path C 확정** — sim2real 진입 권고

---

## 1. Fix 7 가설 + 진단 근거

세션 14 데이터:
- iter 200 sweet spot 도달율 31% 후 collapse → close 학습 collapse 가설
- `lifting_object` (weight 75) 가 close 직접 보상 안 함 → close reward 부재

→ 가설: close reward 추가하면 close timing 안정화 + 도달율 유지.

## 2. 사용한 Reward 함수

기존 `gripper_closure_near_object` 활용 (rewards.py:91, plan 의 binary 형 대신 smooth):
```python
proximity = torch.exp(-(distance / std) ** 2)            # std=0.03
closure = torch.clamp(gripper_pos / closure_target, 0, 1) # closure_target=-0.6
return proximity * closure
```

- 5cm 거리: proximity 0.06
- 3cm: 0.37
- 1cm: 0.89

## 3. 학습 progression

| iter | succ% | **ee<5cm%** | ee<2cm% | ee_min | close% | close_dist | cube_v |
|---|---|---|---|---|---|---|---|
| 0 (BC) | 14.0 | 2.0 | 0.0 | 0.166 | 0 | nan | nan |
| 50 | 15.0 | 1.0 | 0.0 | 0.117 | 99 | 0.31 | 0.0003 |
| **100** | **15.0** | **21.0** ⭐ | 0.0 | 0.089 | 100 | 0.18 | 0.54 |
| 150 | 14.0 | 3.0 | 0.0 | 0.111 | 100 | 0.18 | 0.46 |
| 199 | 14.0 | 2.0 | 0.0 | 0.107 | 100 | 0.16 | 0.21 |

**해석**:
- iter 100 sweet spot — 도달율 21% (s14 iter200 의 31% 보다 **낮음**)
- iter 50 부터 close% 99~100 (close reward 가 행동 학습)
- 그러나 close_dist 16~31cm — close timing 학습 실패. cube_v 0.21~0.54 = random push
- iter 100 이후 도달율 다시 떨어짐 (3%, 2%) — collapse 패턴 세션 14 와 유사

## 4. 비교 표 (세션 13/14/15)

| 정책 | succ% | ee<5cm% | close% | close_dist | cube_v |
|---|---|---|---|---|---|
| s13 Oracle | 15.0 | **86.0** | 98 | 0.05 | 0.02 |
| s13 BC | 14.0 | 2.0 | 9 | 0.10 | 0.0003 |
| s13 PPO best | 13.0 | 1.0 | 100 | 0.14 | 0.34 |
| **s14 iter200** | 14.0 | **31.0** | 100 | 0.21 | 0.16 |
| s14 iter999 | 13.0 | 2.0 | 0 | nan | nan |
| **s15 iter100** | 15.0 | 21.0 | 100 | 0.18 | 0.54 |
| s15 iter199 | 14.0 | 2.0 | 100 | 0.16 | 0.21 |

→ 도달율 best: **s14 iter200 31%**. s15 추가 reward 가 도달율 감소 시킴.

## 5. 핵심 게이트 판정

| 게이트 | 목표 | s14 iter200 | s15 iter100 |
|---|---|---|---|
| success ≥ 18% | | 14% ❌ | 15% ❌ |
| ee<5cm ≥ 40% | | 31% ❌ | 21% ❌ |
| close_dist 1~5cm | | 21cm ❌ | 18cm ❌ |

**전부 미달**.

## 6. 가설 7 (close reward 부재) 결과

**가설 부분 확인**:
- close 행동은 즉시 학습됨 (close% 99~100)
- 그러나 close 의 *위치* 학습 못 함 (close_dist 18cm)
- proximity × closure 가 distance > 5cm 에서 사실상 0 → 학습 시 멀리서 close 해도 reward 0 변함 없어 close 위치 학습 incentive 없음
- 또는 lifting weight 75 가 catch 시 큰 보상 → close 위치 부정확해도 catch 시 reward 회수

**결론**: close reward 만으론 timing 학습 부족. 도달능력 (reaching) 와 close timing 둘 다 필요.

## 7. 새 가설 (세션 15 결과)

| # | 가설 | 증거 |
|---|---|---|
| **9 reaching std=0.02 자체가 너무 sharp** | 도달율 max 31% (s14) → 21% (s15) 감소. 추가 reward 가 reaching 학습 방해 | std=0.03 ablation |
| 10 BC saddle 자체가 PPO update 로 deepen | iter 100 후 도달능력 다시 잃음 | BC fresh init (학습 안 함) |
| 11 lifting weight 75 가 catch 보상에 압도적 | close timing 학습 incentive 약함 | lifting weight 30 으로 줄여 ablation |

## 8. 시연 path 결정

| 결과 | path | 다음 세션 |
|---|---|---|
| ≥ 25% AND 도달 ≥ 40% | Path A | sim2real |
| 18~24% AND 도달 ≥ 30% | Path B | sim2real |
| **< 18% OR 도달 < 25%** | **Path C 확정** | **sim2real with BC/Oracle** |

success 15%, 도달 21% → **Path C 확정 영역**.

## 9. 영상

`~/jabis_sim/day5/videos/session15/`:
- `iter100/` — best (도달 + close 시도 시각)
- `iter199/` — final (도달 잃은 후 close 만 함)
- 12 mp4 (top + diag × 3 seed × 2 iter)

## 10. 다음 세션 권고

**Path C 확정** — Sim2Real 진입.

| 옵션 | 시연 자산 |
|---|---|
| **A (권장)** | BC actor (s8 v1) 메인 + Oracle 백업 — 가장 안정 |
| B | + s14 iter200 PPO 보조 (도달 31% 시연 가치) |
| C | RL 추가 ablation (lifting weight 30, 가설 11) — 시간 추가 |

**중요**: 사용자 결정 사항.

## 11. RL 트랙 종료 결정

세션 13 → 14 → 15 검증 후:
- saddle 진짜 원인 식별 ✓
- reward fix 단독 부분 효과 (transient)
- close reward 추가 효과 미흡
- 도달율 max 31% 가 RL 의 한계 (40% 안 넘음)

**결론**: 현재 reward shaping 전략으로 saddle 못 깸. DAgger 또는 demo 수정 같은 큰 fix 필요. 시연 시간 우선 → **RL 트랙 종료**.

## 12. Cleanup Whitelist 준수 ✓

- USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged
- cfg 1 RewTerm 추가 (`grasp_object`) — `tasks/backups_session15/lift_env_cfg.py.session15_pre` 백업
- mdp/rewards.py 수정 0 (기존 `gripper_closure_near_object` 활용)
- 신규: 5 csv + analyze_session14.py + report + 12 mp4 + 2 ckpt
