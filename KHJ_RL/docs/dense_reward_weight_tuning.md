# Dense Reward Weight Tuning Matrix (A 옵션 결과 대응)

**목적**: A 옵션 학습 후 결과(특히 30-50% 부분 성공)별로 어떤 weight을 어떤 방향으로 조정할지 사전 정의. 결정 시간 30분 → 1분으로 단축.

**사전 조건**: `docs/dense_reward_design.md`에 정의된 수식 + weight를 베이스라인으로 가정.

```
r_dense(t) = - 0.5  · d_EE_cube_t                                       # w_ee_cube_dist
             + 0.03 · I(lifted_t)                                       # w_lifted_bonus
             - 1.0  · min(d_cube_goal_xy_t, 0.1) · I(lifted_t)          # w_cube_goal_dist
             + 0.5  · I(released_t) · I(near_goal_t) · I(stable_t)      # w_released_bonus

α(t) = 0.01 · max(0, 1 - step / 300_000)
```

---

## 결과 시나리오별 진단 + 처방

### 시나리오 A: 성공률 ≥ 70% (PASS)

- **진단**: A 작동, curriculum stage 0 통과
- **처방**: **weight 조정 없음**. CurriculumCfg stage 1 (10cm spawn) advance
- **다음 단계**: cfg.randomization의 cube spawn xy range를 10cm로 확대 후 추가 학습

---

### 시나리오 B: 성공률 50-70% (NEAR PASS)

- **진단**: Dense 신호가 학습을 끌어올렸으나 마지막 한 발 부족
- **처방 (1순위)**: **추가 학습** — 같은 weight로 300k step 더. anneal은 이미 0이라 sparse만으로 미세 조정 기간
- **처방 (2순위, per-condition 패턴별)**:

| per-condition 패턴 | weight 조정 |
|---|---|
| lift_history 70%+ but release_retreat < 40% | `w_released_bonus 0.5 → 0.7`, `near_goal_radius_xy 0.06 → 0.08` (조건 완화) |
| velocity_stability < 80% | cube 굴리기 의심 — 영상 확인. 굴리기 확인 시 `w_cube_goal_dist 1.0 → 0.5` (saturate 강화) |
| visual_agreement < 60% | (드물지만) ee가 카메라 시야 가림. `w_ee_cube_dist 0.5 → 0.3` (lift 후 EE 분포 다양화) |

---

### 시나리오 C: 성공률 30-50% (PARTIAL)

- **진단**: Dense 신호가 효과 있지만 본질 학습 부족. 또는 hover/local optimum
- **처방 (per-condition 회귀 검사 우선)**:

| 진단 신호 | 처방 |
|---|---|
| lift_history >> BC 36% but stable_placement < 30% | "들고 떨어뜨림" — `w_cube_goal_dist 1.0 → 1.5`, `dist_cap 0.1 → 0.15` (cap 완화, 학습 신호 풍부) |
| lift_history ≈ BC 36% but release_retreat 거의 같음 | 학습 정체 — **decay_steps 300_000 → 400_000** (anneal 늦추기, dense 신호 더 오래) |
| lift_history < BC 36% | **hover/짓누르기 attractor** — `w_lifted_bonus 0.03 → 0.01`, `w_ee_cube_dist 0.5 → 0.3` (접근 신호 약화) |
| 모든 condition 비슷한 회귀 | dense가 BC 분포 망가뜨림 — `α_0 0.01 → 0.005` (전체 신호 절반) |

---

### 시나리오 D: 성공률 < 30% (FAIL)

- **진단**: Dense 도입이 도움 안 됐거나 정책 망가뜨림
- **Rollback 트리거**:
  - lift_history < BC baseline(36%)의 80% (=29%) → dense가 lift 학습 망가뜨림
  - 영상에서 reward hacking (cube 굴리기, hover, trivial release)
- **처방**:
  1. **rollback**: `cfg.dense_reward.enabled = False`, 같은 BC ckpt로 sparse-only 재학습 (D 옵션 그대로)
  2. **D 옵션 결과보다 낮음** (D 0% 대비 < 5%): 즉시 다음 카드로 — `docs/action_chunking_design.md` 또는 `docs/dagger_design.md`
  3. **D 옵션 결과보다 살짝 높음** (5-30%): A 단독 한계, ACT 도입 검토 (BC 자체 강화 필요)

---

## Weight 조정 시 일반 가이드

| 항목 | 안전 한계 | 위험 신호 |
|---|---|---|
| `α_0` 증가 | 0.02 까지 | sparse 신호 압도 가능성 |
| `w_lifted_bonus` 증가 | 0.05 까지 | hover attractor 복귀 |
| `w_cube_goal_dist` 증가 | 1.5 까지 | cube 굴리기 hack 시도 ↑ (영상 확인 필수) |
| `w_released_bonus` 증가 | 1.0 까지 | trivial release 시도 ↑ |
| `dist_cap` 증가 | 0.15 까지 | 너무 멀면 학습 신호 노이즈 |
| `decay_steps` 증가 | 500_000 까지 | pure sparse 검증 기간 부족 |

**Codex 검토 (2026-05-17) 핵심 제약**:
- `w_lifted_bonus` 0.1로 되돌리지 말 것 (hover attractor 보고됨)
- `w_released_bonus` 단순 release만으로 활성화 X (3중 조건 유지)
- `w_cube_goal_dist`에 saturate (dist_cap) 항상 유지

---

## 재학습 명령 템플릿

각 시나리오별로 train_ppo.py CLI는 동일하고, **cfg.dense_reward 값만 코드에서 수정**.

```bash
# 시나리오 B/C: 일부 weight 조정 후 재학습
# 먼저 src/khj_rl/envs/cube_lift/cfg.py의 DenseRewardCfg defaults 수정
# 그 후:
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
  python scripts/train_ppo.py \
    --bc-ckpt runs/stage0_bc_3k/bc.pt \
    --run-name stage0_ppo_a2 \    # ← 새 run name (이전 결과 보존)
    --total-steps 500000 \
    --lr 5e-5 --ent-coef 0.0 --critic-warmup-iters 5 \
    --dense \
    --video-every-steps 50000

# 시나리오 D: rollback to sparse-only
# cfg.dense_reward.enabled = False 또는 --dense flag 제거
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
  python scripts/train_ppo.py \
    --bc-ckpt runs/stage0_bc_3k/bc.pt \
    --run-name stage0_ppo_sparse_rerun \
    --total-steps 500000 \
    --lr 5e-5 --ent-coef 0.0 --critic-warmup-iters 5 \
    --video-every-steps 50000
    # ← --dense 빼면 sparse only
```

---

## 의사결정 트리 요약

```
A 결과 (runs/stage0_ppo_a/eval_result.json)
  ├─ success ≥ 70% → 시나리오 A (advance to curriculum stage 1)
  ├─ 50-70% ─────→ 시나리오 B (추가 학습 또는 per-cond 패턴별 1-2개 weight 조정)
  ├─ 30-50% ─────→ 시나리오 C (per-cond 회귀 검사 → weight matrix 표 참조)
  └─ < 30% ──────→ 시나리오 D
       ├─ lift < 29% → rollback (sparse only)
       └─ 그 외 → ACT or DAgger (별도 설계 문서 참조)
```
