# Dense Reward Shaping — A 옵션 (도입 결정)

**상태**: 도입 결정 (2026-05-17). Codex 검토 완료, weight 조정 반영. CLAUDE.md 결정사항 5도 갱신됨.

**도입 근거**:
- D 옵션 검증 결과: BC weight 보존만 됐고 정책은 짓누르기 attractor로 변형 → sparse signal 부재가 본질
- B 옵션 결과: demo 3배(2877 success) → BC 19% (15%→19%, +4%p 미미). lift_history 36% / release_retreat 21% — lift→release 단계에서 covariate shift 누적
- 데이터 증강만으로는 본질 해결 불가. **PnP 후반부 학습을 위해 phase-gated dense 신호 도입**

---

## 1. 안전 원칙 (jabis_sim_v2 reward hacking 회피)

| 원칙 | 의도 |
|---|---|
| **Bounded** | dense reward의 한 step 절대값이 sparse +1보다 항상 작도록 weight 캡 |
| **Annealed** | 학습 진행에 따라 weight `α(t) → 0`. 학습 끝엔 pure sparse |
| **Phase-gated** | hack에 취약한 항은 조건부 활성화 (cube 굴리기, hover, trivial release 차단) |
| **Saturating** | 거리 신호를 cap해서 lift 직후 cube far일 때 거짓 신호 없게 |
| **Sparse 메인** | success +1은 항상 dense 합보다 압도적. dense는 "방향만 제시" |
| **GT 기반** | dense 계산은 `SuccessState`(GT pose)만 사용. 노이즈/perception은 obs에만 적용 |
| **logging-필수** | 각 항을 따로 로깅해 reward hacking 의심 시 분리 진단 가능 |

---

## 2. 수식 (Codex 검토 반영, 2026-05-17)

```
r_t = r_sparse(t) + α(t) · r_dense(t)

r_sparse(t) = 1 if (success at t) else 0   # 기존 그대로 유지

r_dense(t) = - w_ee_cube_dist     · d_EE_cube_t
             + w_lifted_bonus     · I(lifted_t)
             - w_cube_goal_dist   · min(d_cube_goal_xy_t, dist_cap) · I(lifted_t)
             + w_released_bonus   · I(released_t) · I(near_goal_t) · I(stable_t)

α(t) = α_0 · max(0, 1 - step / decay_steps)   # 선형 감쇠
```

### 항별 의도 (Codex 검토 반영)

| 항 | 형태 | 의도 | 안전장치 |
|---|---|---|---|
| `d_EE_cube` | 거리 음수 | EE를 cube 쪽으로 유도 | bounded; weight 0.5로 낮춤 (lift bonus 대비 dominant 방지) |
| `lifted_bonus` | binary, **per-step 작게** | lift된 상태 유지 약하게 보상 | weight 0.03 (hover attractor 회피, codex 권고) |
| `cube_goal_dist` | 거리 음수 (saturate) × `I(lifted)` | lift 후 goal로 유도 | **`min(d, 0.1m)` cap**: 멀리 있을 때 거짓 신호 X. `* I(lifted)`: cube 굴리기 hack 차단 |
| `released_bonus` | binary, **3중 조건** | release+near_goal+stable 동시 만족 시 보상 | trivial release hack (cube 위치 무관하게 gripper만 열기) 차단 |

### Codex 검토 (2026-05-17) 핵심 권고 반영

> **가장 큰 risk**: dense reward가 lifted 이후 hover/near-goal 체류를 강화 → release_retreat 21% 병목 해결 못 함.
>
> **Top weight change**: `w_lifted_bonus 0.1 → 0.03` — per-step hover attractor를 줄이고 lift는 sparse/condition success가 주도하게.

추가 권고 반영:
- `w_ee_cube_dist` 1.0 → 0.5 (접근 신호가 lift 신호 dominant 방지)
- `cube_goal_dist`에 0.1m saturate cap (lift 직후 far 상태에서 거짓 신호 X)
- `released_bonus` 조건에 `near_goal AND stable` 추가 (trivial release hack 차단)
- `decay_steps` 500k → 300k (dense bias 빨리 사라지게, 200k는 pure sparse로 검증)
- per-condition rollback 기준 추가

### Weight (최종)

```python
α_0                  = 0.01
w_ee_cube_dist       = 0.5          # ↓ from 1.0
w_lifted_bonus       = 0.03         # ↓ from 0.1 (codex 직접 권고)
w_cube_goal_dist     = 1.0          # 거리는 min(d, dist_cap)로 saturate
w_released_bonus     = 0.5          # ↑ from 0.2 (조건 강화하면서 보상 보강)
dist_cap             = 0.1          # cube_goal_dist saturate 임계
decay_steps          = 300_000      # ↓ from 500_000 — 200k는 pure sparse 검증 기간
```

**Sparse vs dense 신호 크기 비교** (수정 weight 기준):
- α_0 = 0.01에서 r_dense의 한 step 최대 절대값:
  - 접근 항: 0.5 × 0.5 = 0.25 (max EE-cube 0.5m 가정)
  - lift 항: 0.03 (binary 1)
  - cube_goal_dist 항 (saturate): 1.0 × 0.1 × 1 = 0.1
  - released 항 (3중 조건): 0.5 × 1 = 0.5
  - 합 최대: ~0.88 (모든 항 동시 만족 unrealistic, 보수 추정)
- α·r_dense 최대 절대값 ≈ **0.0088**
- r_sparse = +1
- **sparse가 dense보다 ~110배 큰 신호** → 정책은 결국 sparse 최적화로 수렴

---

## 3. cfg.py 추가 (DenseRewardCfg)

```python
@dataclass
class DenseRewardCfg:
    """Phase 1 dense shaping — disabled by default.

    Enable for A-option training (cfg.dense_reward.enabled=True). Codex
    reviewed 2026-05-17, see docs/dense_reward_design.md for safety
    contract.
    """
    enabled: bool = False
    alpha0: float = 0.01
    decay_steps: int = 300_000          # 200k는 pure sparse 학습 기간
    w_ee_cube_dist: float = 0.5
    w_lifted_bonus: float = 0.03
    w_cube_goal_dist: float = 1.0
    w_released_bonus: float = 0.5
    dist_cap: float = 0.1               # cube_goal_dist saturate
    near_goal_radius_xy: float = 0.06   # release 보상 trigger (success place_radius*2)
    velocity_stable_lin: float = 0.05   # cube 선속도 임계 (m/s)
    velocity_stable_ang: float = 0.5    # cube 각속도 임계 (rad/s)
```

---

## 4. 새 모듈 (`src/khj_rl/envs/cube_lift/reward_dense.py`)

수식 + 안전장치를 단일 함수로 모음. `success.py`는 평가용(절대 변경 금지) — dense는 학습용 보조라 책임 분리.

```python
"""Dense reward shaping for Phase 1 cube-lift (A option).

Activated via cfg.dense_reward.enabled=True. Bounded + annealed +
phase-gated; see docs/dense_reward_design.md for the safety contract
and Codex review (2026-05-17).
"""

import numpy as np
from khj_rl.envs.cube_lift.cfg import DenseRewardCfg, GoalCfg, SuccessCfg
from khj_rl.envs.cube_lift.success import SuccessState


def compute_dense_reward(
    state: SuccessState,
    goal: GoalCfg,
    success_cfg: SuccessCfg,
    cfg: DenseRewardCfg,
) -> tuple[float, dict[str, float]]:
    """Returns (total, per_term) for step-level dense reward (pre-anneal)."""
    cube = state.cube_xyz_m
    ee = state.ee_xyz_m

    d_ee_cube = float(np.linalg.norm(ee - cube))
    lifted = float(cube[2] >= success_cfg.lift_z_m)

    goal_xy = np.asarray(goal.pos_xyz_m[:2], dtype=np.float32)
    d_cube_goal_xy_raw = float(np.linalg.norm(cube[:2] - goal_xy))
    d_cube_goal_xy = min(d_cube_goal_xy_raw, cfg.dist_cap)

    # Released = success #3 condition (gripper open + EE retreat).
    released_now = float(
        state.gripper_opening >= success_cfg.gripper_open_threshold
        and d_ee_cube >= success_cfg.retreat_dist_m
    )
    # near_goal/stable additional gates (codex review 2026-05-17)
    near_goal = float(d_cube_goal_xy_raw < cfg.near_goal_radius_xy)
    stable = float(
        np.linalg.norm(state.cube_lin_vel_m_s) < cfg.velocity_stable_lin
        and np.linalg.norm(state.cube_ang_vel_rad_s) < cfg.velocity_stable_ang
    )

    terms = {
        "ee_cube_dist": -cfg.w_ee_cube_dist * d_ee_cube,
        "lifted_bonus": cfg.w_lifted_bonus * lifted,
        "cube_goal_dist": -cfg.w_cube_goal_dist * d_cube_goal_xy * lifted,
        "released_bonus": cfg.w_released_bonus * released_now * near_goal * stable,
    }
    return sum(terms.values()), terms


def dense_alpha(step: int, cfg: DenseRewardCfg) -> float:
    """Linear anneal: cfg.alpha0 at step 0 → 0 at cfg.decay_steps."""
    if cfg.decay_steps <= 0:
        return 0.0
    frac = max(0.0, 1.0 - step / cfg.decay_steps)
    return cfg.alpha0 * frac
```

---

## 5. env.py 후크 (env.py:468 부근)

```python
sparse = float(report.success)

if self.cfg.dense_reward.enabled:
    dense_raw, dense_terms = compute_dense_reward(
        gt_state, self.cfg.goal, self.cfg.success, self.cfg.dense_reward,
    )
    alpha = dense_alpha(self._global_step, self.cfg.dense_reward)
    reward = sparse + alpha * dense_raw
    info["dense_reward_raw"] = dense_raw
    info["dense_alpha"] = alpha
    info["dense_terms"] = dense_terms
else:
    reward = sparse

terminated = bool(report.success)
```

`self._global_step`: 학습 trainer가 매 step 호출 시 갱신 (전체 학습 step 카운터). PPOTrainer 쪽에서 `env._global_step = self._global_step`처럼 동기화하거나, env에 `set_global_step` 메서드 추가.

---

## 6. 로깅 (PPOTrainer)

```python
if "dense_terms" in info:
    for term, val in info["dense_terms"].items():
        self._log_running[f"dense/{term}"] = val
    self._log_running["dense/alpha"] = info["dense_alpha"]
    self._log_running["dense/raw"] = info["dense_reward_raw"]
```

학습 중 진단 신호:
- `dense/alpha`가 step에 따라 0으로 감쇠 (anneal 작동)
- `dense/cube_goal_dist`가 0이다가 음수로 (lift 후 goal 접근 신호)
- `dense/lifted_bonus`가 0에서 0.03쪽으로 (lift 빈도 증가 신호)

---

## 7. 도입 절차 / 검증 (Codex 검토 반영)

**도입 시 체크리스트**:
- [x] codex 리뷰 완료 (2026-05-17) — weight 조정 + saturate + 3중 조건 반영
- [ ] CLAUDE.md 결정사항 5 갱신 (다음 단계)
- [ ] 본 문서를 cross-ref로 CLAUDE.md에 링크
- [ ] success.py 변경 없음 확인 (sparse 신호 unchanged)
- [ ] DenseRewardCfg 디폴트 enabled=False 확인

**학습 후 (dense ON 학습 1회 끝나면)**:
- [ ] 영상 검증: cube 굴리기 / 짓누르기 / 끌고가기 / hover 행동 부재
- [ ] **per-condition 회귀 검사** (codex 권고): 각 조건이 baseline(BC 3k) 이상 유지
  - lift_history ≥ 36%
  - release_retreat ≥ 21%
  - 기타 조건 ≥ baseline
- [ ] α=0 시점(step 300k+) eval로 측정한 성공률이 baseline(BC 19%) 이상
- [ ] α=0 시점 success rate > sparse-only(D 옵션 0%) 이상 — dense 도입 의의 확인

**Rollback 조건** (any one triggers rollback):
1. α=0 시점 overall success rate < BC 19% (dense가 정책 망가뜨림)
2. per-condition 중 어떤 조건이 baseline에서 10%p 이상 회귀 (특히 lift_history)
3. 영상에서 reward hacking 행동 (cube 굴리기, hover, trivial release) 확인

---

## 8. 결정 트리 (현재 진행)

```
A 옵션 도입 (현재 위치)
  ↓
PPO 500k 학습 (BC 3k ckpt + dense ON)
  ↓
α=0 시점 eval + per-condition 회귀 검사
  ├─ 성공 ≥ 50% AND 모든 condition baseline 이상 → A 성공, 본격 진행
  ├─ 성공 30-50% AND 조건 회귀 < 10%p → 부분 성공, weight 튜닝
  └─ 성공 < 30% OR 큰 회귀 → Rollback, B+A 외 카드 검토
                                 ├─ DAgger 라운드 (oracle로 표류 보강)
                                 └─ Action chunking (covariate shift 직접 해결)
```

---

## 9. 미해결 / 후속 결정

- **`released_bonus` 조건의 정확한 임계값**: `near_goal_radius_xy=0.06` (success place_radius_xy=0.03 의 2배). 너무 빡빡하면 안 발동, 너무 너그러우면 hack 가능. 학습 후 발동 빈도 봐서 조정
- **`dist_cap=0.1m`**: 너무 작으면 멀리서 신호 없음, 너무 크면 hack. 100ep eval 후 cube 평균 초기 거리 보고 조정 가능
- **dense ON 학습 후 BC 19%와 비교**: dense가 BC 분포 자체를 망가뜨리지 않게 lr=5e-5 유지 (D 옵션 hyperparam 그대로)
