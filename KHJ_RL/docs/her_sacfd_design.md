# HER + SACfD (Card C') Design — 2026-05-19

목표: A variant 결과 (bc_w=0.054에서도 ep_return=0) 이후 actor BC anchor의 lock-in 문제와 demo distribution 안에서만 학습되는 본질적 한계를 우회. HER이 fail rollout도 reward signal로 변환 → sparse env 정공법.

## 1. 핵심 결정

| 결정 | 이유 |
|---|---|
| Actor: SAC random init (no BC weight transfer) | A에서 BC anchor가 lock-in 원인 확인. random init이 SAC paradigm 안에서 자유로움. |
| Q-net: 3-way replay (online + HER + demo) | Demo는 진짜 success transitions (early Q signal). HER은 fail rollout을 hindsight success로 변환. Online은 actor의 현재 분포. |
| Actor BC loss: **제거** | A에서 lock-in 직접 원인. C'은 actor를 BC로 anchoring하지 않음. |
| target_entropy: -3 | A에서 -3가 -6보다 alpha 보존에 효과 있었음. 유지. |

## 2. 우리 Env에 HER이 잘 맞는 이유 (결정적)

success 5조건 중 **stable_placement 1개만 goal-dependent**:

| # | 조건 | goal에 의존? |
|---|---|---|
| 1 | lift_history (cube z ≥ 0.08, 20 step) | ❌ goal-independent |
| 2 | stable_placement (cube xy in goal radius 3cm) | ✅ goal-dependent |
| 3 | release_retreat (gripper open + EE 6cm 떨어짐) | ❌ goal-independent |
| 4 | velocity_stability (cube vel ≤ 5cm/s) | ❌ goal-independent |
| 5 | visual_agreement (top + wrist 카메라 visibility) | ❌ goal-independent |

→ HER relabel(goal을 cube의 final position으로) 후에는 stable_placement만 자동 충족. 나머지 4조건은 episode 중 실제 행동이 결정. **즉, 일반 PnP env (goal-only reward)와 달리, 우리 env에서는 HER relabel만으로는 success 보장 X — actor가 진짜 lift/release/stop을 해야 함**. Reward hacking 회피 정공법.

## 3. HER Buffer 설계

```python
@dataclass
class HERConfig:
    strategy: str = "future"        # "final" | "future" | "episode"
    k_future: int = 4               # 1 transition 당 추가 relabel 개수
    enabled: bool = True

class HERBuffer:
    """Per-episode storage + on-the-fly relabeling on sample.

    Episodes are stored intact; relabeling happens at sample time so the
    same episode can be sampled with different goals across mini-batches.
    For the "future" strategy, t에서 sampling 시 t' > t 범위에서 새 goal 선택.
    """

    def push_episode(self, transitions):
        """transitions = list of dicts with:
          obs (31,), action (6,), reward float, next_obs (31,), done bool,
          cube_xyz_base (3,), next_cube_xyz_base (3,),
          ee_xyz_base (3,), gripper_open_frac float,
          cube_vel_lin (3,), cube_vel_ang (3,),
          visibility_top bool, visibility_wrist bool.
        """

    def sample(self, batch_size, rng):
        # Per sample:
        # 1. Random transition index
        # 2. With p=k_future/(k_future+1): pick t' > t, set goal_new = cube_xyz[t']
        # 3. With p=1/(k_future+1): keep original goal
        # 4. Recompute obs/next_obs (target_delta_base index 17:20) + reward
        # Return (obs, action, reward, next_obs, done) tensors.
```

## 4. Relabel — obs + reward 재계산

### 4.1 obs relabel
obs는 31-D. goal-dependent 위치는:
- `target_delta_base` (3, index 17:20) = `goal_xyz - cube_xyz`

다른 30개 dim은 goal-independent. Relabel 시 index 17:20만 재계산.

```python
new_target_delta = new_goal_xyz - cube_xyz_base
obs_relabeled = obs.copy()
obs_relabeled[17:20] = new_target_delta
# next_obs도 동일하게 next_cube_xyz_base 기준 재계산
```

**중요 (Codex REVISE 2 반영, 2026-05-19)**: Normalizer는 demo obs **+ HER potential relabel target_delta sample**으로 fit. HER relabeled target_delta는 (cube_xyz[t'] - cube_xyz[t]) 분포 → 원 demo의 target_delta(goal - cube_xyz) 분포보다 작은 magnitude. demo로만 fit하면 HER samples가 normalize 후 unusual scale로 들어감.

**구현**: trainer 부팅 시 normalizer fit 단계:
1. demo episodes 각각에서 random t, t' 쌍 K개 sample (K = demo n_episodes × 4)
2. 각 쌍에서 hypothetical relabel: `new_target_delta = cube_xyz[t'] - cube_xyz[t]`
3. demo 원 obs + 위 hypothetical samples로 obs[17:20] (target_delta_base) 분포 보정
4. 나머지 28 dim은 demo 원 분포 그대로

Phase 1 단순화: K = 50000 sample이면 demo + HER 통합 분포 robust.

### 4.2 reward relabel (Codex REVISE 1 반영 — 2026-05-19)

**원 정의 정확히 재계산**: HERBuffer가 episode-level full trajectory 저장. relabel 시 new_goal 기준 episode 모든 step의 placement bool list 생성 → rolling 20-step window 검사로 place_hold 정확히 재현.

```python
def relabel_episode(ep_transitions, new_goal_xyz, cfg):
    """Per-episode relabel: produces success_per_step[t] following all 5
    conditions with the SAME definitions as env._compute_success.
    """
    T = len(ep_transitions)
    cube_xyz = np.stack([t["next_cube_xyz_base"] for t in ep_transitions])  # (T,3)

    # 1. lift_history (goal-indep, latching): episode 안 어디든 z≥0.08을
    #    lift_hold_steps(20) 연속 충족하면 latch.
    z_above = (cube_xyz[:, 2] >= cfg.success.lift_z_m).astype(np.int32)
    lifted_at = _rolling_all(z_above, cfg.success.lift_hold_steps)  # (T,)
    lifted_latch = np.maximum.accumulate(lifted_at)

    # 2. placement (goal-DEP, 새 goal로): xy radius 안 20-step 연속.
    in_radius = (
        np.linalg.norm(cube_xyz[:, :2] - new_goal_xyz[:2], axis=1)
        <= cfg.goal.radius_xy_m
    ).astype(np.int32)
    place_hold = _rolling_all(in_radius, cfg.success.place_hold_steps)

    # 3. released_retreat (goal-indep): episode 안에 저장된 ee/gripper로 재계산
    released = np.array([t["released_state"] for t in ep_transitions])

    # 4. vel_stable, 5. visual (goal-indep)
    vel_stable = np.array([t["vel_stable_state"] for t in ep_transitions])
    visual = np.array([t["visual_state"] for t in ep_transitions])

    success_per_step = (
        lifted_latch & place_hold & released & vel_stable & visual
    ).astype(np.float32)
    return success_per_step  # reward[t] = success_per_step[t]
```

`_rolling_all(arr, k)`: 길이 T → boolean (T,) where `out[i] = all(arr[max(0,i-k+1):i+1])`. NumPy로 vectorize. **다른 4조건의 _state도 episode 안에 저장돼 있음** (env step info에서 trainer가 capture).

## 5. Goal Sampling Strategies

| Strategy | 설명 | 적용 |
|---|---|---|
| `final` | episode 마지막 cube 위치를 goal로 | 간단, k=1 |
| `future` | t' > t 범위에서 cube 위치 sample | k=4 표준 (HER 원논문) |
| `episode` | episode 안 어디든 sample | future + 과거 포함 |

**선택**: `future`, k_future=4. HER 원논문 표준. 각 transition에 4개 추가 relabel + 1 원본 = 5배 signal.

## 6. Trainer Architecture

기존 `SACfDTrainer`에 `--no-bc-loss` 모드 + HER buffer 통합.

```python
class SACfDTrainer:
    def __init__(self, ..., her_buffer=None, no_bc_loss=False, demo_replay=None):
        # actor BC anchor 비활성 (no_bc_loss=True)
        # demo_replay: optional (C'에서는 사용, C에서는 None)
        # her_buffer: per-episode rollout 저장

    def train(self):
        for step:
            # env step → push to online + her episode buffer
            # episode 종료 시 her_buffer.push_episode(ep_transitions)
            # if step >= learning_starts:
            #   batch = mix(online, her, demo) by ratios
            #   SAC update without BC loss
```

### Batch 비율 (Phase 1 시작값)
- 0~50k step: **online 0.3 / HER 0.4 / demo 0.3**
- 50k~ step: **online 0.5 / HER 0.3 / demo 0.2** (online 비중 증가)
  - Online 비중을 늘려 actor의 own distribution 안에서 학습 강화
  - Demo는 안전망으로 minority signal 유지

## 7. Env 추가 노출 (relabel용)

env가 step()에서 info dict에 raw state 추가 필요:

```python
info["raw"] = {
    "cube_xyz_base": cube_xyz_obs,           # already in obs index 14:17
    "next_cube_xyz_base": ...,
    "ee_xyz_base": ee_xyz,
    "gripper_open_frac": ...,
    "cube_vel_lin": ...,
    "cube_vel_ang": ...,
    "visibility_top": ...,
    "visibility_wrist": ...,
    "lift_history_state": report.per_condition_state["lift_history"],
    "released_state": report.per_condition_state["release_retreat"],
    "vel_stable_state": report.per_condition_state["velocity_stability"],
    "visual_state": report.per_condition_state["visual_agreement"],
    "current_goal_xyz": goal_xyz,
}
```

대안 (덜 침습적): 기존 obs vector에 이미 `cube_xyz_base`(14:17), `target_delta_base`(17:20)가 있음 — `current_goal_xyz = cube_xyz_base + target_delta_base` 로 derivable. EE/gripper/velocity는 trainer가 episode rollout 동안 별도 list로 누적해 her_buffer에 넘김. info dict는 success_per_condition 이미 있음 (조건 raw bool은 그 안에 있음).

→ **선택**: env 코드 minimal 변경. trainer가 step out에서 obs + info(success_per_condition) + 별도 simulator state query로 모든 raw state 모음.

## 8. Critical Implementation Risk (Codex 검토 반영, 2026-05-19)

1. **HER label correctness — RESOLVED**: 단일-step 근사 폐기. HERBuffer가 episode-level full trajectory 저장 → place_hold 정확히 재계산 (Section 4.2 _rolling_all). 다른 4조건의 _state도 episode 안에 저장돼 정확 재현.

2. **Normalizer shift — RESOLVED**: demo + HER hypothetical relabel sample 통합 fit (Section 4.1).

3. **3-way batch mix over-fit**: HER이 너무 많으면 actor가 HER hindsight 분포로 편향. Mitigation: (a) online 비중 schedule (50k에 0.5로 증가), (b) `q_real vs q_relabeled` 분리 metric으로 over-fit 감시 (Section 10).

4. **A 결과 해석 (path dependence)**: Codex (1) caution — BC anchor가 lock-in root cause인지 100% 증명 안 됨. C' (BC weight transfer 0 + actor BC loss 0)이 그 가설을 separately 검증. fail해도 추가 정보.

5. **Performance**: HER 매 sample마다 4 relabel → numpy vectorize. Episode 단위 _rolling_all 계산은 episode push 시점에 한 번만 (full success_per_step list precompute), sample 시 인덱싱만.

## 9. Phasing

1. **Phase A**: HER + DemoReplay + SAC 통합 구현 (2-3h 추정).
2. **Phase B**: smoke 50k. 게이트: ep_return > 0 첫 발생.
3. **Phase C**: full 300k + video.

## 10. Rollback + 보조 metric (Codex CAUTION 4, 6 반영)

ep_return 단독은 부족. trainer가 매 log step에 다음 metric 모두 기록:

| Metric | 의미 | 게이트 신호 |
|---|---|---|
| `real_ep_return` | 실제 env reward sum | > 0 한 번이라도 = paradigm valid |
| `relabeled_success_rate` | HER buffer에서 relabel 후 success rate | 0.3+ 이상 = HER 작동 |
| `per_condition_state[i]` | 5조건 각각의 hit rate | lift_history > 30% = actor lift 학습 |
| `q_real / q_relabeled` | Q value를 real vs relabeled batch에서 분리 측정 | 비슷 = healthy. q_relabeled >> q_real = HER over-fit |
| `actor_distance_to_demo` | actor mean과 demo action MSE | 너무 가까움 = lock-in (BC 없는데도?) |

**Rollback 게이트 (50k smoke)**:
- `real_ep_return > 0` 한 번이라도 + `lift_history > 20%`: PASS, full로
- `real_ep_return = 0` AND `relabeled_success_rate > 0.3`: PARTIAL — HER 작동하지만 actor 못 옮김. dense reward 보조 검토
- `real_ep_return = 0` AND `relabeled_success_rate < 0.1`: FAIL — HER 자체 작동 X. relabel 정확성 진단

**Fallback Card D**: Dense reward + HER + SAC (phase-gated bounded·annealed, `docs/dense_reward_design.md`). 또는 paradigm 재평가.
