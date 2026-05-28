# G+ Design — Narrow Curriculum + RL-Friendly Reward — 2026-05-19

목표: 오늘 SACfD 5 fail의 본질 진단 (sparse 5조건 AND + 약한 1-step BC) 후 환경/reward 자체를 RL 친화적으로. 진짜 RL 차별화 우선 (결과 시간 vs 결과 보장 trade-off).

## 1. 진단 (오늘 5 fail 통합)

| 시도 | 결과 |
|---|---|
| SACfD orig (bc_w=1.0) | lock-in fail |
| A (bc_w=0.3, anneal 50k) | lock-in fail |
| C' (no BC, HER) | random actor → HER 무용 |
| C'' (BC + no BC loss + HER) | HER success 0 |
| D (C'' + lift dense) | dense 일부 작동, HER success 0 |

본질: 
- 1-step BC actor 3% (chunked actor_mean도 5%)
- 5조건 AND는 actor가 4 sub-skill 동시 학습 필요 → RL exploration으로 첫 success 도달 불가능
- HER이 4 goal-independent 조건 못 만듦
- Sparse +1 reward는 actor가 모든 sub-skill 학습 후에만 신호 받음

## 2. 핵심 결정: 환경/reward 자체를 RL이 풀 수 있게

| 변경 | 동기 |
|---|---|
| Cube spawn 5cm → **2cm narrow** | demo 일관성 ↑ → BC 시작점 ↑ → RL 시작점 ↑ |
| **Sub-task curriculum** (lift-only → lift+place → full PnP) | RL이 각 단계 reward 받으며 점진 학습 |
| Phased dense reward | sub-task별 dense gradient + global sparse +1 유지 |

## 3. Curriculum 재설계

기존 `CurriculumCfg.side_length_m = (0.05, 0.10, 0.15, 0.20)` — cube xy spawn radius 단계. G+는 두 축으로 확장:

```python
@dataclass
class CurriculumCfg:
    # Axis 1: cube xy spawn (narrower → wider)
    side_length_m: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)  # 0.02 추가
    current_stage_idx: int = 0  # 디폴트 stage 0 = 2cm

    # Axis 2: task complexity (sub-task 분해)
    task_level: int = 0  # 0=lift-only, 1=lift+place, 2=full PnP

    advance_eval_episodes: int = 100
    advance_success_rate: float = 0.7
```

### Sub-task 분해 의미

| task_level | reward 조건 | eval 정의 |
|---|---|---|
| 0 | **lift_history만** | cube z ≥ 8cm for 20 step. Place/release/vel/visual 무시. |
| 1 | lift_history AND stable_placement | 2조건 AND. release/vel/visual 무시. |
| 2 | lift + place + release + vel + visual (현재 5조건) | 풀 PnP. |

### Sub-task progression (Codex REVISE 3, 5 반영 — 숫자 closure)

각 (level, stage_idx) 셀에서:

| Gate | 값 |
|---|---|
| Eval episode 수 | 100 |
| 최근 K eval (rolling) | 50 |
| Advance success rate | **≥ 70%** (rolling 50ep) |
| Min env steps per cell | **200k** (이 안에 reach 못 하면 rollback) |
| Max env steps per cell | **800k** (도달 못하면 stall) |
| Rollback condition | stall = 800k step 도달 + advance < 70% |
| Replay 유지 | **이전 stage_idx demos 50% 비율로 mix** (sim2real overfit 회피, Codex 6번) |

### Level 간 policy transition (Codex REVISE 6번 차단 이슈)

**핵심 위험**: level 0 lift-only policy가 hover/awkward grasp 습관 → level 2 진입 시 깨짐. 대응:

1. **Actor logstd unfreeze**: Level 전이 시 logstd를 -1.0으로 reset (exploration 재시작). 학습된 mean_action은 보존.
2. **Demo replay 비중 일시 증가**: Level 전이 직후 first 50k step demo_ratio = 0.5 (보통 0.25), demo policy로 anchor.
3. **이전 level eval guard**: Level k+1로 advance 시 level k 정의의 success rate 검증. K 정의 성공률 < 50% 떨어지면 → 진입 안 함, 추가 학습.

이는 level 간 transition을 monitor + safety net으로 닫음.

### Progression flow
- task_level 0, stage_idx 0 (2cm) → advance criterion → task_level 0, stage_idx 1 (5cm) → … → stage_idx 4 (20cm)
- task_level 0 전체 클리어 → **level transition (위 안전장치)** → task_level 1, stage_idx 0 → 4
- task_level 1 전체 클리어 → transition → task_level 2, stage_idx 0 → 4

각 cell에서 RL이 자체 학습 (sparse reward achievable, online success 자체 수집).

## 4. Reward 재설계

### 4.1 task_level별 reward 함수 (Codex REVISE 2 반영 — hover attractor 차단)

```python
def reward_at_level(self, level: int, report: SuccessReport, step: int) -> tuple[float, bool]:
    """Returns (reward, force_terminate).

    Codex review (2026-05-19) 2번 차단 이슈:
    level 0 = lift_history only → cube hover 학습 위험.
    안전장치:
      (a) Force-terminate AFTER lift latched + max_post_lift_steps (e.g. 30 step).
          Policy가 lift만 하고 hover 무한 못 함.
      (b) Lift bonus는 episode당 단 한 번 +1 (sparse). Per-step bonus 아님.
      (c) Place approach bonus (level 0에서도 미약): -|cube_xy - goal_xy| * 0.005 (small).
          Lift 후 hover보다 place로 이동하도록 유도. Bounded.
    """
    if level == 0:
        sparse = float(report.per_condition["lift_history"] and step <= self._lift_seen_step + self._max_post_lift_steps)
        # Force terminate: lift latch 후 max_post_lift_steps 지나면 episode 끝.
        force_term = (
            report.per_condition["lift_history"]
            and step > self._lift_seen_step + self._max_post_lift_steps
        )
        return sparse, force_term
    if level == 1:
        return float(
            report.per_condition["lift_history"]
            and report.per_condition["stable_placement"]
        ), False
    return float(report.success), False  # level 2: full 5조건 AND
```

- `_lift_seen_step`: lift_history latch된 step. Reset 시 -1.
- `_max_post_lift_steps`: 30 step (=1.5s @20Hz). Lift 후 1.5초 안에 place 단계 진입 안 하면 episode 종료.
- Place approach bonus: dense_reward의 `w_cube_goal_dist` 매우 작게 (0.01 정도) 켜둠. Bounded·annealed 안전 contract 유지.

### 4.2 Phased dense (optional)

task_level별 dense weights을 cfg에 두고 자동 선택:

```python
def dense_weights_for_level(level: int) -> dict[str, float]:
    if level == 0:
        return {"w_ee_cube": 0.5, "w_lifted": 0.10, "w_cube_goal": 0.0, "w_released": 0.0}
    if level == 1:
        return {"w_ee_cube": 0.3, "w_lifted": 0.05, "w_cube_goal": 0.5, "w_released": 0.0}
    return {"w_ee_cube": 0.2, "w_lifted": 0.03, "w_cube_goal": 0.3, "w_released": 0.3}
```

### 4.3 Reward hacking 안전장치

- Sparse component는 task_level에 맞춰 진짜 sub-task 만족 시에만 +1
- Dense는 bounded·annealed (현재 codebase 안전 contract)
- per-condition logging은 그대로 유지

## 5. Oracle / Demo

기존 `scripts/collect_demos.py`는 stage_idx 통해 cube spawn radius 선택. G+에서 추가로 task_level도 환경에 전달 — oracle은 항상 full PnP 시도하지만 reward는 task_level 기준으로 기록.

**중요**: oracle은 task_level에 무관. 항상 full PnP 시도. demo가 task_level 0/1/2 모두에서 success로 마킹됨 (full PnP가 sub-task의 superset). 즉 같은 demo가 모든 level에서 사용 가능.

Demo 재수집 (Codex CAUTION 4 반영 — staged collection):

| 단계 | Demo 수 | 시간 |
|---|---|---|
| Stage 0 narrow (2cm) — level 0 부트 | **500 success ep** | 2시간 |
| Stage 0/1 (2cm + 5cm) — level 1/2 위해 | **+1000 success ep** | 4-5시간 (학습 중 병렬 가능) |
| Level 2 진입 시 stage 1/2 추가 | **+500-1000 ep** | 학습 중 fail mode 보고 결정 |

총 ~2000-2500 ep 단계별. 첫 단계 500이면 Q-net 부트스트랩에는 충분 (DDPGfD original도 500-1000).

## 6. BC retrain

기존 `scripts/train_bc.py --no-chunking` 그대로. Demo만 narrow stage 0으로 교체.

Gate: stage 0 narrow eval (deterministic, 1-step) ≥ **50%**. 이전 3%보다 17배. 좁은 분포에서 oracle action 학습이 더 쉬워 달성 가능 예상.

## 7. SACfD with G+

Task-level 0부터 시작:
1. `--task-level 0` → reward = lift_history. SAC가 lift 학습.
2. Stage 0 narrow eval ≥ 70% → task_level 1 advance.
3. `--task-level 1` → reward = lift + place. SAC가 transport/place 학습.
4. Same advance criterion → task_level 2 (full PnP).

각 task_level에서 sparse reward가 actually achievable (lift만 만족하면 +1) → RL이 reward 받으며 학습 가능. **이게 진짜 RL paradigm.**

## 8. Sub-task curriculum이 진짜 RL인 이유

| | C''/D | G+ |
|---|---|---|
| reward 도달 가능성 | 5조건 AND, BC 약함 시 불가능 | task_level별 1-2 조건만, RL이 직접 학습 가능 |
| Q-net signal | demo only | demo + online success (자체 수집) |
| Actor improvement | BC 시작점에서 거의 안 움직임 | RL이 sub-task 학습 → 단계적 향상 |
| 차별화 | BC anchor가 95% | RL이 단계적 학습이 50%+ |

## 9. 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `cfg.py` CurriculumCfg | `side_length_m` 0.02 추가, `task_level` 필드 |
| `env.py` step() | reward 계산을 task_level별 분기 |
| `cfg.py` DenseRewardCfg | task_level별 dense weight helper (optional) |
| `collect_demos.py` | 변경 없음 (oracle은 full PnP) |
| `train_bc.py` | 변경 없음 |
| `train_sac.py` | `--task-level` CLI 추가 |
| `eval_policy.py` | `--task-level` CLI 추가 |

## 10. 가드 + Rollback (Codex REVISE 5 반영 — 숫자 closure)

### 가드
- per-condition logging (5조건 각각의 만족률) 그대로
- video_sampler 매 10k step (hover/굴리기 패턴 모니터)
- obs_alignment + obs_distribution 그대로
- **Level 0 추가 가드**: episode 마다 (lift event, hover_steps_post_lift) 메트릭 로깅. hover_steps_post_lift > 60% episode 발생 시 force_terminate 발화 확인.

### Rollback (cell별 숫자)

| 상황 | 게이트 | Rollback |
|---|---|---|
| Level 0, stage 0 (2cm) — 200k step 안에 SR ≥ 30% | 미달 시 | curriculum 더 좁힘 (1.5cm) + 추가 demo 500ep |
| Level 0, 어느 stage — 800k step 안에 SR ≥ 70% | 미달 시 | stage rollback (이전 stage 더 학습) + dense weight 조정 |
| Level 0 → 1 transition — level 0 eval ≥ 70% maintained | 미달 시 | level 0 stage 더 학습. transition 보류 |
| Level 1 학습 — release/retreat per-condition rate < 10% (학습 진행 중) | 미달 시 | level 1.5 추가 도입 (lift+place+release만, vel/visual 빼고) |
| Level 2 full — 모든 stage 클리어해도 SR < 30% | 미달 시 | paradigm 재평가, handoff |

### Hover attractor 감시 (Level 0 전용)

Episode 끝에 다음 metric 로깅:
- `lift_seen_at_step`: lift latch step (none이면 -1)
- `hover_steps_post_lift`: lift latch 후 cube z ≥ lift_z인 step 수
- `cube_xy_drift_post_lift`: lift 후 cube xy 이동 거리 (총)

조기 진단:
- `hover_steps_post_lift > 50` AND `cube_xy_drift_post_lift < 5cm` AND eval episode 50% 이상이면 hover attractor 발생. 즉시 학습 중단 + 진단.

## 11. 시간 estimate

| 단계 | 시간 |
|---|---|
| Env code changes | 1-2h |
| Demo re-collect (2cm narrow, 1500 success) | 5-7h |
| BC retrain + gate | 30min |
| SACfD task_level 0 smoke + full | 4-6h |
| SACfD task_level 1 smoke + full | 4-6h |
| SACfD task_level 2 smoke + full | 4-6h |

총 ~24-30시간. 사용자 "비용 OK" 입장에 부합.
