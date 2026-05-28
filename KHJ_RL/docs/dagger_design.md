# DAgger (Dataset Aggregation) — 사전 설계

**상태**: 설계만. A 옵션 + ACT 둘 다 실패할 때의 마지막 카드.

**도입 근거 (조건부)**:
- A (dense shaping) + D (action chunking) 둘 다 50% 미만
- 핵심 진단: BC가 oracle 분포에서 *어느 시점에 표류 시작하는지* 알 수 있는데 그 obs 영역에 대한 oracle action이 demo에 없음
- DAgger는 표류 지점을 직접 학습 데이터에 보강 — covariate shift의 *근본 처방*

---

## 1. DAgger 핵심 아이디어

```
Iteration:
  1. 현재 BC policy로 sim rollout (1000 ep 같은)
  2. 매 step의 obs를 기록
  3. 그 obs에 oracle (motion planning) 을 query → expert action
  4. (obs, expert_action) pair를 demo dataset에 추가
  5. 보강된 dataset으로 BC retrain
  6. 반복
```

핵심: BC가 표류한 곳 (= covariate shift 일어난 obs)에 oracle이 "이런 상태에선 이렇게 행동"이라고 보강. 1-step BC의 누적 오차를 점진적으로 메움.

### 우리 환경에서 oracle query 가능성

- `MotionPlanningOracle` (oracle.py)이 **state 기반** — joint, cube pose, goal pose만 알면 action 계산. obs noise 없는 GT 사용.
- 즉 *임의 obs*에서도 oracle을 깔끔하게 query 가능. 이게 DAgger 적용 가능성의 핵심.
- 단점: oracle 자체가 phase-aware (descend → grasp → lift → ...). DAgger는 obs만 가지고 query하니까 oracle이 *그 obs에서 어느 phase에 있어야 하는지* 다시 판단해야 함. 이미 oracle은 cube 위치 기반 phase 전환이라 *대부분* 작동.

---

## 2. 안전 원칙

| 원칙 | 의도 |
|---|---|
| **β-mixing** (DAgger 원논문) | 초기 iter는 oracle action 비중 ↑, 후기는 BC action 비중 ↑ |
| **표류 지점만 보강** | 매 step 추가하면 데이터 폭발. BC와 oracle action의 *차이* 큰 step만 추가 |
| **GT obs** | 사용자 정책 obs는 noise 포함, oracle query는 GT obs로 |
| **Oracle phase 일관성 검증** | oracle이 임의 obs에서 phase 잘 판단하는지 sanity (state.cube_xyz_m이 갑자기 멀어진 경우 등) |

---

## 3. DAgger Round 구조

```
Round 0:
  - 기존 BC ckpt (stage0_bc_3k/bc.pt, 19% baseline)
  - 기존 demo (964 + 1913 = 2877 success demo)

Round 1:
  - BC ckpt로 sim rollout 1000 ep
  - 매 step 다음 기록:
      (obs, BC_action, oracle_action, |BC - oracle|)
  - |BC - oracle| > threshold인 step만 새 demo 데이터로 저장
  - 기존 demo + 새 demo로 BC retrain

Round 2..N:
  - Round 1 BC로 다시 rollout, 보강, 재학습
  - 보통 3-5 round면 수렴
```

### β-mixing schedule (BC vs oracle action 비율)

```python
def beta(round_idx, max_rounds=5):
    """초기 round는 oracle 비중 ↑, 후기는 BC 비중 ↑."""
    return max(0.0, 1.0 - round_idx / max_rounds)
```

Rollout 시:
```python
action = beta(r) * oracle_action + (1 - beta(r)) * bc_action
```

단, action space [-1, 1] 정규화돼 있어 weighted sum이 의미 있음 (joint delta + gripper 모두 작은 변동).

---

## 4. 표류 감지 — `|BC - oracle|` threshold

매 step 다 보강하면 데이터 크게 늘어남. 표류 지점만 선택:

```python
diff = np.linalg.norm(bc_action - oracle_action)  # 6-D action vector
if diff > drift_threshold:
    add_to_demo(obs, oracle_action)
```

`drift_threshold` 디폴트: **0.1** (action 정규화 [-1, 1] 기준, ~5% 차이). 너무 작으면 데이터 폭발, 너무 크면 미세 표류 못 잡음.

---

## 5. 코드 후크

### 새 스크립트: `scripts/collect_dagger.py`

기존 `collect_demos.py` 변형. 차이:
- Oracle만 돌리지 않고 BC policy도 함께 query
- β-mixing으로 action 결정
- (obs, oracle_action) 저장 시점은 drift_threshold 통과한 step만

```python
def main():
    env = CubeLiftEnv(cfg)
    oracle = MotionPlanningOracle(...)
    bc_net = load_bc(bc_ckpt)
    normalizer = ObsNormalizer.load(...)

    for ep in range(episodes):
        obs = env.reset()
        obs_buf, action_buf = [], []
        for t in range(max_steps):
            oracle_action = oracle.act(obs, info)
            with torch.no_grad():
                bc_action = bc_net.actor_mean(
                    normalizer.normalize(obs)[None]
                ).squeeze(0).numpy()
            mixed = beta * oracle_action + (1 - beta) * bc_action
            mixed = np.clip(mixed, -1, 1)

            # Record (obs, oracle_action) if BC drifted
            if np.linalg.norm(bc_action - oracle_action) > threshold:
                obs_buf.append(obs.copy())
                action_buf.append(oracle_action.copy())

            obs, reward, term, trunc, info = env.step(mixed)
            if term or trunc:
                break
        # save if any drift point recorded
        if obs_buf:
            save_npz(ep, obs_buf, action_buf, success_flag=info["success"])
```

### Pipeline 통합

`scripts/auto_phase1_pipeline.py`를 DAgger 지원 mode로 확장 또는 별도 `scripts/auto_dagger_loop.py`:

```
loop {
    1. collect_dagger.py (1000 ep)
    2. train_bc.py (기존 + 새 demo merged)
    3. eval_policy.py
    4. if success_rate >= threshold: break
    5. else: round_idx++, repeat
}
```

---

## 6. 도입 트리

```
A 결과 (per docs/dense_reward_weight_tuning.md 결정)
  ├─ ≥ 50% → A 성공, DAgger 불필요
  └─ < 50% AND covariate shift 패턴
       ↓
     ACT 시도 (docs/action_chunking_design.md)
       ├─ ≥ 50% → ACT 성공
       └─ < 50%
            ↓
          DAgger 도입
            ├─ Round 1 (β=0.8, BC가 표류한 곳 oracle 보강)
            ├─ Round 2 (β=0.6)
            ├─ Round 3 (β=0.4)
            ├─ Round 4 (β=0.2)
            └─ Round 5 (β=0.0, 순수 BC retrain)
```

---

## 7. 검증 / Rollback

- 매 round 끝 eval 100ep success rate가 이전 round보다 ≥ 5%p 향상해야
- 5 round 이후도 < 50%면 DAgger 자체로 부족 → demo 품질 문제 (oracle 자체가 표류 지점 잘 처리 못 함) 또는 model capacity 부족

Rollback 조건:
1. 어떤 round에서 직전 round보다 5%p 이상 하락 → 그 round의 보강 데이터가 noisy. 이전 round로 복귀
2. drift_threshold 통과한 step이 episode당 1개도 없음 (BC가 거의 oracle 따라감) → DAgger 의미 없음, 다른 카드로

---

## 8. 시간 추정 (1 round 기준)

- collect_dagger.py 1000 ep: ~3시간 (oracle + BC inference라 collect_demos보다 느림)
- BC retrain: ~30분-1h
- Eval: ~10분
- **1 round = ~4시간**
- 5 round 다 돌리면 **~20시간** (병렬 안 됨, sequential)

→ DAgger는 비싼 카드. A/ACT 둘 다 명백히 실패할 때만 적용.

---

## 9. 미해결 / 후속 결정

- **drift_threshold**: 0.1 디폴트. 보강 ep당 step 수 보고 조정 (50% 이상 → 너무 너그러움, 5% 이하 → 너무 빡빡)
- **β schedule**: 선형 감쇠 (1.0 → 0.0 over 5 rounds). exponential이나 cosine도 가능
- **Oracle phase 안정성**: 표류된 obs에서 oracle이 잘못된 phase 선택할 가능성. round별 sanity 영상 검증 필수
- **새 demo의 success_flag**: drift 보강한 episode가 결국 success 못 해도 oracle action은 유효. success_only filter 빼고 학습할지 결정 필요
