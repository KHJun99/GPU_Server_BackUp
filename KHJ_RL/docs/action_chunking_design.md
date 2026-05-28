# Action Chunking (ACT-style) — 사전 설계

**상태**: 설계만. A 옵션(dense shaping) 결과 < 50% AND 영상에서 covariate shift 무너짐 확인 시 도입 검토.

**도입 근거 (조건부)**: BC 자체가 더 강해져야 PPO가 안정 학습 가능. 데이터 증강(2.5x)으로 BC 15→19% 미미한 향상 → 1-step prediction 자체의 한계. ACT는 covariate shift를 *모델 구조 차원에서* 차단.

---

## 1. ACT (Action Chunking Transformer) 핵심 아이디어

| 항목 | 1-step BC (현재) | k-step ACT |
|---|---|---|
| Prediction | 다음 action 1개 | 다음 k개 action 전체 (k=10~16) |
| Loss | MSE on action(t) | L1 on action(t..t+k-1) |
| Inference | 매 t마다 1개 출력 | 매 t마다 k개 출력, **temporal ensembling** |
| Covariate shift | 누적 오차로 표류 | k step 앞으로 행동 계획해 표류 차단 |

### Covariate shift가 ACT로 차단되는 이유

1-step BC는 각 t마다 actor가 obs(t) → action(t)만 학습. policy가 oracle 분포에서 미세하게 벗어나면 다음 obs(t+1)도 분포 밖, 그 다음은 더 큰 분포 밖... 누적.

ACT는 1번의 prediction이 k step을 미리 그림 — 그 안에서는 actor가 "표류해도 plan은 정확". temporal ensembling으로 여러 prediction의 평균을 사용해 추가 안정화.

---

## 2. 안전 원칙

| 원칙 | 의도 |
|---|---|
| **PPO와 호환** | ACT 모델도 같은 ActorCritic 인터페이스(actor + critic). PPO transfer 보장 |
| **k는 horizon보다 짧게** | k=12 (160 horizon의 7.5%). 너무 길면 distribution shift 자체가 plan 안에 들어옴 |
| **temporal ensembling 가중** | 가까운 prediction에 더 큰 weight (e^{-α·offset}). hover나 stale plan 회피 |
| **GT-action 학습** | 노이즈 영향 받지 않은 oracle action으로만 학습. obs는 noise 포함 (sim2real) |

---

## 3. 모델 구조

### 입력 / 출력 형태

```
input:  obs (B, T, obs_dim)        # T = context window (보통 1, 즉 single-step obs)
output: action (B, k, action_dim)  # k = 12 (chunk length)
```

### 네트워크 (간단 버전 — Transformer encoder-decoder 대신 MLP+chunked head)

```python
class ChunkedActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, chunk_size=12, hidden=256):
        self.backbone = MLP(obs_dim, hidden, hidden)
        self.actor_head = nn.Linear(hidden, action_dim * chunk_size)  # k action 동시
        self.critic = nn.Linear(hidden, 1)
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -1.0))
        self.chunk_size = chunk_size

    def actor_mean(self, obs):  # for PPO transfer compat
        feat = self.backbone(obs)
        flat = self.actor_head(feat)
        chunks = flat.view(*flat.shape[:-1], self.chunk_size, action_dim)
        return chunks[..., 0, :]  # 첫 step만 (PPO compat)

    def actor_chunks(self, obs):  # for ACT BC training
        feat = self.backbone(obs)
        flat = self.actor_head(feat)
        return flat.view(*flat.shape[:-1], self.chunk_size, action_dim)

    def value(self, obs):
        return self.critic(self.backbone(obs)).squeeze(-1)
```

**핵심 트릭**: `actor_mean`은 chunk의 첫 action만 리턴해서 PPO ActorCritic 인터페이스 그대로 호환. BC 학습 때만 `actor_chunks` 사용.

### Loss (BC 학습)

```python
chunk_pred = net.actor_chunks(obs[t])     # (B, k, action_dim)
chunk_target = action[t : t+k]            # (B, k, action_dim)
loss = F.l1_loss(chunk_pred, chunk_target)
```

L1 (Mean Absolute Error) 이유: outlier oracle action에 robust (oracle도 가끔 jitter 있음).

---

## 4. Inference — Temporal Ensembling

핵심 ACT 트릭. 매 step t마다 k개 action 예측인데, 실제 실행은 1개만. 다음 step에서 또 k개 예측 — 이제 같은 timestep에 여러 prediction이 누적.

```python
class ChunkBuffer:
    """At step t, store last k chunks. action(t) = weighted avg of
    predictions that targeted t."""
    def __init__(self, k):
        self.k = k
        self.buf = []  # list of (origin_step, chunk_pred)

    def push(self, t, chunk):  # chunk: (k, action_dim)
        self.buf.append((t, chunk))
        self.buf = [(o, c) for o, c in self.buf if t - o < self.k]

    def get_action(self, t, alpha=0.1):
        contribs = []
        for origin, chunk in self.buf:
            offset = t - origin
            if 0 <= offset < self.k:
                w = np.exp(-alpha * offset)
                contribs.append((w, chunk[offset]))
        if not contribs:
            return None
        total_w = sum(w for w, _ in contribs)
        return sum(w * a for w, a in contribs) / total_w
```

**효과**: 새 prediction(weight 1.0)과 옛 prediction (weight e^{-α·offset})이 평균. policy가 갑자기 무너지는 step을 옛 prediction이 brake.

---

## 5. cfg.py 추가

```python
@dataclass
class ActionChunkingCfg:
    """ACT-style action chunking for BC. Disabled by default."""
    enabled: bool = False
    chunk_size: int = 12             # k step
    inference_alpha: float = 0.1     # temporal ensemble decay rate
    use_chunked_actor: bool = True   # if False, BC stays 1-step
    loss_type: str = "l1"            # "l1" or "l2"
```

`BCConfig`와 `PPOConfig`에 `action_chunking: ActionChunkingCfg` 필드 추가 (또는 `train_cfg`에 글로벌).

---

## 6. 코드 후크

### BCTrainer 변경

```python
# bc.py
if self.bc_cfg.action_chunking.enabled:
    # Need (obs, action) pairs with k-step lookahead
    # Modify DemoBuffer to yield (obs[t], action[t:t+k]) tuples
    for obs_batch, action_chunk_batch in chunked_dataloader:
        pred_chunks = self.net.actor_chunks(obs_batch)
        loss = F.l1_loss(pred_chunks, action_chunk_batch)
        ...
else:
    # 현재 코드 그대로
```

### DemoBuffer chunk-yield

```python
def chunked_minibatches(self, k, batch_size):
    """Yield (obs[t], action[t:t+k]) for t in valid range."""
    valid_starts = []  # t such that t..t+k-1 all in same episode
    for ep_start, ep_end in self._episode_boundaries():
        for t in range(ep_start, ep_end - k + 1):
            valid_starts.append(t)
    # shuffle + minibatch...
```

### Inference (eval_policy.py / PPO inference)

```python
chunk_buf = ChunkBuffer(cfg.chunk_size)
for t in range(...):
    if cfg.use_chunked_actor:
        chunk = net.actor_chunks(obs[None]).squeeze(0).cpu().numpy()
        chunk_buf.push(t, chunk)
        action = chunk_buf.get_action(t, alpha=cfg.inference_alpha)
    else:
        action = net.actor_mean(obs[None]).squeeze(0).cpu().numpy()
```

---

## 7. PPO transfer

`actor_mean`이 chunk의 첫 step만 리턴해서 **PPO는 1-step 학습 그대로**. ACT BC로 얻은 weight는 backbone + actor_head 둘 다 사용. PPO 학습 중 chunk_size 변경 X — chunk_head는 BC 단계에서만 fully trained 후 PPO에서는 effectively 첫 슬라이스만 사용.

**한계**: PPO가 chunk를 완전히 활용하지 못함. 대안 — PPO도 chunked rollout 학습 가능하지만 코드 변경 큼 (rollout buffer가 chunk 단위로). 우선은 BC 강화 목적으로만 도입.

---

## 8. 도입 결정 트리

```
A 옵션 결과
  ├─ ≥ 50% AND lift/release 회귀 없음 → A 성공, ACT 불필요
  └─ < 50% OR 영상에서 covariate shift 패턴 (BC가 시간에 따라 점진 무너짐)
       ↓
     ACT 도입 검토
       ├─ codex 리뷰
       ├─ cfg.action_chunking.enabled = True
       ├─ BC retrain (3k demo + chunk_size=12)
       ├─ Eval BC + temporal ensembling
       │   ├─ 50%+ → PPO retrain (1-step compat 모드)
       │   └─ < 50% → DAgger로 표류 지점 보강
       └─ rollback: ACT BC가 1-step BC보다 못하면 1-step 복귀
```

---

## 9. 검증 / Rollback

- BC eval 100ep success rate ≥ 1-step BC baseline (19%) 이상
- per-condition 모두 baseline 이상 (특히 lift_history 36%, release_retreat 21%)
- temporal ensembling on/off 비교 (α=0이 1-step와 동일해야)
- 영상에서 hover / staleness 행동 없음

---

## 10. 미해결 / 후속 결정

- **chunk_size**: 12 vs 16 vs 8. horizon 160의 5-10% 권장. 데이터 양 영향
- **temporal ensembling α**: 0.1 (decay 10 step). 너무 작으면 stale plan, 너무 크면 effectively 1-step
- **Encoder 아키텍처**: MLP vs Transformer encoder. 단순 MLP가 빠르지만 obs 시계열 활용 못 함. Phase 1은 single-step obs이라 MLP로 충분
- **PPO와 ACT chunk 호환**: 현재 설계는 PPO가 첫 step만 활용. PPO를 chunk-aware로 만드는 건 별도 결정 (코드 변경 大)
