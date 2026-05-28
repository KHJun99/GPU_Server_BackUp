# SACfD Design (Card 1) — 2026-05-19

목표: 어제 PPO + BC ACT 실패(#19, #21) 이후 RL 트랙으로 정공법 복귀. 1-step BC (no chunking) 새로 학습 → **SAC + Demo (SACfD)** fine-tune.

## 1. 선택 근거

| 결정 | 근거 |
|---|---|
| 1-step BC (no chunking) | ACT chunked actor 1-step interface(`chunk[0]`) 단독 성공률 5%. RL fine-tune의 시작점이 너무 낮음. MLP 1-step actor는 deploy 시점에도 ChunkBuffer 없이 동작. |
| **SAC** (vs PPO) | (1) Off-policy로 demo buffer 재사용 가능. (2) Auto-entropy tuning이 BC 시작점 변동에 강함. (3) Sparse env에서 sample efficiency가 on-policy PPO보다 압도적. (4) 어제 PPO actor freeze + critic warm-up 후에도 sparse signal 부재로 실패한 것에 대한 직접 대응. |
| **+ Demo buffer (fD)** | Sparse env에서 SAC 단독은 cold start 어려움. Demo replay로 (a) Q-net 초기 anchoring, (b) BC auxiliary loss로 actor manifold 보존. |
| **차별화 가치** | 팀원 (LeRobot ACT, image+state, abs joint, 분해/조립 task) 대비 **off-policy RL + sparse reward + state-based** — paradigm·obs·task 모두 명확히 분리. |

## 2. Architecture

### 2.1 Network (`khj_rl/training/sac_network.py`, 신규)

```
GaussianActor(obs_dim=31, action_dim=6, hidden=256, n_layers=2, tanh_squash=True):
    trunk: Linear(31, 256) → ReLU → Linear(256, 256) → ReLU
    mean_head: Linear(256, 6)
    logstd_head: Linear(256, 6)  # state-conditional (PPO와 다름)
    forward(obs) → mean, logstd (clip [-20, 2])
    sample(obs) → action(tanh squashed), log_prob (with tanh jacobian correction)
    mean_action(obs) → tanh(mean) (deterministic eval)

TwinQ(obs_dim=31, action_dim=6, hidden=256):
    Q1: Linear(31+6, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1)
    Q2: 동일 구조 (independent init)
    forward(obs, action) → (Q1, Q2)
```

**왜 새 네트워크**: 기존 `ActorCritic`은 (a) state-independent logstd, (b) no tanh squash, (c) shared trunk가 아닌 분리. SAC는 state-conditional std + tanh squash + Q-net이 필요. 기존 `ActorCritic`은 PPO 용도로 보존.

### 2.2 BC compatibility (`khj_rl/training/bc.py` 수정)

기존 BCTrainer는 `ActorCritic` / `ChunkedActorCritic` 사용. SACfD를 위한 BC는 **새 `GaussianActor` 백본**과 호환되어야 함.

**옵션 A (선택)**: BCTrainer에 `network_type` 인자 추가. `"actor_critic"` (기존), `"chunked"` (기존 ACT), `"gaussian_sac"` (신규). `gaussian_sac` 모드에서:
- Actor: `GaussianActor`
- Critic: 학습 안 함 (SAC가 처음부터 학습)
- Loss: **NLL** (BC가 stochastic policy를 학습하기 위함). `-mean(log_prob(action_pre_squash))`. 또는 단순화하면 MSE(mean, action) — 시작값으론 MSE 권장 (구현 단순, 결과 동일에 가까움).

**옵션 B (rejected)**: BCTrainer를 두 개로 분리. 보일러플레이트 늘어남.

### 2.3 SAC trainer (`khj_rl/training/sac.py`, 신규)

```python
@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    # SAC hyperparams (standard CleanRL/Stable-Baselines3 defaults)
    gamma: float = 0.99
    tau: float = 0.005                  # soft target update rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = None        # default = -action_dim = -6
    init_alpha: float = 0.2
    # Replay
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000        # online steps before first update
    update_every: int = 1               # gradient steps per env step (UTD=1)
    # Demo
    demo_batch_ratio: float = 0.25      # 0.25 → 64/256 minibatch from demo
    bc_loss_weight: float = 1.0         # auxiliary BC loss on actor
    bc_loss_q_filter: bool = True       # use Q-filter (DDPGfD/SACfD style)
    bc_anneal_steps: int = 100_000      # linearly decay bc_loss_weight → 0
    # Resources
    device: str = "cuda:0"
```

**학습 루프**:
1. `learning_starts` 까지 random action 또는 BC actor inference로 env step + buffer fill
2. 매 env step 후:
   - online buffer에 transition push
   - `update_every` gradient step:
     - online buffer에서 `(1 - demo_batch_ratio) * batch_size` 샘플
     - demo buffer에서 `demo_batch_ratio * batch_size` 샘플
     - merged batch로:
       - Critic update (TD error on Q1, Q2)
       - Actor update (Q + α·entropy + **bc_loss_weight × BC_loss on demo subset**)
       - α update (auto entropy tuning)
       - Target Q soft update
3. 매 N step마다 (a) eval rollout (deterministic mean_action), (b) video_sampler 호출

### 2.4 BC auxiliary loss (Q-filter)

```python
# demo subset 안에서만 적용
with torch.no_grad():
    q_actor = Q1(obs, actor.mean_action(obs))  # actor 현재 출력
    q_demo = Q1(obs, demo_action)              # demo의 oracle action
    mask = (q_demo > q_actor).float()           # demo가 더 좋을 때만 따라가기
bc_loss = (mask * (actor.mean_action(obs) - demo_action).pow(2).sum(dim=-1)).mean()
total_actor_loss = sac_actor_loss + bc_loss_weight(step) * bc_loss
```

**Q-filter 의미**: BC loss를 무조건 따르면 demo가 suboptimal한 곳에서 SAC가 학습한 더 좋은 행동을 망친다. Q-filter는 "demo가 현재 actor보다 나을 때만" BC pull을 적용 → BC를 floor로만 사용. (Rajeswaran et al. 2018, DAPG; Vecerik et al. 2017, DDPGfD).

## 3. Demo buffer 통합

기존 `DemoBuffer` (NPZ 로더)는 (obs, action) 페어만 있음. SAC가 필요한 건 (obs, action, reward, next_obs, done). 두 옵션:

**옵션 A (선택)**: `DemoReplay` 신규 class. NPZ 파일을 재로딩할 때 (obs[t], action[t], reward[t], obs[t+1], done[t]) 튜플 재구성. NPZ에 `obs`, `actions`, `rewards`는 이미 있음. `next_obs[t] = obs[t+1]` (마지막 step은 `done=True`로 처리). RAM 한 번에 다 올림 (~404k step × 31 float = 50MB → 무시 가능).

**옵션 B (rejected)**: Demo에서 reward 무시하고 BC loss만 사용. → Q-net이 demo로 부트스트랩 못 함. 정공법 SACfD가 아님.

```python
class DemoReplay:
    def __init__(self, demo_dir, success_only=True):
        # 기존 DemoBuffer.from_dir의 spans 활용해 next_obs 재구성
        ...
    def sample(self, batch_size, rng):
        return obs, action, reward, next_obs, done  # all np.float32
```

## 4. Phase 순서

### Phase A: 1-step BC retrain (1-2h)
1. `train_bc.py` 호출 시 `--no-chunking` flag → `ActionChunkingCfg(enabled=False)`
2. 새 `--network-type gaussian_sac` flag → `GaussianActor` 백본 (또는 우선 기존 `ActorCritic`로 호환성 유지하고 SAC 전환 시 weight 매핑)
3. **단순화 선택**: Phase A는 **기존 ActorCritic으로 1-step BC 학습**. SAC trainer가 시작 시 `ActorCritic.actor` weight을 `GaussianActor.trunk + mean_head`로 surgical transfer. logstd_head는 random init.
4. Stage 0+1 demo 전부 사용 (964+1913+359+700+500 = 4436 ep). epoch 50, batch 256, lr 3e-4. ~10분 학습.
5. **게이트**: eval_policy.py로 stage 0 100ep deterministic eval. 시작점 ≥ **30%**. 미달이면 BC architecture 재검토.

### Phase B: SACfD smoke (30분)
- `train_sac.py --bc-ckpt ... --total-steps 50_000 --learning-starts 5_000`
- 모니터: ep_return, Q1/Q2 mean, actor_loss, bc_loss, alpha, demo/online ratio
- **게이트**: 50k step (~25 iter, 50k/2k) 안에 ep_return > 0 한 번 이상. 또는 Q-value가 reward scale에 맞게 증가 (sparse 1+5 = 6점 만점이라 Q≈6 부근 수렴 흐름).

### Phase C: SACfD full (4-6h)
- 300k-500k step. video_sampler 매 5k step (반드시 — reward hacking 모니터).
- 영상 검증: 짓누르기/굴리기/hover/trivial release **0건** 필수.
- 최종 eval 100ep → ≥ 70%면 stage 1 advance.

## 5. 가드 (CLAUDE.md guardrails 준수)

| 가드 | 적용 |
|---|---|
| Video auto-sampling | `--video-every-steps 5000` 디폴트 ON. PPO와 동일 sampler 인터페이스 재사용. |
| Obs alignment | `assert_obs_alignment` + `assert_obs_distribution` startup에서 |
| Single source of truth | obs/action normalize는 BC frozen mean/std. SAC가 running update X. |
| Success 5조건 logging | env가 이미 episode 종료 시점에 info dict로 5조건 노출 — train_sac.py에서 그대로 로그 |
| GPU isolation | `CUDA_VISIBLE_DEVICES=0` 명시 (팀원 lerobot이 GPU 1) |

## 6. 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `src/khj_rl/training/sac_network.py` | 신규: `GaussianActor`, `TwinQ` |
| `src/khj_rl/training/sac.py` | 신규: `SACConfig`, `SACTrainer` |
| `src/khj_rl/training/demo_buffer.py` | 추가: `DemoReplay` (next_obs/reward/done 재구성) |
| `src/khj_rl/training/bc.py` | 변경: `BCConfig.network_type` 또는 기존 ActorCritic 1-step 그대로 사용 (Phase A에서 결정) |
| `src/khj_rl/training/__init__.py` | export 추가 |
| `scripts/train_sac.py` | 신규: BC ckpt 로드, SACTrainer 부트, video sampler 옵션 |
| `scripts/train_bc.py` | 변경: `--no-chunking` 옵션 (or 디폴트가 chunking=False 되도록) |
| `scripts/chain_sacfd.sh` | 신규: BC retrain → smoke → full pipeline. killer marker 포함. |

## 7. Risk + Rollback

| Risk | Mitigation |
|---|---|
| BC retrain 1-step 성공률이 ACT actor_mean보다 더 낮음 | Demo가 충분 (4436 ep). 시작점 게이트 30%는 ACT chunk[0]의 5%보다 6배. 미달 시 demo 양 늘리거나 actor capacity 키움. |
| SAC가 demo manifold 못 따라감 (entropy 폭주) | Auto-entropy target_entropy = -6 (액션 차원), init_alpha=0.2. BC loss로 actor를 anchor. |
| Q-net이 sparse reward에서 학습 안 됨 | Demo buffer에 success episode reward(+1)가 다수 → Q1/Q2 첫 update에서 signal 보장. |
| Reward hacking 재발 | Video sampler 5k step마다 + 영상 visual 검증 매 50k step. (어제 D 옵션 짓누르기 attractor 학습) |
| Close-hang | killer script with marker grep + pkill (운영 패턴 #17, #22) |

**Rollback**: SACfD pilot이 50k step 안에 ep_return > 0 못 보면 → 카드 2 (HER + SAC) 전환. SAC 인프라는 그대로 재사용, demo buffer 끄고 HER buffer 켬.

## 8. Codex 검토 (2026-05-19) — PROCEED + 4개 조정

판정: **PROCEED**. Architecture / Q-filter / demo buffer 재구성은 표준 SACfD 설계와 일치. 다음 조정 반영:

### 8.1 logstd 초기화 (Surgical transfer 시)
- BC 단계에서 `logstd_head`는 **frozen**, **고정값 -1.0** (std ≈ 0.37). CLAUDE.md Phase 1 결정 7의 `actor_logstd=-1.0`과 일관.
- SAC 첫 iter에서 unfreeze. Oracle demo는 결정론적 motion planning이라 분산 매우 낮음 → BC가 logstd 학습하면 SAC 진입 직후 entropy 폭발 위험.

### 8.2 BC loss type
- **MSE(mean, action)** 채택. NLL은 logstd 학습 강제 → 8.1과 충돌.

### 8.3 demo_batch_ratio 단계별 전환
- 0~50k step: **0.5** (cold start, online buffer 희박)
- 50k step 이후: **0.25** (DDPGfD 표준)
- 선형 전환 OR step 기반 hard switch (구현 단순성으로 hard switch 선택).

### 8.4 BC weight anneal 연장
- **bc_anneal_steps = 200_000** (기존 100k → 200k). sparse env + curriculum 전환 시점 보호.
- 또는 milestone-based: success rate ≥ 70% 달성 후 50k step에 걸쳐 decay (구현 복잡 → Phase B에서 step-based 우선).

### 8.5 Q-filter warm-up
- `learning_starts(5000 step)` 전까지 BC loss는 **unconditional** (Q-filter 우회). 이유: 무작위 Q-net 초기 단계에 Q-filter는 무의미한 mask 생성.
- 5000 step 이후 Q-filter `q_demo > q_actor (margin=0)` 활성화.

### 8.6 Surgical transfer 안전장치
- BC 단계에서 action target에 atanh 적용 **금지**. SAC tanh squash 경로에서 mean이 [-1,1] 안에 있을 때 BC weight 영향 최소화.
- SAC 첫 eval에서 `mean_action(obs)` 분포 히스토그램 → 포화 진단.

### 8.7 NPZ rewards fallback (확인 완료)
- NPZ에 `rewards` 필드 **존재 확인** (shape=(T,) float32). 재구성 불필요.
- `terminated`는 scalar(episode-level) → `done[t] = (t == T-1)` 로 재구성.
- DemoReplay 구현 시 위 schema 그대로 활용.
