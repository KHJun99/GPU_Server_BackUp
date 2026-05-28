# 세션 10 결과 요약 보고서 — Camera Fix + Entropy Ablation

**일자**: 2026-05-09
**브랜치**: `khj-rl-track`
**Commit**: `af7dc78`
**대상**: 양팔 시연 RL 트랙 (한 팔 담당)

---

## TL;DR

- **카메라 view 2개** (top + diag) 정리 완료, 영상 발표 자료 18 mp4 확보
- **Entropy ablation 결과**: v0(0.02) → v1(0.005) → v2(0.001+lr 3e-4)
  - std 발산 회피 성공 (54× → 0.15, **366× 작아짐**)
  - reward shape 정상화 (0.93 → 4.54, **4.9× 개선**)
  - **그러나 success rate 13% plateau (BC 14% 동등)** — entropy 문제 아님 확정
- **시연 strategy**: Demo path 권고 (BC 14% + Oracle 4.C 영상 주력, PPO ablation 자료)

---

## 1. 배경

세션 9 PPO warmstart 실패:
- entropy_coef=0.02 → action_std 1.01 → 54.92 (54× **발산**)
- iter 999 success **10%** (BC 14%보다 낮음)
- entropy bonus(32.5) 가 reward(1.24) 무력화

세션 10 가설:
- entropy 보수화 (0.005 → 0.001) → BC saddle 탈출
- 학습 안정화 후 success ≥ 25% 도달 가능

---

## 2. 카메라 정리

### 문제
- 세션 8/9 의 카메라 1개 (`tiled_camera`, ROS convention rot=(0,0,1,0))
- 결과: 옆 누운 시점 — 발표용 부적합

### 해결
- **카메라 2개 등록**: `top_camera` + `diag_camera`
- **OpenGL convention** 채택 (forward=-Z, identity rot 으로 자연 top-down)
- diag quaternion **numpy R→quat 직접 계산** (look_at (0.2, 0, 0.05))
  - pos=(0.7, 0.5, 0.5), 결과 quat=(0.3354, 0.1841, 0.4446, 0.8099)
- render_policy.py multi-view 지원 (`--views top,diag`)

### 결과
- 18 mp4 (3 정책 × 2 view × 3 seed)
- archive_session_9_view/: 구 view 12 mp4 보존

---

## 3. Entropy Ablation 결과

### 학습 cfg

| Variant | entropy_coef | learning_rate | 의도 |
|---|---|---|---|
| v0 (세션 9) | 0.02 | 1e-4 | (실패) std 발산 |
| **v1** | 0.005 | 1e-4 | 4× 보수 |
| **v2** | **0.001** | **3e-4** | 20× 보수 + active update |

### Iter 별 평가 (100 ep, action_repeat 2)

| iter | v0 succ | v0 std | v1 succ | v1 std | v2 succ | v2 std |
|---|---|---|---|---|---|---|
| 0 | 14.0 | 1.01 | 14.0 | 1.00 | 14.0 | 1.00 |
| 100 | 15.0 | 7.85 | 13.0 | 1.22 | 13.0 | 0.69 |
| 200 | 13.0 | 21.80 | 14.0 | 1.61 | 13.0 | 0.50 |
| 500 | 13.0 | 32.72 | 13.0 | 2.37 | 13.0 | 0.21 |
| **999** | **10.0** | 54.92 | **13.0** | 2.57 | **13.0** | **0.15** |

(plot: `tasks/ablation_session10_curves.png`)

### 학습 reward decomposition

| metric | v0 | v2 |
|---|---|---|
| Mean reward | 0.93 | **4.54** (4.9×) |
| Mean action_std | 54.92 | **0.15** (366× ↓) |
| Mean entropy_loss | 32.5 | **-9.4** (음수) |
| reaching_object | 0.70 | 0.70 |
| **lifting_object** | **0.14** | **0.16** |
| action_rate penalty | -0.15 | -0.07 |
| joint_vel penalty | -0.54 | -0.39 |

---

## 4. 핵심 결론

### v2 의 의미
- **entropy 문제 정확히 회피**: std 0.15 안정, reward shape 정상화, MSE vs BC 3.2 (v0 의 0.007%)
- **그러나 success rate 13% plateau** (BC 14% 동등)

### 진단
**BC saddle exit 가 entropy 문제 아님 확정.** 진짜 원인 후보:
- **Reward misalignment** (codex caveat): reaching (0.70) + joint_vel penalty (-0.39) 가 lifting_object (0.16) 보다 dominant. PPO 가 reward 잘 받지만 success 와 alignment X
- **Demo coverage**: 402 demos 가 cube init 의 특정 영역만 cover, BC 가 그 영역에 saddle 수렴

---

## 5. 시연 Strategy 권고 — Demo Path

| 지표 | Oracle | BC | PPO v2 best |
|---|---|---|---|
| success rate | 13% (4 seed mean) | 14% | 13% |
| z_max max | 0.189 | 0.194 | 0.189 |
| 시연 영상 자료 | ✓ (4.C threshold) | ✓ (lift 0.04~0.19m) | ✓ (ablation) |

**권고 메시지** (Codex 검토 정직 구조):
- "Oracle 로 task 가능성 입증 (cube lift 정상)"
- "BC 14% 로 재현 가능한 부분 성공 (sim 학습 자료)"
- "PPO 는 reward misalignment ablation"

양팔 시연에서 한 팔 RL 트랙 자료 충분.

---

## 6. Codex 마무리 Caveats (다음 세션 우선순위)

1. **Reward misalignment 의심** ⭐: reward 4.9× 늘었는데 success 무변화 → reaching/joint_vel 이 lifting 보다 dominant. lifting_object weight ↑ (0.16→0.5), joint_vel penalty ↓ (-0.39→-0.1) 권고.

2. **Demos stratify 구현**: 기존 `demos_session8_total400.pt` meta 에 cube init position 없음. **재수집하며 cube_pos 추가 기록** 이 가장 실용적 (z_max 근사는 약함).

3. **Fixed_std 후순위**: v2 의 std 0.15 가 사실상 already fixed-like. fixed_std 더 작게 시도해도 noise 문제 아닌 reward/coverage 문제 dominant.

4. **시연 path 정직**: "Oracle 가능 / BC 부분 성공 / PPO ablation" 구조 합리적.

5. **다음 세션 우선순위**:
   - (a) **demos stratify** (cube init 재수집) — coverage 실패 vs reward 실패 분리
   - (b) **reward shape 검토** (lifting weight ↑, joint_vel penalty ↓)
   - (c) demos 추가 수집 (cube init 다양화)
   - (d) ~~fixed_std~~ (후순위)

---

## 7. 산출물

### 코드 (commit `af7dc78`)
- `src/.../tasks/lift/joint_pos_env_cfg.py`: SoArm101LiftCubeEnvCfg_VIDEO 카메라 2개 추가
- `src/.../tasks/lift/agents/rsl_rl_ppo_cfg.py`: entropy 0.001, lr 3e-4
- `tasks/render_policy.py`: multi-view 지원
- `tasks/plot_ablation_session10.py`: ablation 곡선 generator

### 백업 (cp .pre)
- `tasks/backups_session10/{joint_pos_env_cfg, rsl_rl_ppo_cfg, render_policy}.session10_pre`

### 데이터
- `tasks/eval_session10_v{1,2}_iter*.log` (10 ckpt eval logs)
- `tasks/ppo_train_session10_v{1,2}.log` (학습 로그)
- `tasks/ablation_session10_curves.png` (시각화)

### 영상 (`~/jabis_sim/day5/videos/`)
- session7/: oracle_4c_v1_{top,diag}_seed{0,1,2}.mp4 (6)
- session8/: bc_v1_{top,diag}_seed{0,1,2}.mp4 (6)
- session10/: ppo_v2_iter999_{top,diag}_seed{0,1,2}.mp4 (6)
- archive_session_9_view/: 구 view 12 mp4 보존

### Cleanup Whitelist 준수 ✓
USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged. 수정은 cfg 3 파일만 (사용자 승인 + cp 백업).

---

## 8. 다음 세션 11 (entropy ablation 한계 → reward/coverage 진단)

**우선 작업**:
1. Demos stratify — cube init 재수집 (cube_pos 메타 추가 wrapper)
2. Reward shape 검토 — lifting weight ↑, joint_vel penalty ↓
3. Demos 추가 수집 (cube init 다양화)
4. (후순위) fixed_std

**시연 path** (사용자 결정):
- Demo path 권고: BC 14% + Oracle 4.C 영상 주력, PPO ablation 자료

**RL 트랙 종료 여부**: 사용자 결정. 세션 11 의 stratify 결과 후 평가 권고.
