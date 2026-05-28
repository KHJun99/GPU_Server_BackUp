# v1 12-session lessons (참고만)

## v19 baseline
Oracle 44%, BC 40%, PPO 43% @ z=0.05
saddle: Mode A 31% (success_z mismatch) + Mode B 49% (env ceiling)

## 확정 fact
- success_z: training/eval 동일하게 강제
- PincOpen mimic: USD config `convert_mimic_joints_to_normal_joints: true`
- gripper close (sim): left_proximal = -0.6
- gripper (실물): gripper.pos 0=close, 100=open
- action scale: 1.5 (v7+)
- Session 17 gripperless ablation → 그리퍼 원인 아님 확정
- v1 install: pip install -e . --no-deps (의존성 우회)

## 회피
- entropy 0.01 + action squashing (v7 lift 0.14 실패)
- env vision화 (5~7일, 시간 부족)

---

## v2 5/12 새벽 saddle 진단 (완료)

### 시도한 fix 8개 + 결과
1. URDF collision 5개 link 추가 → ✅ reconvert 성공, state machine 통과 빨라짐
2. Oracle v1 lessons 5개 적용 (base frame, jacobian, ee_cap, close freeze, grip sign) → ✅
3. max_ee_step 0.02 → 0.50 (사실상 uncapped) → ✅ 진척
4. descend_dz 0.005 → 0.13 (gripper_link finger 9cm offset 보정) → ✅
5. finger_x_offset -0.04 (finger 가 wrist 보다 앞이라서) → ✅
6. close_command_expr right_proximal 부호 flip (mimic multiplier -1) → ✅
7. cube size 3cm → 5cm + close 강화 ±0.8 → ❌ cube 안 잡힘
8. IK pose 모드 (orientation 강제) → ❌ 5-DOF 한계로 APPROACH stuck
9. left_proximal only (mimic 에 맡김) + 단계적 close → ❌ cube 안 잡힘

### v19 saddle 진짜 원인 (확정)
1. **PincOpen mimic 부분 작동** — `convert_mimic_joints_to_normal_joints=True` 적용해도 mimic 일부 살아있음. 좌우 비대칭 close (L=-0.77, R=+0.58).
2. **5-DOF arm orientation 한계** — 6-DOF orientation IK 풀 수 없음. position-only IK = finger 자세 자유 = cube 잡기 정렬 어려움.
3. **Finger gap mismatch** — full close 시점 finger gap ≈ 4.8cm. cube 3~5cm 와 mismatch.

### 진짜 fix 후보 (모두 비현실적)
- URDF mimic 제거 + closed kinematic chain 별도 구현 (큰 작업, 메커니즘 깨질 위험)
- IK 후처리로 wrist_flex/wrist_roll 강제 (1~2시간 작업, 5/12 낮 메인 트랙 침해)
- Gripper geometry 자체 재설계 (CAD 작업, 일정 불가)

### 결론
RL/Oracle 트랙 sim 검증은 saddle 그대로 인정. 시연 plan B (γ): v1 demo 재활용 + BC 학습만 v2 환경. 메인 시연은 IL 양팔 (다른 팀원).

---

## v2 5/12 morning — saddle 돌파 ✅

### 결정적 fix 4개 조합
1. **NullspaceIK** — 5-DOF arm 의 redundancy 활용
   - Primary task: position 도달 (3-DOF)
   - Secondary task (nullspace): wrist 자세 default 유지 (2-DOF)
   - 결과: finger 거꾸로 향함 문제 해결
2. **PD stiffness 강화**
   - gripper: 60 → 500 (8x stiffness)
   - finger_distal: 5 → 50 (10x)
   - close target -0.8 까지 진짜 도달
3. **URDF mimic 완전 제거**
   - mimic 3개 (left_distal, right_proximal, right_distal) 모두 제거
   - 4 finger 다 명시적 close target
   - 결과: 좌우 대칭 close, cube 안 미끄러짐
4. **finger_x_offset +0.006** — geometry 보정

### 결과
- **success rate 0% → 12.5%** (1/8 env, cube 11cm lift)
- 추가 3 env 도 lift 진행 중 (7~10cm)
- v1 12세션 saddle 풀림

### 미해결 (improvement 여지)
- 일부 env 가 cube 못 잡음 (cube random spawn 위치 reach 한계)
- close timing / lift trajectory 최적화 가능

---

## v2 5/12 morning — saddle 돌파 ✅

### 결정적 fix 4개 조합
1. **NullspaceIK** — 5-DOF arm 의 redundancy 활용
   - Primary task: position 도달 (3-DOF)
   - Secondary task (nullspace): wrist 자세 default 유지 (2-DOF)
   - 결과: finger 거꾸로 향함 문제 해결
2. **PD stiffness 강화**
   - gripper: 60 → 500 (8x stiffness)
   - finger_distal: 5 → 50 (10x)
   - close target -0.8 까지 진짜 도달
3. **URDF mimic 완전 제거**
   - mimic 3개 (left_distal, right_proximal, right_distal) 모두 제거
   - 4 finger 다 명시적 close target
   - 결과: 좌우 대칭 close, cube 안 미끄러짐
4. **finger_x_offset +0.006** — geometry 보정

### 결과
- **success rate 0% → 12.5%** (1/8 env, cube 11cm lift)
- 추가 3 env 도 lift 진행 중 (7~10cm)
- v1 12세션 saddle 풀림

### 미해결 (improvement 여지)
- 일부 env 가 cube 못 잡음 (cube random spawn 위치 reach 한계)
- close timing / lift trajectory 최적화 가능

---

## v2 5/12 BC 학습 성공 ✅

### Pipeline 완성
1. `scripts/collect_demos.py` — oracle 로 success episode 수집
2. `scripts/train_bc.py` — BCActor 학습 (v1 그대로 copy)
3. `scripts/eval_bc.py` — sim 에서 BC 평가

### Demos
- 30 success episodes 수집 (4분 elapsed)
- obs_dim=32, act_dim=6, ep_len=300
- z_max range: 0.075~0.154

### BC 학습
- 30 epoch, MSE loss
- train: 0.083 → 0.001 (100x 감소)
- val: 0.028 → 0.0007 (overfitting 없음)
- BCActor 32→256→128→64→6 ELU

### BC eval (sim)
- **57.3% success rate** (3 trials × 32 envs = 96 episodes)
- Oracle 31% → BC 57% (variance reduction + pattern learning)
- z_max mean 0.077, max 0.116

### v1 saddle 진짜 풀림 — 진짜 진척
- 0% → 20% (baseline)
- → 31% (Oracle)
- → 57% (BC)
- → ? (PPO warmstart)

### 주요 디버깅 lesson
- `time_out_term` 함수가 `time_out=True` 플래그만 두고 항상 False return 하면 episode 안 끝남
- IsaacLab v2.3.0 에서는 함수가 진짜 `env.episode_length_buf >= max_episode_length` 체크해야 함
- 디버깅에 ~30분 소요 (보이지 않는 hang 처럼 보였음)

---

## v2 BC 다양한 실험 결과 (5/12 오후)

### 최종 비교
| Setup | Demos | Success |
|---|---|---|
| Oracle (sim) | - | 31% |
| **30 ep BC (single)** | 30 | **51-55%** ← BEST |
| 500 ep BC (single) | 517 | 2.2% (mode collapse) |
| 500 ep MoG (K=4, argmax) | 517 | 30.6% |
| 500 ep MoG (K=4, sample) | 517 | 26.9% |
| 30 ep MoG | 30 | 0.6% (underfit) |
| 100 ep BC narrow spawn | 106 | 20.6% |

### Lesson — 많은 데이터가 항상 좋은 것 아님
- 30 ep 가 좁은 spawn 영역만 cover → BC 가 그 mode 안정 학습 → high success in covered area
- 500 ep 는 다양한 spawn 까지 cover → BC single-mean 이 다 평균 → mode collapse → fail
- MoG (K=4) 가 500 ep 의 mode collapse 일부 회복 (2% → 30%)
- 진짜 fix 는 dataset 다양성 vs BC 표현력 균형

### Lesson — PPO warmstart with BC 효과 미미
- 500 iter / 1500 iter 시도 둘 다 BC 정책 파괴 (0~3%)
- reward shaping 함정: lift reward 가 z 비례라 cube 잡고 hold 만 해도 reward 받음
- success_threshold (0.10) 와 success_z (0.07) 불일치도 문제
- PPO 가 lift reward 의 local optimum 에 빠짐

### spawn range 좁힘 실험 (실패)
- pose_range x ±2cm → ±1.5cm, y ±3cm → ±2cm
- 의도: 30 ep BC 가 잘 작동하는 영역으로 데이터 집중
- 결과: Oracle success rate 31% → 18%, BC eval 20.6% (BC 더 약해짐)
- 좁힌 영역이 Oracle 의 강한 영역과 다름
- 좁힘이 만능 아님

### 최종 결정
- **30 ep BC 51% 가 진짜 BEST**
- 추가 시도 모두 그보다 못함
- 메인 트랙 (실물 deploy) 우선

---

## v2 BC 강화 시도 (5/12 오후 ~3시간) — 모두 실패

### 시도 정리
| 실험 | 결과 | 메모 |
|---|---|---|
| **30ep BC obs32 single** | **51-58%** | **BEST, 진짜 ceiling** |
| 500ep BC single | 2.2% | mode collapse |
| 500ep MoG K=4 (argmax) | 30.6% | 부분 회복 |
| 500ep MoG K=4 (sample) | 26.9% | |
| 30ep MoG | 0.6% | underfit |
| 100ep BC narrow spawn | 20.6% | spawn 좁힘 역효과 |
| PPO 50/500/1500 iter | 0-3% | BC 파괴 |
| 200ep BC obs39 (state aug) | 1.2% | obs scale mismatch |
| 200ep BC obs39 + normalize | 21.2% | normalize 부족 |
| 100ep BC top50 filter | 0-6% | over-filtered |
| BC ensemble 3 seeds | 0-6% | seeds 비슷 |

### 핵심 lesson — Counter-intuitive
1. **데이터 많을수록 안 좋음** — 30ep BC 가 다양한 z_max episode 학습 → 다양한 grip 패턴
   500ep 는 oracle 의 모든 mode 학습 → BC single mean 이 다 평균 = mode collapse
   (이건 ML 의 진짜 surprise. 데이터 다양성 vs 모델 capacity 불일치)

2. **State augmentation 도 normalize 없으면 역효과**
   ee_pos (std 0.006) vs joint_vel (std 1.28) = scale 200배 차이
   normalize 없이 추가하면 BC 가 큰 std features 만 의존

3. **PPO warmstart 함정 (reward shaping)**
   lift reward (z 비례) + success threshold mismatch (0.10 vs 0.07)
   PPO 가 cube 잡고 hold 만 학습 → local optimum, success bonus 못 받음
   noise 0.1 + lr 1e-5 보수적이어도 50 iter 만에 BC 파괴

4. **30ep BC 가 좁은 spawn 영역에서만 강함**
   y in [0.0, 0.02): 75% / y in [0.02, 0.04): 14%
   = spawn-dependent variance 가 BC 의 본질적 한계

### 발표 메시지
- v1 12세션 saddle (0%) 풀고 v2 에서 BC 51% 달성
- 그 위 시도들 (PPO warmstart, MoG, state aug, ensemble, filter) 모두 ceiling 못 넘김
- BC + MSE + single Gaussian + small data 가 진짜 sweet spot 인 학습 결과
- multimodal action distribution 표현이 진짜 다음 단계 (Diffusion / CQL)


---

## 5/12 BC spawn dependency 측정

확장 spawn range (x 18~32cm, y -5~+5cm) 으로 30ep BC eval (640 episode):

### X 거리별
| x (cm) | n | success |
|---|---|---|
| 18-20 | 103 | 0% |
| 20-22 | 91 | 14% |
| **22-24** | **81** | **43%** ← peak |
| 24-26 | 87 | 34% |
| 26-28 | 96 | 17% |
| 28-30 | 100 | 1% |
| 30-32 | 82 | 0% |

### Y 좌우별
| y (cm) | n | success |
|---|---|---|
| -5 ~ -3 | 134 | 2% |
| -3 ~ -1 | 137 | 17% |
| **-1 ~ +1** | **132** | **41%** ← peak |
| +1 ~ +3 | 126 | 11% |
| +3 ~ +5 | 111 | 1% |

### Sweet spot 진짜 좌표 (sim base frame)
- x = 22~26cm (peak 22~24)
- y = -1 ~ +1cm

### Lesson
- BC 가 진짜 spawn-dependent
- 학습 데이터 영역 (x 22~26, y -3~+3) 안에서만 작동
- 그 밖 = 0% (학습 안 됨 + reach 한계)
- 시연 시 cube 위치 결정적


---

## 5/12 Residual RL 시도 — 망함 (final)

### Setup
- BC actor frozen (51%) + Residual MLP (학습)
- action = (BC(obs) + residual(obs) * scale).clamp(-1, 1)
- Critic learned from scratch

### Results
| scale | iter 0 | iter 499 |
|---|---|---|
| 0.1 | 0% | 0% |
| 0.05 | 1% | 2% |
| 0.0 (verify) | 53% | - |

### Root cause
- scale=0 = BC와 정확히 동일 (53% 회복)
- scale=0.05 (mean abs delta 0.013) 부터 BC 망침
- BC 정책 fragile, action 미세 변경에 진짜 약함
- chaotic dynamics + cube grip의 sensitive contact

### Lesson
- BC 51% = 진짜 ceiling
- 표준 RL fine-tune (PPO warmstart, Residual RL) 모두 실패
- 진짜 필요한 방법: SAC + replay (off-policy), offline RL (CQL/IQL), KL constraint PPO


---

## 5/12 밤 — Pure PPO from scratch 시작

### Setup (reward shaping 재설계)
- lift: cube_z → clamp(z - 0.04) (hold-only trap 제거)
- success: weight 50 → 200 (huge bonus)
- drop: NEW -2.0 penalty
- action_rate: -0.01 → -0.05

### Training cfg
- num_envs: 1024
- max_iter: 10000
- init_noise_std: 1.0 (large exploration)
- learning_rate: 1e-3
- entropy_coef: 0.01
- 총 timestep: ~245M
- 예상 시간: ~4시간
- GPU: CUDA_VISIBLE_DEVICES=1 (L40S)

### Log: /tmp/scratch_train.log
### 시작 시간: 5/12 19:31
### 예상 끝: 5/12 ~23:30
### 결과 확인: 5/13 아침


---

## 5/12~13 Pure PPO from scratch — 99% SUCCESS! 🎉

### Setup (reward shaping 진짜 fix)
- lift: cube_z → clamp(z - 0.04) (hold-only trap 제거)
- success: weight 50 → 200 (big bonus)
- drop: NEW -2.0 penalty
- action_rate: -0.01 → -0.05
- init_noise_std: 0.1 → 1.0
- num_envs: 64 → 1024

### Training
- 5000 iter × 1024 env, 491M timesteps
- lr 1e-3, entropy 0.01
- 시간: 2시간 55분

### Eval results (3 trials × 32 env = 96 episodes)
| ckpt | success | z_max mean |
|---|---|---|
| model_500 | 93.8% | 0.501 |
| model_1000 | 95.8% | 0.525 |
| **model_2000** | **99.0%** | **0.543** ← BEST |
| model_3000 | 89.6% | 0.498 |
| model_4000 | 96.9% | 0.534 |
| model_4999 | 96.9% | 0.519 |

### Comparison
- Oracle: 31%
- BC 30ep: 51%
- PPO warmstart: 0~3%
- Residual RL: 0~3%
- **Pure PPO + reward redesign: 99%** ← winner

### Lesson
1. Reward shaping이 진짜 RL의 art — lift=cube_z 같은 dense reward는 hold-only local optimum 형성
2. Threshold (z>0.04 시만 lift) + 큰 success bonus가 진짜 답
3. Init noise std 1.0 + num_envs 1024 = 진짜 큰 exploration
4. BC warmstart는 fragile policy 만들고 RL이 그것 destroy
5. Pure PPO from scratch + 좋은 reward = 진짜 winner

