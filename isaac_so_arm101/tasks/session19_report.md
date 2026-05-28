# 세션 19 결과 — success_z=0.05 Demos 재수집 + BC/PPO 재학습

**일자**: 2026-05-10
**브랜치**: `khj-rl-track`
**핵심 변경**: success_z 0.10 → 0.05 (env cfg 변경 없음, CLI arg만)

---

## TL;DR

- success_z=0.05 wiring 방식: env cfg 변경 X (option B 선택), `collect_demos.py / diagnose_policy.py --success_z 0.05` 인자만
- demos 재수집 219개 (target 200), Oracle 34.2% (collect_demos params different from diagnose)
- **BC v19 = 40%** (BC v8 14% 대비 +26%p — z=0.05 threshold 기준)
- **PPO v19 best = 43% (iter300)**, plateau (BC 위 학습 gain 거의 없음)
- **honest caveat**: 동일 정책을 z=0.10에서 보면 **11~15%** — historical saddle 동일. 향상은 threshold 완화에서 옴
- 시연 path: success 정의 명확화 필수 — z=0.05 정의 채택 시 Path A, z=0.10 정의 유지 시 Path C

---

## 1. 배경

세션 18 결정:
- Oracle baseline 14% saddle 진짜 원인 = success_z=0.10과 Oracle 실제 lift 도달 (5–10cm) misalignment
- Bimodal fail: Mode A 31% mid-lift (z=5–10cm) + Mode B 58% no-lift (z<2.5cm)
- 옵션 B 권장 (env 변경 없음, CLI arg만)

이번 세션 자율 진행: demos 재수집 → BC → PPO. 멈춤 조건 도달 시 보고.

## 2. Step 1 — env cfg 변경 (skip)

옵션 B 채택. lift_env_cfg.py / joint_pos_env_cfg.py 변경 없음. 기존 cfg(5s episode / y±20cm / minimal_height=0.025) 그대로.

## 3. Step 2 — Oracle 재검증 (skip, 세션 18 A3 재사용)

세션 18 Ablation 3 = `tasks/diagnose_oracle_session18_a3_z5cm.csv` = **44.0%** (Oracle 100ep, success_z=0.05).
≥ 40% 게이트 통과 → 신규 측정 생략, demos 수집 진행.

## 4. Step 3 — Demos 재수집

```
DISPLAY="" CUDA_VISIBLE_DEVICES=1 python collect_demos.py \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --num_envs 64 --target_episodes 200 --max_total_episodes 1000 \
  --success_z 0.05 \
  --output tasks/demos_session19_z5cm.pt
```

결과:
- saved **219 demos** (target 200, slight overshoot)
- finished 640 episodes, success rate **34.22%** (Oracle 44% 대비 낮음)
- 차이 원인: collect_demos.py oracle params 더 tight (reach_dist 0.02 vs diagnose 0.04, descend_z_dist 0.005 vs 0.025)
- elapsed 58s

## 5. Step 4 — BC 학습

```
python train_bc.py --input tasks/demos_session19_z5cm.pt \
  --output tasks/bc_actor_session19_z5cm.pt \
  --epochs 30 --batch_size 256 --lr 1e-3 --device cuda
```

결과:
- 30 epoch 수렴 (loss train=0.00002, val=0.00001)
- BC actor 209 KB 저장

## 6. Step 5 — BC 평가 (100ep, success_z=0.05)

| 지표 | BC v19 | BC v8 (z=0.10 historical) |
|---|---|---|
| success @ z=0.05 | **40.0%** | n/a |
| success @ z=0.10 | **11.0%** (post-hoc) | 14.0% |
| fail z_max mean | 1.71cm | n/a |
| fail z_max max | 4.98cm | n/a |
| fail ee_to_cube_min | 6.26cm | 14cm (s13 BC v8) |
| fail z_max<2.5cm (no-lift) | 51/60 | n/a |
| fail z_max in [2.5, 5)cm | 9/60 | n/a |

→ BC v19는 z=0.05 기준에서 40%. z=0.10 기준에서는 11%. BC v8 14%와의 3%p 차이는 n=100, p≈0.12 binomial SE 3.4%p 안 → **통계적 noise** (의미 있는 회귀 아님).

## 7. Step 6 — PPO 학습 (1500 iter, BC warmstart)

```
python train.py --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --num_envs 4096 --max_iterations 1500 \
  --bc_init tasks/bc_actor_session19_z5cm.pt \
  --experiment_name session19_v1_z5cm --headless
```

결과:
- log: `~/isaac_so_arm101/logs/rsl_rl/lift/2026-05-10_15-44-36/`
- model_0 ~ model_1499 (32 ckpt)
- 학습 시간 ~25분

## 8. Step 7 — PPO 5 checkpoint 평가

| iter | success @ z=0.05 | success @ z=0.10 (post-hoc) | ee_min succ (cm) | fail z_max mean (cm) | no-lift fail/total |
|---|---|---|---|---|---|
| 0 (BC warmstart) | 41% | 13% | 11.61 | 1.72 | 50/59 |
| 300 | **43%** | **15%** | 10.37 | 1.72 | 48/57 |
| 600 | 40% | 11% | 9.46 | 1.86 | 49/60 |
| 1000 | 40% | 12% | 9.98 | 1.77 | 50/60 |
| 1499 | 40% | 12% | 9.93 | 1.77 | 50/60 |

**관측**:
- iter 0 = 41% — BC warmstart 정상 (BC v19 40%와 일치, +1%p stochastic)
- iter 300 peak = 43% — 미미한 PPO gain
- iter 600 ~ 1499 plateau 40% — PPO learning gain 사실상 없음
- no-lift fail (z<2.5cm) 50/60 모든 ckpt 동일 — Mode B (no-lift) 학습 안 됨

→ 세션 9~17 BC saddle 패턴 재현. PPO는 BC를 넘지 못함.

## 9. 비교 표 — Session 19 vs 역사

| 정책 | success @ z=0.05 | success @ z=0.10 | 비고 |
|---|---|---|---|
| Oracle s13 (z=0.10) | n/a | 15% | session 18에서 z=0.05시 44% |
| Oracle s18 A3 | 44% | 15% | threshold 만 변경 |
| BC v8 (s8/s13) | n/a | 14% | BC saddle |
| BC v19 (z=0.05) | 40% | 11% | z=0.05 demos 재학습 |
| PPO best historical | n/a | ~13% | s14 iter200 도달 31% but success 14% |
| PPO v19 iter300 (best) | **43%** | **15%** | session14 v2 cfg + new demos |
| PPO v19 iter1499 (final) | 40% | 12% | plateau |

## 10. 시연 path 결정 분기

| 조건 | success 기준 | path | 다음 세션 |
|---|---|---|---|
| z=0.05 채택 | 40~43% (BC/PPO) | **Path A — PPO 메인** (기준 ≥40%) | sim2real (5/12 월) |
| z=0.10 유지 | 11~15% (모든 정책) | **Path C — BC + Oracle 백업** (기준 <30%) | sim2real |

**현실적 권고 (사용자 결정 사항)**:

- **option 1 (보수)**: Path C 유지. z=0.10은 historical baseline. BC v8 + Oracle 백업으로 5/12 시연. "saddle = task threshold 문제" 메시지 명확히 전달
- **option 2 (개선)**: Path A 채택. success 정의를 z=0.05로 재정의. PPO v19 iter300 메인. 시연 시 "5cm lift = success"로 framing. 실측 향상 (z=0.10 기준)은 미미하지만 task 정의 자체가 재정의된 셈

## 11. Honest Assessment

이번 세션 향상의 본질:
- 실제 정책 capability 개선 = **거의 없음** (z=0.10 기준 11~15%, historical 14% 수준)
- 향상의 원천 = **success threshold 완화** (z=0.10 → z=0.05)
- BC v8 → BC v19 demos 재수집 효과 = 미미. 도달 ee_min은 14cm → 6.3cm로 개선되었으나 success @z=0.10 지표 차이는 통계적으로 무의미 (BC v8 14/100 vs BC v19 11/100, n=100에서 binomial SE ≈ 3.4%p — 한 표준오차 안)
- PPO v19는 BC saddle 그대로 답습. Mode B no-lift fail 비율 (이번 세션 100ep 기준): iter0 50/59, iter300 48/57, iter600 49/60, iter1000 50/60, iter1499 50/60 — 5 ckpt 모두 fail 중 ~83%가 z<2.5cm

→ 세션 18 결론 재확인: **Saddle 두 개 분리 (Mode A 31% threshold 의해 회복 가능, Mode B 58% 별도 진단 필요)**.

## 12. 다음 세션 20 권고

**Path A 채택 시**:
- sim2real 즉시 (5/12)
- BC v19 actor + PPO v19 iter300 ckpt + Oracle 백업 deploy
- success 정의 명확화 (5cm = success)

**Mode B no-lift 진단 (시간 있을 때)**:
- 가설: grip 이후 cube z 안 올라가는 메커니즘 = ?
  - 가설 1: gripper-cube contact force 부족 (slip 측정 못 한 미세 미끄러짐)
  - 가설 2: arm kinematic singularity 근처에서 lift 불가
  - 가설 3: oracle lift_dz=0.10 자체가 낮음 → 0.15로 보강 시 어떻게 변할지 (s18 권고 옵션 A)
- 1세션 ablation: oracle lift_dz=0.15로 demos 재수집 + BC + 평가

## 13. Validation Checklist

- [x] env cfg 변경 결정 (옵션 B 선택, 변경 없음)
- [x] Step 2 Oracle 재검증 (s18 A3 재사용 = 44%)
- [x] Step 3 demos 재수집 (219개)
- [x] Step 4 BC 학습 (loss 수렴)
- [x] Step 5 BC 평가 (40% @ z=0.05)
- [x] Step 6 PPO 1500 iter 학습 완료
- [x] Step 7 PPO 5 checkpoint 평가 (best 43% @ z=0.05)
- [x] Step 8 비교 표 + path 권고
- [x] honest caveat (z=0.10 vs z=0.05 명시)
- [ ] 시연 path 결정 (사용자 결정)

## 14. 산출물

- `tasks/demos_session19_z5cm.pt` (18.5 MB, 219 demos)
- `tasks/bc_actor_session19_z5cm.pt` (209 KB)
- `tasks/diagnose_bc_session19_z5cm.csv`
- `tasks/diagnose_ppo_session19_iter{0,300,600,1000,1499}.csv`
- PPO ckpt: `~/isaac_so_arm101/logs/rsl_rl/lift/2026-05-10_15-44-36/model_*.pt`
