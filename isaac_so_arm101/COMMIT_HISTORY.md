# isaac_so_arm101 — 내 작업 커밋 히스토리 (실험 과정·결과 기록)

- 이 repo는 MuammerBay/isaac_so_arm101 의 fork. 아래는 **내(김건호) 커밋만** 추출.
- 내 커밋: 17개 / 전체 208개
- 추출 브랜치: `khj-rl-track`, 시간 정순.

---

## 2026-05-09 · `15f3bb3` · 김건호

khj rl track: session 1-6 cfg + diagnostics


---

## 2026-05-09 · `4463190` · 김건호

session 7 start: pre-fix snapshot (cfg + diagnostics)


---

## 2026-05-09 · `92b8c69` · 김건호

session 7: 102 demos collected, decimation 1 fix


---

## 2026-05-09 · `4f5a3c8` · 김건호

session 8: 400 demos + BC 50Hz + first videos

- Camera variant SoArm101LiftCubeEnvCfg_VIDEO + Isaac-SO-ARM101-Lift-Cube-Video-v0
  (top-down TiledCamera 320x240, num_envs=1)
- Demo expansion 102+300=402 (merge_demos.py with metadata consistency check)
- 50Hz downsample dataset (downsample_demos_50hz.py, stride=2) for HW match
- BC train 30 epochs: train 0.00489->0.00112, val 0.00137->0.00107
- BC standalone eval 14% (oracle baseline 13% mean of 4 seeds, codex predicted)
- render_policy.py oracle/bc unified wrapper, fixed seeds 0/1/2
- 3 seed robustness pass: 7/14/16% (default 15%) -> mean 13% std 4pp

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-09 · `ae3435f` · 김건호

session 9: PPO warmstart fail (BC saddle, entropy explode)

- train.py --bc_init added (rsl-rl 2.3.x: runner.alg.policy.actor)
- PPO entropy_coef 0.01 -> 0.02 (codex saddle-escape)
- Camera view pos z 0.6 -> 1.2, FOV 24->18 (wider work area)
- render_policy.py extended with --policy ppo --ppo_ckpt
- 1000 iter PPO (15min, BC init verified at iter 0 = 14%)
- iter 999: 10% (regress), action_std 1->55 (54x explode)
- gates ALL FAIL: iter200 13% / iter500 13% / iter999 10%
- 12 new mp4 videos (session7/8/9) with new wider view
- old narrow-view archived to videos/archive_old_view/

Next session: entropy 0.005 or noise_std fix, critic burn-in.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-09 · `af7dc78` · 김건호

session 10: camera fix + entropy ablation v1/v2 (BC saddle confirmed)

- camera: top (OpenGL identity) + diag (numpy quat for look_at), 2 views
- render_policy.py --views, sensor_name='<view>_camera' multi-view dispatch
- v1 entropy 0.02->0.005: std 54x->2.57 (no explode), success 13% (plateau)
- v2 entropy 0.001+lr 3e-4: std 0.15 (366x smaller), reward 0.93->4.54, success 13% (still plateau)
- ablation curves png: success/std/MSE-vs-BC across v0/v1/v2
- 18 mp4 videos (3 policies x 2 views x 3 seeds), session-9 view archived
- BC saddle exit NOT entropy issue, codex caveat 4 (stratify) priority confirmed

Next: demos stratify + try 3 (fixed_std).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-10 · `61701f2` · 김건호

session 11: reward shape ablation (saddle exit ≠ reward)

- v1 (lift 3x, joint_vel 0.3x): success 13%, reward decomp lifting 0.47 (3x verify)
- v2 (reach 0.5x, lift 5x): success 14% (BC exact), reward decomp lifting 0.82 (lifting/reaching 800x ratio)
- Reward shape changed faithfully but success plateau confirms misalignment hypothesis FAIL
- 3 hypotheses (entropy / reward / fixed_std equivalent) all unable to escape BC saddle
- Remaining: codex caveat 4 (demos coverage)
- 6 mp4 videos session11/ (PPO v2 ablation footage)
- session11_report.md (Path C Demo path recommendation)

Next: demos stratify + cube_pos meta re-collection.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-10 · `a98d137` · 김건호

session 12: demos coverage analysis -> RL track close (Path C)

- demos_session8_total400.pt cube init: x sigma 6.1cm range [0.10, 0.30], y sigma 13.4cm range [-0.20, +0.20]
- Plan assumption (narrow ±2cm distribution) INVALID -> demos coverage already broad
- codex caveat 4 (demos coverage) hypothesis FAIL
- 5-session ablation stack exhausted: mimic / actuator / decimation (TRUE FIX 0->15%) / entropy / reward / coverage
- 15% ceiling root cause unresolved; further RL attempts not justified
- User decision: Path C (RL track close, sim2real transition)
- Demo path: Oracle 13% + BC 14% + PPO ablation videos sufficient

Next: sim2real session (BC + Oracle hw deployment, action_repeat 2 validation).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-10 · `1751149` · 김건호

session 13: policy behavior diagnosis - saddle root cause identified

- ee_to_cube_min < 5cm: Oracle 86% / BC 2% / PPO 1% (decisive signal)
- BC actor learns trajectory 14cm away from cube, never grasps closely
- PPO closes 100% but ee 14cm from cube, random catch (cube_vel 0.34 push)
- 14% success across all = lucky init when cube spawned near ee
- success/fail cube_init distribution identical -> hyp 4 (narrow region) FAIL
- NEW hypothesis 5: BC trajectory drift (val loss 0.001 but action drift accumulates)
- 12 mp4 videos session13/ (BC + Oracle behavior comparison)
- session13_report.md (8 sections) with fix candidates

Fix candidates: reward distance std 0.05->0.02, DAgger, demos perturbation.

User decision: (a) fix 3 ablation 1 session, (b) keep Path C, (c) DAgger 1-2 sessions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

---

## 2026-05-10 · `eab82da` · 김건호

session 14: reward distance std fix (transient improvement, close collapse)

Fix 1 ablation: reaching_object std=0.05 → 0.02 (1줄 cfg, 다른 weight 동결).
세션 13 진단 (BC ee 14cm drift, ee<5cm 도달율 BC 2% / PPO 1%) 의 첫 fix.

핵심 결과:
- iter 200 sweet spot: ee<5cm 도달율 31% (BC/PPO 대비 15× 개선)
- iter 400 재 spike: 26%
- iter 500+ collapse: ee<5cm 2-3%, close 시도 0~2%
- success rate 14% (s13 동일) — 도달 spike 시점도 close timing 무너짐

핵심 게이트 ≥50% 미달 → fix 1 단독 부족 확정. 그러나 reward 방향 옳음 입증.

새 가설: (6) sharper reward → exploration 망가짐, (7) lifting reward 가
close 직접 보상 안 함 (가장 promising), (8) entropy 로 close 행동 잊음.

산출물:
- tasks/diagnose_session14_iter{0,100,150,200,250,300,400,500,999}.csv
- tasks/analyze_session14.py
- tasks/reward_landscape.png
- tasks/ppo_session14_iter{200,999}.pt
- 12 mp4 ~/jabis_sim/day5/videos/session14/iter{200,999}/
- tasks/session14_report.md (10 sections)
- tasks/lessons.md §14 + tasks/todo.md §14

User decision: (A) Fix 7 close reward + std 유지 1세션 / (B) DAgger 1-2세션
/ (C) Path B + iter 200 ckpt 시연 / (D) Path C 확정 sim2real.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `2e29649` · 김건호

session 15: close reward + early stop (path C confirmed, all gates fail)

Fix 7 ablation: grasp_object RewTerm 추가 (gripper_closure_near_object,
std=0.03, weight=2.0) + max_iterations 200 (collapse 방지).
세션 14 의 close timing 무너짐 가설 검증.

핵심 결과:
- s15 iter100 sweet spot: 도달율 21%, success 15%
- s14 iter200 31% 도달 → s15 21% 감소 (close reward 가 reaching 방해)
- close% 활성 (99-100) 그러나 timing 잘못 (close_dist 18cm)

핵심 게이트 (3개 모두 미달):
- success ≥ 18%: 15% ❌
- ee<5cm 도달 ≥ 40%: 21% ❌
- close_dist 1~5cm: 18cm ❌

Path C 확정 — RL 트랙 종료, sim2real 진입.

세션 13/14/15 검증:
- saddle 진짜 원인 식별 ✓ (도달 못 함)
- reward shaping 단독 부분 효과 (transient max 31%)
- DAgger 또는 demo 수정 같은 큰 fix 필요

새 가설: (9) reaching std=0.02 자체 너무 sharp, (10) BC saddle 이 PPO update
로 deepen, (11) lifting weight 75 가 catch 보상에 압도적.

산출물:
- tasks/diagnose_session15_iter{0,50,100,150,199}.csv
- tasks/ppo_session15_iter{100,199}.pt
- 12 mp4 ~/jabis_sim/day5/videos/session15/iter{100,199}/
- tasks/session15_report.md (12 sections)
- tasks/lessons.md §15 + tasks/todo.md §15

User decision: (A) BC + Oracle sim2real (권장) / (B) + s14 iter200 PPO 보조
/ (C) RL 추가 ablation (lifting weight).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `d2eaff9` · 김건호

session 16: pathon-ai baseline eval attempt (failed: env cfg mismatch)

PathOn AI checkpoint (model_1950, 2100) 평가 시도. 동일 task name 였으나
환경 cfg fundamentally different.

Dim mismatch 발견:
- obs: PathOn 28 vs 우리 36 (last_action 8 dim + 2 dim 추가)
- action: PathOn 6 vs 우리 8 (gripper control 부재)
- → PathOn 환경은 6-DOF arm only (gripper 없음)

평가 진행 불가 — obs hack 만으론 action dim mismatch 해결 못 함.
"high success rate" 주장은 우리와 다른 task 환경에서 측정.

의미 있는 발견:
- task name 동일성이 환경 동일성 보장 안 함
- 우리의 14% saddle 이 gripper binary control 학습 어려움일 가능성
- 외부 baseline 비교 실패 → Path C 결정 번복 근거 없음

산출물:
- tasks/diagnose_pathon.py (diagnose_policy 기반 + obs strip)
- tasks/pathon_ai_comparison.md (9 sections)
- tasks/lessons.md §16 + tasks/todo.md §16
- (PathOn ckpt 자체는 commit 제외 - tasks/pathon_ai_baseline/ untracked)

세션 13~16 종합 → RL 트랙 종료, sim2real 진입 (Path C 확정).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `c469980` · 김건호

session 17: gripperless ablation (hypothesis FAIL — env limit confirmed)

Gripper-less ablation (open=close=항상 closed, grasp_object reward 제거).
2000 iter from-scratch (BC warmstart 없음, PathOn-AI 방식).

가설 PASS/FAIL:
- gripperless final success 14% (BC saddle 동일)
- 도달율 max 5% (s14 iter200 31% 보다 낮음)
- → 가설 FAIL: gripper 무관, 환경/task 자체 한계 확정

세션 13~17 4 ablation 종합:
- 13: saddle 진짜 원인 (도달 못 함) 식별
- 14: reward fix transient 31% (collapse)
- 15: close reward 효과 미흡
- 16: PathOn 비교 불가 (cfg 다름)
- 17: gripper 무관 확정

→ saddle = cube random init 우연 catch (어떤 정책이든 14%, Oracle 도 15%)
→ Path C 확정 강한 보강 (BC + Oracle sim2real)

cfg 변경 (additive):
- joint_pos_env_cfg.py: SoArm101LiftCubeGripperlessEnvCfg + _PLAY + _VIDEO
- __init__.py: 3 new gym.register
- lift_env_cfg.py 보존

산출물:
- tasks/diagnose_session17_iter{0,200,500,1000,1500,1999}.csv
- tasks/ppo_session17_iter1999_gripperless.pt
- 6 mp4 ~/jabis_sim/day5/videos/session17/iter1999/
- 7 mp4 + 3 plot ~/jabis_sim/day5/videos/presentation/
- tasks/ablation_evolution.png (4-panel session 13~17)
- tasks/session17_report.md (11 sections)
- tasks/lessons.md §17 + tasks/todo.md §17

다음 세션 (5/12): Path C sim2real 진입.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `9648d4e` · 김건호

session 18: task spec ablation (saddle root cause = success_z threshold + Oracle lift plateau)

5세션 ablation에서 누락된 Oracle baseline 14% 자체 검증:
- A1 (episode 5s→10s): 15% (변화 없음, 시간 무관)
- A2 (cube_y ±20→±10cm, A1 stack): 2% (역효과, success가 cube_y abs >10cm에 집중)
- A3 (success_z 0.10→0.05, env baseline): 44% (+29%p, 단일 변수 최대)

Bimodal fail mode 식별 (codex 검증):
- mid-lift 26/85 (31%): Oracle z 5–10cm 도달, 임계 미달 → success_z로 회복 가능
- no-lift 49/85 (58%): grip 후 cube z<2.5cm, 메커니즘 미식별

env cfg는 baseline 원복 (5s / y±20cm) — A1/A2 효과 없거나 역효과, A3는 측정 인자만.

다음 세션 권고: Oracle --lift_dz 0.15 검증 (현실적 ~41% ceiling), no-lift 49 별도 진단.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `80fca74` · 김건호

session 19: success_z=0.05 demos 재수집 + BC/PPO 재학습 (Path A 후보, honest caveat)

env cfg 변경 X (옵션 B 채택, CLI arg --success_z 0.05만):
- Demos 재수집: 219개 (target 200, oracle rate 34.2%)
- BC v19: loss 수렴, bc_actor_session19_z5cm.pt
- PPO v19: 1500 iter BC warmstart, session19_v1_z5cm

평가 결과 (success_z=0.05):
- BC v19: 40% (BC v8 14% 대비 게이트 통과)
- PPO v19 iter300 best: 43%
- PPO v19 iter1499 final: 40%

Honest caveat: z=0.10 기준 모든 정책 11~15% (historical saddle 동일).
실제 정책 capability 개선 거의 없음 — 향상의 원천은 threshold 완화. PPO learning gain 사실상 0
(BC saddle 그대로, no-lift fail ~83% 유지).

Path 결정: 사용자 결정 사항 (Path A z=0.05 채택 / Path C z=0.10 유지).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `17b25d8` · 김건호

session 20: Mode B (no-lift 48%) Oracle param ablation FAIL — env-level fix required

Mode B 정체:
- Oracle 47/48이 CLOSE 도달, ee_at_close 4.91cm (gripper가 cube에서 5cm 떨어진 상태로 닫힘)
- final_state LIFT 47/48 — CLOSE 거쳐 LIFT 진입했으나 cube 안 들림
- → grasp 실패 (lift 실패 아님)

3가지 가설 ablation (CLI arg, 파일 수정 X):
- H1 lift_dz=0.15: 베이스라인과 완전 동일 (44%/15%, Mode B 48, ee_at_close 4.91)
- H2 friction: cube friction 이미 1.5 적용 중, skip
- H3 reach=0.025/desc=0.015: ee_at_close 4.91→3.56cm, but Mode B 48 동일
- H4 reach=0.015/desc=0.008/close=16: ee_at_close 2.76cm (cube에 거의 닿음), but Mode B 49

→ 모든 Oracle close-timing/reach 변경에서 Mode B 48±1. ee_at_close가 cube에 닿는 거리까지
와도 grasp 회복 안 됨. Mode B는 Oracle 파라미터 수준 변경으로 fix 불가.

추정 원인: gripper jaw geometry vs cube physical contact mismatch (USD 수준 한계).
직접 검증 안 함 (USD 수정 범위 외).

Step 5 (BC/PPO 재학습) skip — fix 없으므로 학습 가치 없음.

12세션 ablation (s9~s20) 종합:
- Mode A 31% mid-lift: success_z threshold 미달 → 0.05로 회복 (s19)
- Mode B 48% no-lift: 환경 수준 fix 필요 (이번 시연 범위 외)
- 기타 21%: 정상 작동

시연 path: Path C 확정 (BC v19 + Oracle 백업, 5/12 sim2real).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-10 · `3f626b2` · 김건호

session 21: USD collision ablation FAIL — Mode B 환경 ceiling 정직 확정 (Path C)

Step 0 — hang debug: PID 2250977 (26분+ GPU util 0%) kill. 17:18 hang 원인 식별:
- config.yaml은 UrdfConverter 출력 (입력 아님), sed 편집 무효
- convert_mimic_joints_to_normal_joints default false → s20 working은 true 빌드
- 무지한 재변환 시 mimic 보존 → gripper 부정합 → hang
→ 신규 wrapper script tasks/usd_convert_session21.py 작성 (UrdfConverterCfg 직접 노출)

3 옵션 시도 (모두 baseline 원복):
- Opt 3 collision_from_visuals=true (10ep): 20%/10%, Mode B 7/10 — 악화
- Opt 2 convex_decomposition (100ep): 44%/14%, Mode B 48/100 — null effect
- Opt 1 URDF 명시적 collision (100ep): 24%/8%, Mode B 71/100 — 큰 악화 (p<0.01)

Counter-intuitive 발견:
- Opt 1: 100/100 episodes에서 first_close_step=-1 (gripper 한 번도 안 닫힘)
- 그럼에도 24개 success (z>=0.05) — passive scoop으로 우연한 lift
- ee_to_cube_min 4.74cm → 7.76cm: visual mesh의 convex hull이 더 큰 footprint
  → gripper jaw가 cube approach 중 phantom contact → DESCEND 단계 차단

URDF 5개 gripper 링크 (mujoco_base, *_proximal, *_distal) collision 누락 발견.
명시적 추가 = regression. baseline의 IsaacLab missing-collision 자동 처리에 의존.

12세션 ablation (s9~s21) — RL/task spec/Oracle params/USD collision axis 검증 종합.
Mode B 환경 ceiling: 검증한 axes 내 fix 없음. 미검증 axes (PhysX solver,
gripper jaw mesh 재설계 등) 다수 존재 — 시연 일정 제약 내 시도 범위 외.

Step 4 (BC/PPO 재학습) skip — fix 없음, 재학습 가치 0.

Path C 확정 (사용자 결정 권고):
- BC v19 (40% @z=0.05) 메인
- PPO v19 iter300 (43% @z=0.05) 보조
- Oracle (44% @z=0.05) 백업
- 5/12 sim2real 시작

산출물: 5 csv + report + lessons §21 + wrapper script. URDF/USD/config 모두 baseline 원복.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---
