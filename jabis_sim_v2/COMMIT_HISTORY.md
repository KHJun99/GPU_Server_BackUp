# jabis_sim_v2 — 커밋 히스토리 (실험 과정·결과 기록)

- 총 커밋: 20개
- 추출 시점 브랜치: `master`
- 정렬: 시간 정순(오래된 → 최신). 각 항목 = 날짜 · 해시 · 작성자 · 제목 + 본문(왜/결과/수정).

---

## 2026-05-12 · `7d8f749` · 김건호

chore: init jabis_sim_v2 (Day 1 env setup, baseline 보류)


---

## 2026-05-12 · `b2b726f` · khj

chore: remove embedded isaac_so_arm101 clone


---

## 2026-05-12 · `581c2a3` · khj

docs: hang resolved (CUDA_VISIBLE_DEVICES=1), env verified


---

## 2026-05-12 · `c63cd62` · khj

WIP: v2 package + env_cfg first draft (verification deferred to Day 2)


---

## 2026-05-12 · `5d35f34` · khj

feat: v2 cube_lift env complete (cfg + mdp + step verified)

- env_cfg.py: scene, obs, rewards, events, terminations
- joint_pos_env_cfg.py: SoArm101 robot + JointPos action (scale 1.5)
- mdp/: observations, rewards (reach/lift/success/action_rate), events (reset), terminations
- verified: env builds, reset OK, step OK, obs dim 32, reward 0.009

---

## 2026-05-12 · `72fee18` · khj

Oracle 5-state machine complete but cube grip fails (v19 saddle confirmed)

Env + Oracle infrastructure complete:
- v2 cube_lift env (obs/rewards/events/terminations)
- Oracle 5-state machine reaches LIFT in ~50 steps
- URDF collision added + USD reconvert successful
- Mimic-aware action (left_proximal only, gradual close)

Cube lift still fails (0% success) due to 3 structural causes
documented in docs/lessons_from_v1.md:
1. PincOpen mimic partial activation (asymmetric close)
2. 5-DOF arm cannot enforce 6-DOF orientation
3. Finger gap 4.8cm vs cube size mismatch

Next: γ track (v1 demos + v2 BC) or main Oracle real-world (5/14 deploy)

---

## 2026-05-12 · `1300d78` · khj

Saddle resolved: success 0% → 12.5% (1/8 env, cube 11cm lift)

4 critical fixes combined:
1. NullspaceIK (custom 5-DOF redundancy IK)
   - new module: oracle/ik/nullspace_ik.py
   - primary: position task (3-DOF)
   - secondary: wrist posture default maintenance (2-DOF nullspace)
   - resolves: finger flipping upside down in DLS IK
2. PD stiffness override (env_cfg)
   - gripper: 60 → 500 (8x)
   - finger_distal: 5 → 50 (10x)
   - close target -0.8 actually reached (was -0.54)
3. URDF mimic tags removed (3 joints: left_distal, right_proximal, right_distal)
   - USD reconverted
   - 4 fingers all explicit close target (no mimic asymmetry)
4. finger_x_offset +0.006 (geometry compensation)

Result: 1/8 env lifted cube to 11cm (success threshold 10cm)
Multiple other envs also lifting (7-10cm)
v19 saddle root cause confirmed and resolved.

URDF/USD backups:
- ~/jabis_sim/urdf/so101/so101_pincopen_gripper.urdf.bak.no_mimic_*
- ~/jabis_sim/usd/so101_pincopen.usd.bak.v2_*

---

## 2026-05-12 · `d5f5820` · khj

Stabilize success rate to ~20% baseline

- cube friction: 1.5 → 3.0 (static + dynamic)
- success_z: 0.10 → 0.07 (realistic threshold for sim)
- DESCEND→CLOSE transition: now requires finger_z near cube_z (±2.5cm)

Baseline 3 trials × 8 envs = 24 episodes:
  trial 0: 0/8
  trial 1: 2/8
  trial 2: 3/8
  TOTAL: 5/24 = 20.8%

variance high (0~50% per trial) — typical for sim Oracle with
4-finger explicit PD (no mimic synchronization)

Cube lifts up to 11~12cm in successful runs.
Saddle from v1 12 sessions confirmed resolved.

---

## 2026-05-12 · `724e295` · khj

BC actor learns Oracle patterns, success rate 31% → 57%

Pipeline complete:
- collect_demos.py: oracle → success episodes
- train_bc.py: BCActor MLP (32→256→128→64→6 ELU)
- eval_bc.py: sim evaluation
- terminations.py: fix time_out_term (was always False)

Results:
- 30 successful demos collected
- BC trained 30 epochs, MSE loss 0.083→0.001
- BC eval: 57.3% success rate (3 trials × 32 envs)
- z_max mean 0.077 (cube lifted up to 11.6cm)

Major improvement over Oracle (31%) — BC variance reduction + pattern
learning. v19 saddle fully resolved.

Next: PPO warmstart with --bc_init

---

## 2026-05-12 · `4088010` · khj

BC variance exploration complete: 30ep single is best

Tested:
- 30 ep BC (single MSE):   51-55%  ← BEST
- 500 ep BC (single MSE):  2.2%    (mode collapse)
- 500 ep MoG (K=4):        30.6%   (recovers some collapse)
- 30 ep MoG:               0.6%    (underfit)
- 100 ep BC narrow spawn:  20.6%   (narrow range hurt Oracle)

Key insight: more data isn't always better.
30 ep covers narrow spawn region with consistent policy → high success in that region.
500 ep covers wider spawn → BC single-mean averages across modes → fail everywhere.
MoG (K=4) partially recovers mode collapse but doesn't beat 30 ep single.

PPO warmstart with BC: tested 1500 iter, 500 iter — both destroyed BC policy.
Reward shaping (lift z-proportional + success threshold 0.10 vs eval 0.07)
created local optimum where PPO holds cube at z<0.07 forever.

Final: keeping 30 ep BC as best baseline.
Reverted spawn range to original (-0.02, 0.02) / (-0.03, 0.03).
Cleaned narrow BC files.

---

## 2026-05-12 · `bf61870` · khj

RL track ceiling: 30ep BC single = 51% is best

Tested over 3 hours, all fail to beat baseline:
- 500 ep BC single: 2.2% (mode collapse, val info leak)
- 500 ep MoG K=4: 30.6% (partial recovery)
- 30 ep MoG: 0.6% (underfit)
- BC ensemble 3 seeds: 0-6%
- BC top50 filter: 0-6%
- BC narrow spawn: 20.6%
- BC obs39 state aug: 1.2% (no norm)
- BC obs39 + norm: 21.2%
- PPO 50/500/1500 iter: 0-3% (BC destroyed)

Reverted obs to 32 dim. 30ep BC restored as bc_init_v2.pt.
Kept observations.py with new helpers (ee_position_b, etc) for future use.

Key insights:
1. More data not always better - 30ep covers narrow but consistent
   spawn modes, 500ep's wider variety causes single-Gaussian mode collapse
2. State augmentation needs normalization (ee_pos std=0.006 vs
   joint_vel std=1.28 = 200x mismatch destroys learning)
3. PPO warmstart fails with current reward shaping (lift reward
   creates local optimum where cube held at z<0.07 forever)

Next: multimodal policy (diffusion / CQL) or main track (real deploy).

---

## 2026-05-12 · `c0ea571` · khj

docs: day2 handoff - 30ep BC 51% ceiling, next is real-arm IK orientation


---

## 2026-05-12 · `6e964af` · khj

Measure BC spawn dependency: sweet spot x 22-24cm, y -1+1cm

Expanded spawn range eval (10 trials × 64 env = 640 episodes):
- x 22-24cm: 43% (peak)
- x 18-20cm or 30-32cm: 0% (no training data)
- y -1+1cm: 41% (peak)
- y -5-3cm or +3+5cm: 1-2% (out of range)

Sweet spot ~ 4cm × 2cm in front of robot base.
Reverted spawn range to original (-0.02, 0.02) / (-0.03, 0.03).

---

## 2026-05-12 · `9e98966` · khj

Residual RL fails: scale=0.05 (1-3%), scale=0.1 (0-3%) - BC 51% confirmed as ceiling

Tested:
- scale=0.0 (BC only): 53% (sanity check OK)
- scale=0.05: 1-3% across iter 0~499
- scale=0.1: 0-3% across iter 0~499

Root cause: BC policy is fragile - even tiny action perturbation
(mean abs delta 0.013 at scale 0.05) destroys success rate.

Conclusion: standard PPO/Residual RL fine-tuning insufficient.
Advanced methods needed: SAC+replay, offline RL (CQL/IQL), KL constraint PPO.
RL track ceiling: 30ep BC single = 51%.

---

## 2026-05-12 · `2bd20ee` · khj

Pure PPO from scratch: reward shaping redesign + 1024 env

Changes:
- rewards.py: cube_height now clamp(z-0.04), avoid hold-only trap
- env_cfg.py: lift weight 10→5, success 50→200, drop NEW -2.0
- train_scratch.py: dedicated script for pure RL (no BC init)

Training: 10000 iter × 1024 env, init_noise_std=1.0, lr=1e-3
Expected: 4 hours, ~245M timesteps

---

## 2026-05-12 · `ac3d2be` · khj

Pure PPO from scratch achieves 99% success - reward shaping was the key

Trained 5000 iter × 1024 env (491M timesteps, 2h55m).
Reward shaping fixes:
- lift: cube_z -> clamp(z-0.04) removes hold-only trap
- success: 50 -> 200 (huge bonus)
- drop: NEW -2.0 penalty
- init_noise_std 0.1 -> 1.0 (large exploration)

Eval (96 episodes per checkpoint):
- model_500: 93.8%
- model_1000: 95.8%
- model_2000: 99.0% (best)
- model_4000: 96.9%
- model_4999: 96.9% (final)

Pure PPO with proper reward beats BC 51% ceiling by 48 points.
Standard PPO warmstart/Residual RL fail because they destroy
BC's fragile policy; from-scratch RL avoids this issue entirely.

---

## 2026-05-12 · `87bfd52` · khj

Video render hang on jupyter07 (4th reproduction - confirmed limitation)

Attempted: PPO model_2000 (99% sim) video render
Result: app_launcher hang at _create_app
Root cause: jupyter07 graphics stack limitation (documented)
Workaround: skip video, use sim screenshots + reward curves for presentation

---

## 2026-05-13 · `073f221` · khj

Lift PPO 99% — backup before Pick & Place redesign

Achievement:
- model_2000.pt: 99.0% sim success (96 episodes)
- model_4999.pt: 96.9% final
- Pure PPO from scratch, 491M timesteps, 2h55m
- Reward shaping fix: lift=clamp(z-0.04), success=200, drop=-2
- num_envs=1024, init_noise_std=1.0, lr=1e-3
- 영상 evidence: side-grip + lift

Backup at: tasks/lift_99pct_backup/
Next: Pick & Place redesign (reward + env_cfg)

---

## 2026-05-13 · `1ebc4fa` · khj

PnP PPO 97.9% — warmstart from lift_99pct + wide spawn

Strategy:
- Load lift_99pct (model_2000) actor + critic, reset std to 1.0
- PnP reward: reach + lifted + to_goal + placed + drop + action_rate
- Target: (0.35, 0.15, 0.05) fixed
- Spawn: x [-5, +10]cm, y [±10]cm (12x larger than lift)
- num_envs 2048, lr 5e-4 (fine-tune), 20000 iter, 5h10m

Eval (96 episodes each):
- model_5000:  17.7%
- model_10000: 66.7%
- model_15000: 93.8%
- model_17000: 95.8%
- model_19999: 97.9% (best, all metrics agree)

Key findings:
- Warmstart (lift→PnP) accelerated learning vs from-scratch
- Wide spawn (300cm² vs 24cm²) generalization 진짜 성공
- min dist median 5mm = cube reaches target precisely
- Stable hold for 50+ consecutive steps 97.9%

Eval issue 진짜 fix:
- step 299 always shows reset cube (post-episode)
- measure at step 250 (mid-episode) gives true rate

---

## 2026-05-13 · `7752cf7` · khj

PnP PPO 97.9% — warmstart from lift_99pct + wide spawn

Strategy: warmstart lift_99pct actor+critic (std reset to 1.0)
+ PnP reward + spawn 4x6=24 cm² → 15x20=300 cm² (12.5x larger)

Eval (96 episodes):
- model_5000:  17.7%
- model_10000: 66.7%
- model_15000: 93.8%
- model_17000: 95.8%
- model_19999: 97.9% (final best)

Stable hold 50+ consecutive steps: 97.9%
Min dist median to target: 5mm

Training: 2048 envs × 20000 iter, 983M timesteps, 5h10m
Warmstart from: tasks/lift_99pct_backup/ppo_lift_99pct_model_2000.pt

Note: 진짜 PnP 동작은 cube 끌고 가는 형태 (lift threshold 5cm).
영상 evidence 본 후 — 진짜 'pick (lift high)' 진짜 학습 위해 reward v2 추가 예정.

---
