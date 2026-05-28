# KHJ_RL — 커밋 히스토리 (실험 과정·결과 기록)

- 총 커밋: 28개
- 추출 시점 브랜치: `master`
- 정렬: 시간 정순(오래된 → 최신). 각 항목 = 날짜 · 해시 · 작성자 · 제목 + 본문(왜/결과/수정).

---

## 2026-05-15 · `cbc5e98` · 김건호

Initial commit: Phase 0 scaffolding

Extension-point interfaces for object/target/learner/sim2real axes,
with one stub each but no concrete implementation. Designed so Phase 1
can plug in the first env without disturbing the framework.

Guardrails carried over from the jabis_sim_v2 PnP reward-hacking
incident — video auto-sampling, obs alignment check, single source of
truth for env constants — are interface-only here and required active
in Phase 1.

---

## 2026-05-15 · `a7d4d6f` · 김건호

Pin Phase 1 dependencies

Fills in the Phase 1 stack on pyproject.toml: numpy/torch/gymnasium for
the RL core, opencv-python and pyrealsense2 for D456 + wrist perception,
omegaconf/tensorboard/tqdm for config and run-time logging.

Isaac Lab is intentionally left out of pip dependencies — it ships with
NVIDIA Isaac Sim and must be installed separately. CleanRL is also not
listed because Phase 1 writes its single-file PPO directly in this repo.

Dev extras: ruff + pytest.

---

## 2026-05-15 · `b195e40` · 김건호

Restructure deps for Jetson Orin Nano + host split

Phase 1 will train on the x86_64 host (L40S) and deploy policy inference
+ perception to a Jetson Orin Nano (aarch64, JetPack 6). PyPI wheels for
torch and pyrealsense2 don't exist or aren't usable on Jetson, so split
them out:

- Base `dependencies` holds only packages with cross-arch PyPI wheels:
  numpy, gymnasium, opencv-python, omegaconf, tensorboard, tqdm.
- `[host]` extras: torch>=2.3,<2.6 (cu12x range covers PyPI cu124 wheels
  and JetPack 6's NVIDIA torch wheels) and pyrealsense2.
- `[jetson]` extras left empty — torch is installed from NVIDIA's Jetson
  wheel and pyrealsense2 is source-built from librealsense.
- requires-python pinned to >=3.10,<3.11 to match JetPack 6.

CLAUDE.md "개발 환경" section now documents the host vs Jetson install
flows, records the L40S GPU index 1 isolation (CUDA_VISIBLE_DEVICES=1),
and reflects the master branch rename.

---

## 2026-05-15 · `5bc0360` · 김건호

Scaffold Phase 1 cube-lift env

Brings up the Phase 1 surface area for the SO-ARM101 cube-lift task. The
Isaac Lab scene and the actuation path are still stubs, but everything
downstream of them — the env_cfg single source of truth, the obs layout,
the multi-condition success judge, the sim2real noise hook, the obs-
alignment guard, the BC demo writer, and the Trainer surface — is fully
wired up so each piece can be tested before the scene exists.

New modules
- envs/cube_lift/{cfg,env,success,oracle}: CubeLiftEnvCfg dataclass tree
  with default thresholds matching CLAUDE.md, EnvLike-conformant
  CubeLiftEnv, the five-condition success judge, and an MP-oracle stub.
- data/{__init__,demos}: DemoBuffer + EpisodeRecord. Episodes land at
  runs/demos/{stage}/{ep_id}.npz with cfg_hash + seed metadata so BC
  pretrain can refuse demos from a different cfg.
- training/{ppo,bc}: Trainer-conformant stubs for CleanRL-style PPO and
  the BC pretrainer; the optimizer loops come in the next pass.

Changed modules
- sim2real/noise: Phase 1 fields (xy/z split, outlier jumps, visibility
  dropout) and apply_* hooks. apply_obs_object_pos returns a dropout
  flag so the env can fall back to a last-known pose without polluting
  the cache on a cold start.
- eval/obs_alignment: implements the reset-obs vs cfg.obs_keys layout
  check that Phase 0 only stubbed out.
- envs/__init__ and training/__init__: export the new symbols.

CLAUDE.md
- Explicit obs vector (joint_pos, joint_vel, cube_xyz_base,
  target_delta_base, gripper_state, last_action, normalized_t).
- Demo path runs/demos/{stage}/{ep_id}.npz.
- Visibility proxy lives in env.py and is passed into the success
  judge; NoiseModel only touches obs-side cube pose.

---

## 2026-05-15 · `5a2d22f` · 김건호

Absorb external SSAFY DeskMate side-track context into CLAUDE.md

External memo pinned several details that were missing from the Phase 1
decision table and are concrete enough to bake in now:

- Motor spec: Feetech STS3215 x 6 on the arm (1/345 gear ratio); the same
  motor with ID=6 / baud=1M on the gripper.
- PincOpen 4-bar handling in USD: PhysicsMimicJointAPI on the finger
  joints, a single 'gripper' reference in ActuatorCfg (driving both sides
  causes drive conflict), URDF <mimic> tags do not survive USD conversion
  so they must be re-added in Isaac Sim.
- Physics rate: 120 Hz x decimation 6 -> 20 Hz policy.
- Action normalization: 7-D in [-1, 1]; arm uses
  JointPositionActionCfg(use_default_offset=True, scale=0.05); gripper
  uses scale=0.5, offset=0.5 (joint range [0, 1]). Oracle must emit the
  same normalized 7-D so BC and PPO see the same action distribution.
- BC -> PPO transfer checklist (the failure mode the external memo flags
  as most-often-broken): shared ActorCritic, actor_logstd = -1.0, critic
  warm-up 10-20 iter, PPO LR 1e-4 right after BC, obs normalization
  frozen at BC's mean/std, dense weight schedule (Phase 1 stays sparse).
- Domain randomization concrete ranges: gripper PD gain +-20%, finger
  friction 0.5-1.5, object mass/size +-20%, action delay 0-50 ms,
  obs noise (joint pos +-0.5 deg, vel +-10%).
- Isaac Lab API volatility guard: API surface changes fast across
  versions; never write ActuatorCfg / JointPositionActionCfg / RewTerm /
  EventTermCfg signatures from memory.

Held off from baking three items into the decision table because they
conflict with current decisions and warrant explicit confirmation. They
live under a new "추후 결정 후보" section: EE-pose-in-obs (would grow obs
from 27-D to 34-D), dense reward shaping (Phase 1 is sparse-only on
purpose), and ManagerBasedRLEnv module layout (envs/mdp/ split vs the
current envs/cube_lift/ package).

Added external reference links section (Isaac Lab docs, PincOpen repos,
SO-ARM101, CleanRL PPO).

---

## 2026-05-15 · `4278b48` · 김건호

Bake codex-reviewed candidates into Phase 1 decisions

Codex picked one "adopt", two "modified adopt" for the three pending
candidates. Resolving them now so the next push into Isaac Lab scene
work starts from a frozen contract.

Candidate 1 — EE pose in obs: adopt.
  cfg.py adds `ee_pose_base` (xyz(3) + quat[w,x,y,z](4)) between
  joint_vel and cube_xyz_base. Total obs grows 27-D -> 34-D.
  env.py routes both paths through a single _fk_ee_pose(joint_pos)
  helper: the obs path passes the *noisy* joint_pos copy so the obs
  ee_pose only contains signals the real robot can produce, and the
  reward path passes GT joint_pos so the success signal stays clean.
  Codex's Q3 catch — naive GT-leak via _sim_ee_pose_gt() — is fixed by
  this routing.

Candidate 2 — dense reward shaping: modified adopt.
  Phase 1 stays sparse-only (the jabis_sim_v2 rolling-cube reward hack
  is the reason for sparse). A future bounded + annealed auxiliary term
  is allowed only if BC pretrain + curriculum still leave learning
  stalled, and only behind a separate decision + codex review.

Candidate 3 — ManagerBasedRLEnv layout: modified adopt.
  Phase 1 keeps the current envs/cube_lift/ single-package layout.
  Phase 2 migrates incrementally to envs/mdp/ split + ManagerBasedRLEnv
  to avoid scrapping the scaffolding that just landed in 5bc0360.

CLAUDE.md's "추후 결정 후보" section is removed — all three now live
inside the Phase 1 decision table.

Verification: assert_obs_alignment passes, obs_total_size = 34
(6+6+7+3+3+1+7+1), reset-obs ee_pose slot is [0,0,0, 1,0,0,0] (identity
quat), inspect confirms obs path uses noisy joint_pos / reward path
uses GT, ruff all-checks-passed.

---

## 2026-05-15 · `00ca0f6` · 김건호

Land Isaac Lab + Isaac Sim into khj-rl env

The cube_lift env scaffolding can now talk to the simulator: isaaclab
0.47.2 + isaacsim 4.5.0.0 + Omniverse Kit are installed in the same
conda env that already holds the RL code, so we don't need to bounce
between two envs to run a scene.

Notable knobs the install dance taught us, baked into CLAUDE.md so the
next person doesn't relearn them:

- Pin torch to the cu128 index. Isaac Lab 0.47.x metadata forces
  torch>=2.7 and the host driver is CUDA 12.8, so 2.7.0+cu128 is the
  one wheel that satisfies both.
- Bake setuptools<81 + flatdict + toml + wheel + build into the env
  before installing Isaac Lab. pip's default build isolation drags in
  setuptools 81, whose missing pkg_resources kills the flatdict build
  and the isaaclab setup.py import of toml. --no-build-isolation
  routes around that.
- isaaclab_contrib has no pyproject.toml in this tree; skip it.
- Kit boot needs OMNI_KIT_ACCEPT_EULA=YES + PRIVACY_CONSENT=Y to skip
  interactive prompts that hang headless runs.

pyproject.toml's [host] extras now point torch at >=2.7,<2.8 (Isaac
Lab strict pin), separated from the Jetson path (which still uses the
NVIDIA torch wheel from JetPack 6 instead of the host's cu128 wheel).

Verified end-to-end: AppLauncher(headless=True) boots, isaaclab.envs /
isaaclab.assets / isaaclab.managers all import, torch reports CUDA on
L40S. khj_rl's own CubeLiftEnv + obs_alignment guard still pass after
gymnasium got downgraded from 1.3 to 1.2 by Isaac Lab's pin. ruff
all-checks-passed.

---

## 2026-05-15 · `3342d3e` · 김건호

Reuse jabis_sim SO-ARM101+PincOpen USD; cfg → 5-DOF arm

The cube_lift env now points at a real robot articulation. The USD layer
under assets/converted/usd/so101_pincopen.usd is a copy of the SO-ARM101
+ PincOpen assets sitting in /home/j-k14d101/jabis_sim/usd/ — the user
authored that asset in an earlier track, and the jabis_sim_v2 reward-
hacking incident was a reward-design problem, not an asset problem, so
reuse is safe under the "CLAUDE.md 직접 복사 금지" rule (which applies to
RL code, not 3D assets).

What landed:
- Copied the URDF + STL meshes (assets/converted/urdf/) and the layered
  USD (assets/converted/usd/so101_pincopen.usd plus the three
  configuration/{base,physics,sensor}.usd sub-layers) into the repo so
  the env doesn't depend on /home/j-k14d101/jabis_sim staying put.
  .gitignore now excludes assets/raw/ and assets/converted/.
- RobotCfg flipped from a hypothetical 6-DOF arm to SO-ARM101's actual
  5-DOF arm + 1 gripper. joint_names list matches the USD prim paths
  under /so101_pincopen/joints/. obs_total_size dropped 34 → 31 and
  action_size dropped 7 → 6 automatically via the size accessors;
  assert_obs_alignment still passes.
- New khj_rl/envs/cube_lift/articulation.py with soarm101_pincopen_cfg()
  builder returning an isaaclab ArticulationCfg pointing at the local
  USD. isaaclab.sim / isaaclab.assets imports are lazy so the path
  constant is importable without booting Omniverse Kit.

Known gap, not in this commit: the USD has 10 PhysicsRevoluteJoints
(5 arm + gripper main + 4 passive 4-bar) but ZERO PhysicsMimicJointAPI
applications. URDF <mimic> tags don't survive the URDF → USD path. The
4-bar linkage is currently unactuated by the ArticulationCfg's
`gripper` actuator group; making the finger linkage track the main
gripper joint needs the mimic API stamped onto the USD as a follow-up.

Verified: assert_obs_alignment passes, env.reset()/step() returns a
31-vector, ruff is clean, and a headless AppLauncher boot + builder
call produces an ArticulationCfg with the expected joint_pos dict and
actuator groups.

---

## 2026-05-15 · `322d54c` · 김건호

Stamp PhysxMimicJointAPI onto PincOpen 4-bar passive joints

The USD landed in the previous commit had 10 PhysicsRevoluteJoints but
zero PhysxMimicJointAPI applications — URDF <mimic> tags don't survive
the URDF → USD path, so the 4-bar passive joints (left/right proximal
+ distal) were independent. Driving only the `gripper` actuator group
in the ArticulationCfg would have left the finger linkage drifting
free instead of tracking the main joint.

scripts/usd_add_mimic_api.py walks the USD, finds each passive joint
under /so101_pincopen/joints/, reads its physics:axis, and applies
PhysxSchema.PhysxMimicJointAPI:rot{axis} pointing at the gripper main
joint. Gearing is -1.0 for all four (PhysX convention: joint =
-gearing * reference, so -1.0 means same direction). If a particular
linkage moves opposite to the gripper in sim, flip its entry in
MIMIC_DIRECTION rather than hand-editing the USD.

Re-running the check that previously reported `mimic count=0` now
reports 4 PhysxMimicJointAPI:rotZ applications on the four passive
joints, matching the gripper axis. The USD itself is gitignored, so
this commit only carries the script; running it is idempotent and
needs to repeat after any URDF re-conversion.

CLAUDE.md updated to mark the mimic step as done and point at the
script as the source of truth for the wiring.

---

## 2026-05-15 · `900b0a4` · 김건호

Wire CubeLiftEnv to Isaac Lab scene end-to-end

CubeLiftEnv now drives a real Isaac Lab scene from reset() and step()
instead of returning zeros. The scaffolding from 5bc0360 stayed at the
EnvLike surface; this change carries the wiring all the way through
SimulationContext, the InteractiveScene, the articulation, the cube
RigidObject, and the GT buffers feeding success/reward.

New: scene.py builds a single-env InteractiveSceneCfg with a ground
plane, dome light, a kinematic table, the SO-ARM101 + PincOpen
articulation (via articulation.soarm101_pincopen_cfg), and the cube
RigidObject. CuboidCfgs carry explicit CollisionPropertiesCfg so the
table actually collides (Codex 6-B design review caught this).

env.py:
- __init__ boots SimulationContext + InteractiveScene, resolves the EE
  body and the arm/gripper joint indices once via find_bodies /
  find_joints with preserve_order=True.
- reset() writes the home pose into the articulation, samples a cube
  start xy inside curriculum stage 0, calls write_root_pose/velocity,
  then scene.write_data_to_sim() + sim.forward() + scene.update(dt)
  in the order Codex's review required (the previous draft only
  buffered the writes without committing them).
- step() un-normalizes the action ([-1, 1] arm delta + gripper target
  in [0, 1]), clamps to soft joint limits, set_joint_position_target,
  then runs the decimation=6 loop with write_data_to_sim → sim.step
  → scene.update(dt) every physics tick.
- _sim_*_gt accessors now read articulation.data.joint_pos /
  joint_vel / body_pos_w / body_quat_w and cube.data.root_pos_w /
  root_lin_vel_w / root_ang_vel_w. _fk_ee_pose still uses sim body
  GT (Pinocchio swap stays Task #9).
- Visibility proxies are simple GT-pose occlusion rules — top camera
  blocked when EE is directly above the cube within 5cm; wrist sees
  it within 30cm.

EE link: gripper_frame_link survives in the URDF but not in the
USD conversion (only its fixed-joint parent mujoco_base_link and the
two dummy children get dropped). gripper_link, the wrist_roll child,
is the nearest surviving body, so it's the chosen EE for now. The
~9cm offset to the actual fingertip is a known approximation; an FK
offset can compensate once accuracy matters.

SimulationContext is a Carb singleton — one CubeLiftEnv per process.
Multi-env support follows the ManagerBasedRLEnv migration in Phase 2.

Verified inside a headless AppLauncher: ENV_INIT_OK (ee_body=5,
arm=[0..4], grip=5), OBS_ALIGN_OK, RESET_OK with a 31-D float32 finite
obs, STEP5_OK over five random actions (per_condition flags behave as
expected: lift/stable_placement/visual_agreement False under random
actions; release_retreat/velocity_stability flip True as the gripper
target sweeps and the cube stays still). ruff all-checks-passed.

---

## 2026-05-16 · `f1793fb` · 김건호

Land oracle MP + livestream pipeline (WIP debug code in)

Oracle (scripts/collect_demos.py + src/khj_rl/envs/cube_lift/oracle.py):
- 7-phase MP oracle (approach → descend → grasp → lift → move → place → retreat) driving Isaac Lab DifferentialIKController in DLS mode.
- Multi-pass IK fixes: jacobian root-DOF offset (+6) for floating base, lambda 0.01→0.1 for stability, per-step clip to soft joint limits to stop the solver from demanding > 11 rad joint targets at singular configs.
- Phase exit_distance relaxed 1.5 → 3 cm because 5 DOF arm + 3 D position task oscillates and never converges to a tight threshold; wrist_flex init pulled to -1.2 so the gripper starts pointing down (removes one redundancy direction in approach).
- Per-step debug prints (cur_arm, ik_target, |delta|max) kept on; oracle still not closed (IK divergence over many descend steps remains).

Env (src/khj_rl/envs/cube_lift/env.py + scene.py):
- Optional viewer_camera (CameraCfg under /World/ViewerCamera) wired through scene cfg subclass; env.py exposes self._viewer_camera and stamps the camera USD xform translate/orient directly because CameraCfg.OffsetCfg in this isaacsim build silently leaves cam.data.pos_w at (0,0,0).
- One-time structural diag printed at env init: joint_names, body_names, ee_body_idx, jacobian shape, per-arm-joint d_ee/d_q with soft limits — proves jacobian indexing is correct and confirms the arm hits soft limits during oracle.
- env.step toggles render=True on the last decimation tick so a viewer camera buffer refreshes once per policy step.
- action_delta_max 0.05 → 0.02 for slower, more stable IK tracking.

Articulation:
- Gripper actuator stiffness/effort bumped (40 → 400 / 5 → 50) so the PincOpen 4-bar can hold a 50 g cube; arm gains still placeholder TODOs.

Mimic stamping (scripts/usd_add_mimic_api.py):
- MIMIC_DIRECTION signs solved from the PincOpen mujoco <equality> block (left_proximal = +gripper, left_distal = -gripper, right_proximal = -gripper, right_distal = +gripper) and re-stamped onto the four 4-bar passive joints.

Livestream pipelines (scripts/launch_viewer.py + launch_ws_viewer.py + viewer_client.html):
- launch_viewer.py: NVIDIA's isaacsim.exp.full.streaming.kit experience over WebRTC 2.x — boots cleanly with the official Isaac Sim WebRTC Streaming Client (v1.0.6). Kit args pin renderer to GPU 1 (CUDA_VISIBLE_DEVICES does not cover the renderer, which uses NVML).
- launch_ws_viewer.py: alternative MJPEG-over-HTTP viewer for VSCode tunnel users; sweep / oracle / random / zero modes, frame-index overlay, replicator render product attempt left in place for future fabric-less render debugging.
- Known limitation: omni.physx.fabric 106.3.2 ABI mismatch (v0.2 requested vs v1.2 provided) prevents physics → USD auto-sync, so the streaming viewport currently shows the reset pose only. Tried updateToUsd / set_world_poses / force_recompute paths; none refresh on this install. Next session: clean Kit user cache (~/.local/share/ov/) per Codex review note.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-16 · `dc11511` · 김건호

Implement BCTrainer + PPOTrainer with shared ActorCritic

Phase 1 single-env PPO with BC pretrain bridge (CLAUDE.md ## Phase 1
결정 사항 7). Five guards that the BC -> PPO transition usually breaks
are all wired here:

1. Shared ActorCritic — BC and PPO import the same class. PPOTrainer.load_bc()
   reads the BC checkpoint's full state_dict directly, so weights land
   in matching shapes.
2. actor_logstd init = -1.0 (std ≈ 0.37) so the first PPO rollout doesn't
   destroy the BC-pretrained actor.
3. Critic warm-up — first 15 PPO iters freeze the actor + logstd
   (requires_grad=False, loss = vf_coef * v_loss only). Skip the policy
   forward graph too so the warm-up path is cheap.
4. PPO LR = 1e-4 (lower than the usual 3e-4) per CLAUDE.md guidance for
   post-BC fine-tuning.
5. Frozen ObsNormalizer — ObsNormalizer.fit() is only called during BC.
   PPO loads the same mean/std and never updates them; running normalization
   would shift the obs distribution out from under the BC actor.

Files:
- network.py: ActorCritic (2x256 tanh, orthogonal init, get_action_and_value
  parity with CleanRL ppo_continuous_action)
- normalizer.py: ObsNormalizer (frozen mean/std, JSON sidecar)
- demo_buffer.py: NPZ loader, success-only filter, minibatch iterator
- bc.py: BCTrainer (MSE on actor mean, actor + logstd only, critic left at
  random init for PPO warm-up to learn)
- ppo.py: CleanRL-style single-file PPO (rollout buffer, GAE, PPO clip,
  value clip, target_kl early-stop, env.step 4/5-tuple compat, gymnasium
  reset (obs, info) compat)
- __init__.py: re-export new public symbols

Codex review pass: 10/12 PASS, 1 efficiency WARN (warm-up pg_loss skip,
addressed), 1 reported FAIL on done-mask off-by-one that re-reading the
code shows is the standard CleanRL pattern (done_buf[t] = next_done means
"this step starts from a fresh reset", GAE uses dones[t+1] to mask the
value[t+1] bootstrap — correct).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-16 · `3bdc095` · 김건호

Oracle IK: null-space bias toward joint_init + safer descend height

Phase 1 oracle still doesn't produce successful grasps end-to-end, but
two changes here move the failure mode from "arm pins itself at a joint
limit during descend" (0/N reach grasp) to "arm reaches place phase but
finger contact pushes the cube forward before close" (0/5 reach a clean
grasp). Logged for next-session visual debugging.

1) Replace Isaac Lab DLS controller with a custom DLS + null-space bias:

      q_dot = J^T (J J^T + lambda^2 I)^-1 x_err              # primary
              + (I - J^T (J J^T + lambda^2 I)^-1 J) k (q_home - q)   # secondary

   The 5-DOF arm has 2 redundancy directions vs the 3-D position target;
   the stock min-norm DLS solution let those redundant DOFs drift toward
   soft limits over the 40+ descend steps, wedging the arm. The
   null-space term projects a "stay near joint_init" gradient onto the
   redundant directions only, leaving primary EE tracking untouched.
   `null_bias_gain=0.15` was the smallest value that prevented limit
   drift in smoke; 0.5 over-constrained lateral reach and missed the
   cube in y.

2) Bump descend phase target z from cube_z+0.05 to cube_z+0.10 (10cm
   clearance above cube top, fingertips ~7cm above cube). With the old
   5cm offset the PincOpen fingertips landed right at cube height while
   IK was still adjusting laterally; smoke showed the arm shoved the
   cube 18cm forward in x before grasp could close. Grasp phase keeps
   the 5cm offset and a looser 4cm exit so it still completes when the
   cube nudges 1cm laterally as fingers close.

Open issue (next session): arm STILL gets close to the cube but not
centered when grasp closes, and lift sees the fingers slip off. Without
a live render we're guessing at the geometric mismatch — fabric/USD
sync fix in viewer is the unblocker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-16 · `b23d95a` · 김건호

Wire Pinocchio FK into _fk_ee_pose

CLAUDE.md ## Phase 1 결정 사항 4 / Task #9: ee_pose_base in the obs vector
must be a pure function of the (potentially noisy) joint_pos so sim2real
transfer doesn't depend on body_pos_w (a sim-only buffer).

Build a fixed-base Pinocchio model from
assets/converted/urdf/so101_pincopen_gripper.urdf in CubeLiftEnv.__init__,
cache the q-vector index map for the 5 arm joints + gripper main + 4 mimic
passives, and rewrite _fk_ee_pose to:

  1. Stuff the 5-D arm vector into its q slots (cfg.robot.joint_names order).
  2. Pin gripper main + mimic at 0 — the gripper assembly rotation doesn't
     translate gripper_dummy_link significantly, and the obs already carries
     gripper_state separately.
  3. pin.framesForwardKinematics → read oMf[gripper_dummy_link].
  4. Return xyz + quat (w, x, y, z) — same 7-D layout as _sim_ee_pose_gt
     so reward / oracle callsites keep working unchanged.

URDF root joint is FIXED (no floating base joint between "universe" and
shoulder_pan), so Pinocchio's world frame coincides with the robot base
— ee_pose_base ends up in the base frame for free.

Smoke verification against the sim's GT body pose at home pose
(joint_init = (0, -0.5, 1.0, -1.2, 0)):

  pinocchio fk: pos=[0.2074, -0.0002, 0.2081]
                quat=[0.9391, 0.0229, -0.3428, 0.0083]
  sim gt:       pos=[0.2074, -0.0002, 0.2081]
                quat=[0.9391, 0.0229, -0.3428, 0.0083]
  pos err: 0.00000m

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `76f3520` · 김건호

Unblock livestream: use_fabric=True, fix_root_link, pinocchio pre-import

Three coupled fixes that together make the WebRTC viewport reflect actual
arm motion instead of the env.reset pose forever.

1) use_fabric=True in CubeLiftEnv's SimulationContext. Without fabric the
   PhysX state never mirrors into USD prim transforms, so the streaming
   viewport (which renders from USD) freezes on the initial scene. The
   blocker that originally forced use_fabric=False was an ABI mismatch
   (v0.2 vs v1.2) inside omni.physx.fabric-106.3.2 — Kit fell back to
   that broken extension because the working 106.5.3 fabric had
   `omni.physx = 106.5.3 (exact)` in its dependency manifest, and the
   installed core is 106.5.7. CLAUDE.md now ships the one-line sed
   patch that relaxes 106.5.3's pin to `version="106.5", exact=false`;
   both the conda site-packages copy and the ~/.local/share/ov/ cache
   copy must be patched (the cache is what Kit's extension manager
   actually reads on subsequent boots).

2) fix_root_link=True on the SO-ARM101 articulation. The URDF root joint
   is fixed (no floating-base joint between universe and shoulder_pan)
   but Isaac Sim still treated the base_link as a free body — random
   action smoke tests with use_fabric=True showed the whole robot
   toppling over within seconds. fix_root_link nails the base to its
   spawn pose so the actuators move the arm and nothing else.

3) Pre-import pinocchio in every entry-point script
   (launch_viewer / launch_ws_viewer / collect_demos) before
   AppLauncher boots. pinocchio's libhpp-fcl.so links against the
   system Assimp; isaacsim ships its own Assimp build whose C++ symbol
   mangling for Assimp::IOSystem::CurrentDirectory differs. When
   isaacsim wins the load race, pinocchio's import then fails with an
   undefined-symbol error and CubeLiftEnv (which uses pinocchio for
   _fk_ee_pose) refuses to construct. Reversing the load order — system
   Assimp first via pinocchio, then isaacsim — lets both coexist.

Verified end to end against the official Isaac Sim WebRTC Streaming
Client (v1.0.6): random-action rollout shows the arm waving visibly
above a static base while the viewport stays live.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `39e6fea` · 김건호

Fix oracle gripper control for cube-lift Phase 1

Override gripper/mimic joint limits at runtime to bypass the USD
degrees-parsing bug (URDF radian limits get re-interpreted as
degrees by Isaac Lab's articulation parser, clamping the gripper to
a 1.5° micro-range). Drive the PincOpen 4-bar mimic joints via PD
actuators instead of per-tick write_joint_state_to_sim — the old
direct write fought PhysX's contact resolver and produced a high
frequency cube vibration with no visible finger contact. Update the
fingertip-center FK to read sim body_pos_w plus a STL-bbox-derived
local tip offset rotated through body_quat_w, so the IK target
lands on the actual grasp contact point rather than the
proximal-distal hinge.

Tune the top-down home pose (sl=-0.30, ef=+0.20, wf=+1.40 — picked
by a Pinocchio FK sweep constrained to distal-X axis aligned with
world -Z), the oracle grasp sequence (descend/grasp both target
cube center now that the arm can reach), and the cube contact
offsets (2 mm rest, 0 mm phantom) so the pick-and-place completes
end-to-end. Gripper close range capped at the fingertip-Z minimum
gripper=-1.0 found by the same FK sweep (spread 3.34 cm) — driving
past that arc-rotates tips back above the cube and the close ends
in air.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `dea1f40` · 김건호

Latch lift_history; tighten gripper visual + cfg

Replace the rolling-window lift check with a per-episode latch:
once the cube holds z >= 8 cm for 20 consecutive steps the
condition stays True so stable_placement and release_retreat
can still reflect end-of-episode requirements. Rolling window
made the five success conditions structurally impossible to
satisfy simultaneously in a PnP. Velocity stability (cond 4)
retains the jabis_sim_v2 rolling-cube guard.

Narrow the gripper joint range from [-1.0, +1.07] to [-1.0, +0.5]
to match the user's real-PincOpen visual reference, and route it
through new cfg.RobotCfg fields so soft_joint_pos_limits and
env.step's action mapping share a single source. Bump the 4-bar
mimic actuator gains (effort 5→50, stiffness 300→2000) so the
PD tracks gripper * multiplier without lag — the visible jaw
geometry now stays in the PincOpen parallelogram shape during
fast open/close. Cap mimic velocity_limit at the gripper main
joint's 1.5 rad/s so the 4-bar doesn't snap ahead of the drive.

Add episode-end per_condition print in launch_viewer.py for
success-criterion debugging. CLAUDE.md Phase 1 결정 사항 5 +
2 updated for the latching semantic and the cfg-driven gripper
mapping.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `4f43b59` · 김건호

Calibrate gripper range from real STS3215 measurements

Pin the cfg.RobotCfg gripper joint target range to the user's
caliper measurements on the right finger: encoder 1175 -> 79 mm
spread (sim gripper +0.150), encoder 2031 -> 44 mm (-0.750),
encoder 2887 -> 4 mm (-1.800). Range solved by a Pinocchio FK
sweep over the top-down home pose matching the left/right distal
tip midpoint to each measured spread. encoder midpoint maps
within 0.075 rad of the gripper-joint midpoint -- close to linear
despite the 4-bar's curved kinematics.

This makes the sim's action=-1 / +1 produce the same physical jaw
shape as the real robot at its OPEN / CLOSE encoder extremes, so
the deployment driver only needs a linear action -> encoder
mapping (no spread-vs-encoder lookup table). Mimic joint target
range bumped to [-1.0, +1.0] to fit the new |0.5 * 1.8| = 0.9
mimic swing.

CLAUDE.md Phase 1 결정 사항 2 updated to record the encoder ->
sim-joint correspondence as the calibration source.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `d88af6e` · 김건호

collect_demos: drop oracle.done early exit, log term/trunc

The oracle's ``done`` property flipped True ~15 steps into the
retreat phase, ~20 steps before the MultiConditionSuccess judge
had collected enough ``stable_placement`` history to flip its
latch — so every demo broke out of the loop a hair before
``env.step`` would have returned ``terminated=True``. Removing
the ``oracle.done`` short-circuit lets the loop run until the
env actually terminates or hits the 160-step truncation.

A sanity test (5 episodes) now collects 4/5 successes; the one
failure had ``lift_history=0`` (cube slipped during lift before
the 1 s @ z>=8 cm latch). Episode log now prints final
term/trunc and per-condition status to make future failures
easier to diagnose.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `d4bc62d` · 김건호

Oracle: lift dwell + side retreat for 100% PnP success

Two changes that push diagnostic success rate from 50% to 20/20:

1. Lift / move ``min_steps`` — lift was finishing in ~25 steps
once the fingertip reached z=12cm, and move exited a few steps
later, so the cube was only above the 8cm threshold for ~12-16
frames. The success.py rolling buffer needs 20 consecutive
``z >= 8 cm`` frames to latch lift_history, so the latch never
fired and every "lift looked fine to a human" episode still
got reward=0. Set lift min_steps=30 (= 1.5 s dwell at z=12 cm)
and move min_steps=10 so the cube spends well past 1 s above the
threshold before place starts the descent.

2. Retreat target side-step — the old retreat went straight up
above the placed cube to z=18cm, which triggered the
visual_agreement top-camera proxy's "EE within 5cm xy AND above
cube" occlusion test on every episode. Step 8cm back in -X
(staying inside the 30 cm wrist-cam radius) clears the
occlusion check without affecting release_retreat distance.

20/20 success on the diag run with both fixes in place.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `5f11f95` · 김건호

Phase 1 eval guards + BC/PPO training wrapper scripts

CLAUDE.md ## Phase 1 결정 사항 8 mandates two safety guards before
BC / PPO training can land. Both are implemented now:

- video_sampler.py — periodic rollout MP4 capture via imageio +
  the env's third-person viewer_camera. Triggered every N steps
  by the trainer; renders rollout_episodes episodes through the
  policy and writes to runs/{run_name}/videos/rollout_step_*.mp4
  at the 20 Hz policy rate so an operator can scrub for the
  reward-hacking failure mode that fooled jabis_sim_v2 (97.9%
  success rate with cube rolling).

- obs_alignment.py — adds ``assert_obs_distribution`` on top of
  the existing structural check. Samples n_samples env resets
  and verifies every obs dim sits inside the BC training
  distribution (mean ± sigma_tolerance × std). Catches sim2real
  wrapper drift before PPO can blow up the actor with garbage
  advantages.

Two new launcher scripts:

- scripts/train_bc.py — runs BCTrainer in a torch-only process
  with a ShapesOnlyEnv (BC never calls env.step). Pinocchio /
  Isaac Sim aren't pulled in so this is cheap to iterate on.

- scripts/train_ppo.py — boots Isaac Sim with the same env path
  as collect_demos / launch_viewer (pinocchio pre-import + GPU 1
  pin), constructs PPOTrainer, optionally loads a BC checkpoint
  (load_bc copies actor / critic / frozen normalizer), runs the
  two guards, then kicks off training. Video sampler hook is
  surfaced as --video-every-steps but not yet wired into
  PPOTrainer.train() (left as a follow-up so this commit lands
  alongside demo collection rather than blocking it).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `a77b691` · 김건호

demo_buffer: read 'actions' key matching writer schema

DemoBuffer.from_dir reached for ``data["action"]`` but the writer
(khj_rl.data.demos.DemoBuffer.write -> np.savez_compressed) stamps
the key as ``actions``. KeyError fired at BC trainer init. Fix the
loader key and refresh the module docstring's schema block to the
real layout (also adds rewards / stage / ep_id that the writer
emits but the docstring was missing).

Smoke-tested on stage0 (175 episodes, 5 epochs, loss 0.19 -> 0.04):
trainer runs end-to-end, bc.pt + norm.json + bc_meta.json all
written at runs/stage0_bc_smoke/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `62b3b8d` · 김건호

PPO: wire video_sampler hook into train() loop

The previous commit landed eval/video_sampler.py but left the
PPOTrainer wired up to its own loop with no place to call it.
Take the optional ``video_sampler`` argument on the constructor;
inside ``train()``, at every iteration boundary, hand it the
current global_step + the training env + a deterministic eval
policy (actor mean only, no exploration noise, frozen normalizer
applied). When the sampler fires it owns the env for a few
episodes — we always reset before resuming the training rollout
so the trainer's ``next_obs`` doesn't desync from sim.

scripts/train_ppo.py now actually constructs a VideoSampler when
``--video-every-steps > 0`` (and only then asks the env for the
viewer camera, so the renderer cost is opt-in). The placeholder
"argument accepted but ignored" warning is removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-17 · `8f28ca4` · 김건호

Address Codex review of eval guards + video sampler wiring

Four follow-ups from the post-commit review:

1) video_sampler trigger never fired. The old
``step % every_steps == 0`` check assumed step increments of 1,
but PPO advances global_step in chunks of num_steps (default
2048); with every_steps=5000 the values 0/2048/4096/6144/...
never line up exactly so capture silently skipped forever.
Switch to a threshold check against ``_last_capture_step``.

2) ``assert_obs_distribution`` masked degenerate dims. ``np.where
(std > 1e-6, std, 1.0)`` turned every "constant" obs dim (e.g.
``normalized_t`` is always 0 at reset) into a no-op z-score
check. The very dims a sim2real wrapper would most obviously
shift were being skipped. Run z-score on healthy dims and an
absolute-tolerance check (default 1e-3) on degenerate ones, and
include the degenerate label in the failure breakdown.

3) ``_make_eval_policy`` toggles ``self._net.eval()`` for the
forward pass and restores ``train()`` after. ActorCritic has no
Dropout/BatchNorm today so the toggle is a no-op, but a future
layer-swap would otherwise silently corrupt every captured
rollout.

4) ``scripts/train_bc.py`` pinned ``TrainerConfig.total_steps``
to ``args.epochs``. BCTrainer doesn't read the field at all,
and the implicit "epochs == steps" wording invites
misinterpretation in logs that might pick it up later. Set
total_steps=0 with a comment so the irrelevance is explicit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-18 · `45a80eb` · 김건호

Phase 1: ACT lands at 73% after PPO A/D failures

Two-day arc:

5/17 — PPO A (BC 3k + bounded·annealed·phase-gated dense shaping, 500k step).
  - 0/100 success, lift_history 0% (BC baseline 36% → 0%로 망가짐)
  - w_ee_cube_dist=0.5 항이 phase gate 없는 거리 minimization
  - lift는 어렵고 EE를 cube에 박는 짓누르기는 쉬워 정책이 attractor 정착
  - decay_steps=300k에서 annealing 0 도달 시 ep_return collapse
  - 영상 검증으로 짓누르기 패턴 확인 (D 옵션 재발)

5/18 — ACT (Action Chunking, k=12) BC.
  - 1-step BC는 covariate shift에 취약 (19%에서 정체)
  - PPO 패러다임은 sparse 환경에서 BC weight 못 지킴 (D/A 모두 0%)
  - ACT가 모델 구조 차원에서 표류 차단
  - L1 loss, temporal ensemble α=0.2, gripper_use_latest=True
  - Codex plan-review 후 REVISE 5개 권고 반영 + final-review PROCEED

결과 (BC ACT 단독, stage 0):
  - 73/100 success (시나리오 A)
  - lift_history 100% / release_retreat 100% / stable_placement 100% / velocity_stability 100%
  - visual_agreement 73% (proxy strictness, 영상 검증 후 정책 결함 아님)
  - 영상 6 ep: 5 succ, reward hacking 0건

Infra:
  - scripts/{auto_phase1_pipeline, auto_phase1_watcher, analyze_phase1_results, health_monitor}.py
  - scripts/rollout_video.py — ChunkBuffer-aware third-person camera 영상 + thumbnail
  - scripts/eval_policy.py — chunking 메타 hard assert + temporal ensemble inference
  - docs/{HANDOFF_2026-05-17, HANDOFF_2026-05-18, action_chunking_design, dagger_design,
          dense_reward_design, dense_reward_weight_tuning, phase2_migration_plan,
          real_robot_deployment, troubleshooting}.md

Bonus fixes:
  - scene.py goal_marker: opacity 0.6→1.0, height 2mm→5mm, z +1mm→+3mm (렌더 visibility)
  - bc.py _json_safe: dataclass 직렬화 (ActionChunkingCfg 저장 사고)
  - DemoBuffer per-episode span tracking + precomputed valid_chunk_starts (off-by-one 방지)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-18 · `6b8d963` · 김건호

Phase 1 ACT v2: stage 0 90%, stage 1 60% via curriculum-aware demo

curriculum-stage CLI + cfg

cfg.CurriculumCfg.current_stage_idx (default 0) exposes the active stage
index as a single source. env.py reads side_length_m[idx] when sampling
cube xy; --curriculum-stage flag added to eval_policy.py /
rollout_video.py / collect_demos.py so callers can advance the
curriculum without code edits.

collect_demos: --max-success (early stop on N successful demos written).
Stage 1 collects at ~22 s/ep, so a 1000-attempt run is ~6 h; cap the
budget instead.

ACT v0/v1/v2 progression

v0 (runs/stage0_bc_act_k12, stage0 demos only): stage 0 73% (시나리오 A),
stage 1 33% (out-of-distribution drop).

v1 (runs/stage01_bc_act_k12, stage0 + stage1=359 merged): negative
transfer. stage 1 → 26%, stage 0 → 67%. Ratio 8:1 (stage0 dominates), so
stage 1 grasp accuracy stays under-learned and the stage 0 release/visual
patterns get polluted.

v2 (runs/stage01v2_bc_act_k12, stage0 + stage1=1059 merged): positive
transfer. **stage 0 90%** (+17pp vs v0 stage-0-only), **stage 1 60%**
(+34pp vs v1). Ratio 2.7:1. Distribution variety also bumps stage 0
visual_agreement 73 → 91%. Six-episode video sample: 5/6 success, fail
case (ep003) is "lifted + moved but placement 3 cm off" — distinct from
v1's "push only, never grasped" pattern.

Per-cond comparison: stage 1 release_retreat v1=59 → v2=82 (+23pp),
visual_agreement v1=47 → v2=81 (+34pp).

Lesson: when adding a new curriculum stage's demos, keep new-stage count
≥ ~1/3 of existing demos. Naively appending a small batch (8:1 ratio)
regresses both stages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-18 · `dec13e5` · 김건호

RL track attempt: PPO + BC v2 fails; ACT trick exposed

Sequence:
1. User redirect: RL is the differentiating track (team has BC done).
   Drop BC ACT extension work, return to RL paradigm.
2. PPO + BC v2 pretrain attempt
   - ppo.py load_bc(): branch into ChunkedActorCritic when ckpt meta says
     chunking_enabled=True; rebuild optimizer for new param set.
   - train_ppo.py --actor-logstd-override: shrink stochastic noise so
     PPO sampling doesn't kick the BC pretrain off-manifold.
   - Two pilots (logstd -1.0, -2.5) both ABORT — 24 iter ep_return=0.
3. Root-cause probe
   - eval_policy.py --no-chunk-buffer: disable temporal ensemble +
     gripper_use_latest so we measure actor_mean(chunk[0]) alone.
   - BC v2 stage 0: 90% (CB on) → 5% (CB off). visual_agreement
     91 → 12. All conditions regress -29~-79pp.
   - PPO uses actor_mean only, so the real starting point is 5%, not
     90%. Stochastic noise + sparse reward → ep_return=0 inevitable.

Implication:
- "BC ACT 90%" is mostly an inference trick (temporal ensemble +
  gripper override). Real policy is weak; deployment without
  ChunkBuffer is unsafe.
- PPO/SAC fine-tune on top of ACT BC is structurally broken without a
  ChunkBuffer-aware learner. Next RL attempt should start from a fresh
  1-step BC (no chunking) or go SACfD with off-policy demo replay.

Docs:
- HANDOFF_2026-05-18: night chapter — RL pivot, attempts, root cause,
  next-card matrix.
- troubleshooting.md: 5 new entries (#19 ACT trick, #20 GPU 1 conflict
  with lerobot, #21 PPO ep_return=0 reasoning, #22 chain bash blocked
  on close-hang, #23 BCConfig dataclass JSON serialize).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---

## 2026-05-28 · `d805666` · 김건호

Backup WIP before server shutdown: cube_lift env tuning, design docs, chain scripts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

---
