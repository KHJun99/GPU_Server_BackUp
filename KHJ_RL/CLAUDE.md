# KHJ_RL — 개인 강화학습 프로젝트

## 배경

`jabis_sim_v2`에서 진행하던 PnP 강화학습이 reward hacking으로 끝났음 (97.9% 성공률인데 실제 영상은 큐브를 굴려서 목표에 도달). 패치보다 처음부터 재설계가 빠르다고 판단해 이 repo를 새로 시작.

- **성격:** 개인 트랙. 팀 프로젝트는 RL을 포기했지만, GPU 서버 시간 활용을 위해 본인이 계속 진행
- **참고 repo:** `/home/j-k14d101/jabis_sim_v2/` — 코드를 직접 복사하지 말 것. 인터페이스를 거쳐 재구현
- **시작일:** 2026-05-15

## 설계 — 4개 확장 축

처음부터 확장성을 내장. Phase 진행 중에 framework를 깨지 않도록 stub만으로라도 자리를 잡아둠.

| 축 | 위치 | 책임 |
|---|---|---|
| Object | `khj_rl.registries.ObjectRegistry` | shape, size, mass, friction |
| Target | `khj_rl.registries.TargetRegistry` | single pose / multi-bin / arbitrary pose |
| Learner | `khj_rl.training.Trainer` (ABC) | PPO / BC / Diffusion / Hybrid 교체 |
| Sim2Real | `khj_rl.sim2real.{RandomizationConfig, NoiseModel}` | Phase 1부터 환경에 내장 |

## Phasing

- **Phase 0 (완료)** — 인터페이스 골격, 구현 없음
- **Phase 1 (다음)** — 1물체 × 1목표 × PPO × Sim2Real (noise/randomization 내장)
- **Phase 2** — Object 축 확장 (+curriculum)
- **Phase 3** — Target 축 확장 (multi-bin, arbitrary pose)
- **Phase 4** — Learner 축 확장 (BC, Diffusion, Hybrid)

## 현재 상태 (Phase 0)

```
src/khj_rl/
  envs/            # placeholder (Phase 1에서 첫 env 도입)
  registries/
    objects.py     # ObjectSpec + ObjectRegistry
    targets.py     # TargetSpec + TargetRegistry
  sim2real/
    randomization.py  # RandomizationConfig + RangeSpec
    noise.py          # NoiseModel
  training/
    base.py        # EnvLike Protocol + TrainerConfig + Trainer ABC
  eval/
    video_sampler.py    # 주기적 rollout 영상 저장 (stub)
    obs_alignment.py    # env↔policy 관측 정렬 가드 (stub)
```

`configs/`, `docs/`, `scripts/`, `tests/` 폴더는 비어 있음.

`git`: 기본 브랜치는 `master` (개인 트랙 관례에 맞춤). remote 없음.

## Guardrails — 이전 사고의 교훈

이 가드들은 인터페이스로만 존재하지만 Phase 1에서 반드시 작동시켜야 함.

1. **Video auto-sampling** (`eval/video_sampler.py`)
   - PnP 97.9% reward hacking은 success metric만 보고 영상을 안 본 탓. 학습 루프에서 N step마다 rollout 영상 자동 저장.

2. **Obs alignment check** (`eval/obs_alignment.py`)
   - `jabis_sim_v2`에서 sim2real wrapper의 hardcoded baseline joint 상수가 env baseline과 어긋나 정책이 학습 분포 밖으로 나간 사고 있었음. 세션 시작 시 env reset obs == 정책 학습 분포 검증.

3. **Single source of truth for env constants**
   - sim2real wrapper나 reward에 환경 상수를 hardcode 금지. 모든 baseline/limit은 `env_cfg` 한 곳에서.

## 개발 환경

```bash
conda activate khj-rl       # Python 3.10 (요구: 3.10.x)
# khj_rl는 editable install 완료 (src layout)
```

### 배포 타겟 두 곳
- **학습 호스트:** x86_64 + NVIDIA L40S 46 GB. 서버에 GPU 4장 있지만 **GPU index 1**만 사용. `CUDA_VISIBLE_DEVICES=1`로 격리하고 single-GPU PPO로 작성(분산 학습 X). driver는 CUDA 12.8.
- **추론 타겟:** Jetson Orin Nano (aarch64, JetPack 6). 정책 추론 + perception을 여기서 돌릴 수 있어야 함.

→ `pyproject.toml`의 의존성은 양쪽 모두 호환되도록 설계.
기본 `dependencies`는 양쪽 PyPI에서 받을 수 있는 cross-arch 패키지만,
플랫폼별 패키지(torch, pyrealsense2)는 `[host]` extras로 분리.

### 설치
**호스트 (x86_64 + CUDA 12.8 driver, L40S 기준 — Phase 1 학습 서버):**
```bash
# 1) torch는 cu128 index로 명시 설치 (Isaac Lab 0.47.x가 torch>=2.7 강제, cu128가
#    시스템 driver 12.8과 best fit).
pip install torch==2.7.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

# 2) khj_rl + host/dev extras.
pip install -e ".[dev,host]"

# 3) Isaac Sim 4.5 + Omniverse Kit.
pip install "isaacsim[all]==4.5.0.0" --extra-index-url https://pypi.nvidia.com

# 4) Isaac Lab editable install (현재 위치: /home/j-k14d101/jabis_sim/IsaacLab/source).
#    setuptools<81 + flatdict + toml을 root에 박아두고 --no-build-isolation 사용
#    (pip의 build isolation이 매번 setuptools 81을 끌어와 flatdict / isaaclab 빌드를 깨뜨림).
pip install "setuptools<81" wheel build toml flatdict
pip install --no-build-isolation \
  -e /home/j-k14d101/jabis_sim/IsaacLab/source/isaaclab \
  -e /home/j-k14d101/jabis_sim/IsaacLab/source/isaaclab_assets \
  -e /home/j-k14d101/jabis_sim/IsaacLab/source/isaaclab_rl \
  -e /home/j-k14d101/jabis_sim/IsaacLab/source/isaaclab_tasks \
  -e /home/j-k14d101/jabis_sim/IsaacLab/source/isaaclab_mimic
```

**환경변수 (kit 부트 시 필수):**
- `OMNI_KIT_ACCEPT_EULA=YES` — 최초 import 시 인터랙티브 EULA 프롬프트 우회 (headless)
- `PRIVACY_CONSENT=Y` — telemetry consent 프롬프트 우회
- `CUDA_VISIBLE_DEVICES=1` — GPU index 1 격리

**검증:**
```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y python -c "
from isaaclab.app import AppLauncher
sim_app = AppLauncher(headless=True, enable_cameras=False).app
from isaaclab.envs import ManagerBasedRLEnv
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
sim_app.close()
"
```


**필수 사후 패치 — `omni.physx.fabric` 의존성 완화 (호스트):**

설치된 Isaac Sim 4.5 번들은 `omni.physx.fabric-106.5.3`이 `omni.physx==106.5.3, exact=true`를 요구하는데 실제로 깔린 코어는 `omni.physx-106.5.7`임 — exact pin 매치 실패 → Kit이 옛 `omni.physx.fabric-106.3.2`로 fallback → `IPhysxPrivate v0.2 vs v1.2` ABI 충돌로 fabric 못 올림 → `use_fabric=True`가 raise. 풀려면 dependency 한 줄 완화:

```bash
F1=/home/j-k14d101/.conda/envs/khj-rl/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3/omni.physx.fabric-106.5.3+106.5.0.lx64.r.cp310.ub3f/config/extension.toml
F2=/home/j-k14d101/.local/share/ov/data/exts/v2/omni.physx.fabric-106.5.3+106.5.0.lx64.r.cp310.ub3f/config/extension.toml
for F in "$F1" "$F2"; do
  cp "$F" "$F.bak"
  sed -i 's|"omni.physx" = { version = "106.5.3", exact = true }|"omni.physx" = { version = "106.5", exact = false }|' "$F"
done
```

이거 안 하면 WebRTC livestream viewport가 env.reset 시점의 USD 상태에 멈춰 — physics 진행해도 시각 안 바뀜. 패치 후엔 use_fabric=True가 정상 동작 + viewport가 매 step 갱신됨. `.bak`로 백업 자동.

**Pinocchio + Isaac Sim Assimp 충돌:**

`pinocchio`의 `libhpp-fcl.so`가 시스템 Assimp에 링크돼있는데, Isaac Sim은 자체 Assimp 빌드를 가져옴 → C++ 심볼 mangling 충돌 (`Assimp::IOSystem::CurrentDirectory`). isaacsim이 먼저 로드되면 pinocchio import가 undefined-symbol로 실패. **모든 entry-point script (`launch_viewer.py`, `launch_ws_viewer.py`, `collect_demos.py`)는 `AppLauncher` 전에 `import pinocchio` 먼저** — system Assimp가 먼저 바인딩되면 동거 가능.

**Jetson Orin Nano (aarch64 + JetPack 6):**
```bash
pip install -e ".[dev,jetson]"   # extras는 비어 있음
# torch: NVIDIA 공식 PyTorch for Jetson wheel을 별도 다운로드해 설치
#   https://forums.developer.nvidia.com/  (JetPack 버전에 맞는 wheel)
# pyrealsense2: librealsense를 source build하여 Python binding 활성화
#   https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
```

### 스택 결정
- **Sim:** Isaac Lab (NVIDIA Isaac Sim 별도 설치, pip 의존성 외부, **호스트에서만** 사용)
- **RL:** CleanRL 단일 파일 PPO를 repo 내부에 직접 작성 (외부 cleanrl 패키지 X)
- **Perception:** OpenCV(HSV+depth), pyrealsense2(D456)

## Phase 1 결정 사항 (확정)

Codex 리뷰(2026-05-15)를 거쳐 확정. 변경 시 이 섹션을 단일 소스로 갱신.

### 1. 시뮬레이션 & 학습 스택
- **Sim 백엔드:** Isaac Lab (`jabis_sim_v2`와 동일 백엔드). **Phase 1은 현재 `envs/cube_lift/` 단일 패키지 유지**, Phase 2 진입 시 `ManagerBasedRLEnv` 패턴 + `envs/mdp/{observations, rewards, terminations, events}.py` 모듈 분리로 **점진 이행** (현재 scaffolding을 통째로 갈아엎지 않음)
- **PPO 구현:** CleanRL 단일 파일 스타일을 `khj_rl.training` 하위에 직접 작성
- **Control rate:** 20 Hz policy
- **Physics rate:** 120 Hz × decimation 6 → policy 20 Hz
- **Episode horizon:** 8 s = 160 step

### 2. 로봇 / 물체 / 좌표계
- **Robot:** SO-ARM101 **5-DOF arm + PincOpen gripper (메인 1축)** = 모터 6개
  - Arm 모터: **Feetech STS3215 × 5** (1/345 기어비) — shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
  - Gripper 모터: STS3215 1개 (모터 ID=6, baud=1000000) + 4-bar 평행 링키지의 passive joint 4개
  - PD gain은 **실 모터 측정 전까지 추측값** → 코드에 `TODO: measure on real motor` 주석
- **PincOpen 4-bar 처리 (USD):**
  - 4-bar passive joint 4개 (`left_proximal`, `left_distal`, `right_proximal`, `right_distal`)에 **`PhysxSchema.PhysxMimicJointAPI:rotZ` 적용 완료** (`scripts/usd_add_mimic_api.py`로 자동 stamp). reference joint = `gripper`, gearing = -1.0 (PhysX convention: `joint = -gear * ref` → 같은 방향). 부호가 어긋난 finger는 sim 보고 조정
  - `ActuatorCfg`에는 reference joint 1개(`gripper`)만 포함. mimic 조인트는 PhysX가 자동 추종 — **양쪽 다 actuator로 잡으면 drive 충돌**
  - URDF의 `<mimic>` 태그는 USD 변환에서 자동으로 넘어가지 **않음** → 위 스크립트로 USD에 직접 stamp 필수 (자산 재변환 시 매번)
- **Action space:** **6-D**, 모두 `[-1, 1]` 정규화
  - Arm: 5-D `JointPositionActionCfg(use_default_offset=True, scale=0.05)` → joint delta
  - Gripper: 1-D 선형 매핑 [-1, +1] → `cfg.robot.gripper_joint_target_{low,high}` 라디안 범위. 디폴트 **[-1.800, +0.150] rad** — 실 robot 우측 finger 측정값 (캘리퍼)으로 calibrate: encoder 1175 = spread 79mm = sim gripper +0.150 (OPEN), encoder 2887 = spread 4mm = sim gripper -1.800 (CLOSE), encoder 2031 = spread 44mm = sim gripper -0.750 (MID, 거의 선형). 같은 cfg 필드가 env init의 soft_joint_pos_limits override에도 사용되어 단일 소스. Sim action=-1 → real encoder 2887, action=+1 → real encoder 1175 (deployment driver layer에서 직접 매핑)
  - **Oracle 정책도 동일한 6-D 정규화 action을 출력** (BC↔PPO action 분포 일치)
- **자산 위치:** `assets/converted/urdf/so101_pincopen_gripper.urdf` + `assets/converted/usd/so101_pincopen.usd` (+ `configuration/{base,physics,sensor}.usd` 분리 layer). `assets/raw/`에는 출처 원본(TheRobotStudio/SO-ARM100, CNURobotics/pinc_open_driver) 보관용. 둘 다 `.gitignore`됨
- **Cube:** 45 mm matte cube, ~50 g, cyan 또는 lime 단색 (markerless 인식 용이)
- **좌표계:** 로봇 base frame이 유일한 universal frame. world frame 별도 캘리브레이션 금지

### 3. Perception (sim2real에서 real측)
- **Top view:** D456를 거치대로 위에 설치 (eye-to-hand)
- **Wrist view:** 각 팔의 wrist 카메라 (eye-in-hand)
- **큐브 좌표:** HSV color segmentation + depth deprojection (markerless)
- **AprilTag 사용 범위:** **EE에 부착하는 one-time hand-eye calibration 시에만**. 작동 중에는 큐브/EE 어디에도 marker 없음

### 4. Target & 관측
- **Phase 1 goal:** 단일 고정 goal + 큐브 시작 위치만 랜덤 (이전 계획의 "에피소드마다 random goal"에서 변경 — sparse reward와 random goal 동시는 학습 신호 부족)
- **obs 구성 (단일 소스 `env_cfg`):** 총 **31-D** (SO-ARM101 arm 5-DOF 반영)
  - `joint_pos` (5) — SO-ARM101 5 arm joint position (정규화)
  - `joint_vel` (5) — SO-ARM101 5 arm joint velocity (정규화)
  - `ee_pose_base` (7) — FK로 계산한 EE pose: xyz(3) + quat[w,x,y,z](4), robot base frame. sim/real 양쪽 호환을 위해 FK만으로 계산 가능한 신호로 한정
  - `cube_xyz_base` (3) — robot base frame
  - `target_delta_base` (3) = `goal_xyz − cube_xyz` (base frame)
  - `gripper_state` (1) — opening 0~1
  - `last_action` (6) — 직전 step action (5 arm delta + 1 gripper. BC·PPO partial observability 방지)
  - `normalized_t` (1) — `step / max_step` (160-step fixed horizon에서 time-dependent action 학습용)
- 모든 obs 항목의 frame·단위·정규화 범위는 `env_cfg`에 단일 소스로 둠

### 5. Reward — sparse only
shaped reward는 jabis_sim_v2 reward hacking 재발 위험이라 금지. **multi-condition success criterion**으로 +1만 부여:

| # | 조건 | 디폴트 임계값 (Phase 1 시작값) |
|---|---|---|
| 1 | lift history (**latching**) | 에피소드 중 cube가 20 step(=1s @20Hz) 연속 z ≥ 8 cm 유지한 적이 한 번이라도 있으면 이후 영구 True |
| 2 | stable placement | cube xy가 goal xy 반경 3 cm 이내로 `place_hold_steps` 동안 유지 (디폴트 20 step) |
| 3 | release + retreat | gripper open ≥ 0.8 AND EE가 cube로부터 6 cm 이상 떨어진 후에도 cube 위치 유지 |
| 4 | velocity stability | cube 선속도 < 5 cm/s, 각속도 < 30 °/s |
| 5 | visual agreement | top + wrist 카메라 둘 다 cube를 goal 영역에서 검출 |

다섯 조건이 success 평가 시점에 모두 True여야 success. 1번은 latching이라 episode 도중 lift만 한 번 충족하면 끝까지 유지 — 2/3/4는 episode 마지막 시점의 placement·release·정지를 본다. 이 분리가 PnP 시퀀스의 "들었다 → 옮겼다 → 놨다" 본질을 반영. 4번이 jabis_sim_v2 굴리기 reward hacking의 직접 차단조건.
임계값은 `env_cfg.success`에 키로 두고, 변경 시 그 한 곳만 수정.

- **dense shaping 정책:** Phase 1은 **sparse only 유지** (전면 dense shaping은 jabis_sim_v2 reward hacking 재발 risk라 금지). BC pretrain + goal-space curriculum 후에도 학습이 정체될 때에만 **별도 결정으로 bounded·annealed 보조항** 도입 검토 (작은 weight + 학습 진행에 따라 0으로 수렴). 이 도입은 결정사항 갱신 + codex 검토 필수
- **A 옵션 도입 결정 (2026-05-17)**: D(PPO hyperparam 보존) + B(BC 강화 to 19% only) 모두 본질 sparse signal 부재 해결 못 함 — A 옵션 발동. Codex 검토 완료, weight 조정 반영. 정확한 수식 + weight + rollback 기준은 `docs/dense_reward_design.md` 참조. 핵심 안전장치: bounded (α_0=0.01) + annealed (300k step에 0) + phase-gated (`* I(lifted)` 곱셈으로 cube 굴리기 hack 차단) + saturating (cube_goal_dist 0.1m cap) + 3중 조건 released_bonus (trivial release hack 차단). per-condition 회귀 검사로 lift_history 등 baseline 이상 유지 검증

### 6. Sim2Real (Phase 1부터 환경 내장)
- **NoiseModel 적용 위치:** sim env wrapper의 **obs post-processing 단계에서만**. reward 계산은 GT pose 사용 (sim 정답 신호 오염 방지)
- **NoiseModel:** cube pose에 xy σ=4 mm, z σ=8 mm Gaussian + 저확률 outlier jump + visibility dropout
- **Randomization (매 reset, `EventTermCfg`):**
  - 그리퍼 PD gain ±20%
  - finger pad 마찰 0.5~1.5
  - 물체 질량/크기 ±20%
  - 물체 초기 xy 위치 균등 분포 (curriculum side_length 안에서)
  - action delay 0~50 ms (USB + 모터 지연 모사)
  - obs noise: joint pos ±0.5°, joint vel ±10%
  - 추후 lighting / camera extrinsics / joint friction·damping 추가
- **visual agreement의 sim 구현:** Phase 1은 **GT pose + camera-visibility proxy**로 판정 (실제 RGB-D 렌더+HSV는 계산 비용·perception failure가 학습 병목이라 회피). proxy(occlusion 룰; e.g. EE가 가렸나, cube가 시야 frustum 안인가)는 `env.py`가 GT pose로 계산해 `MultiConditionSuccess.update(state)`의 `visibility=(top, wrist)`로 넘김. 실 렌더 기반 검증은 sim→real 전환 직전 별도 step
- 모든 baseline/limit 상수는 `env_cfg`에 단일 소스로 둠 (jabis_sim_v2 hardcode 사고 재발 방지)

### 7. 학습 부트스트랩 (필수)
- **BC pretrain:**
  - oracle = motion planning 기반 스크립트 정책 (top-down approach → grasp → 직선 lift → 직선 move → place → retreat)
  - demo 수량: 1000 episode부터 시작, 성공한 demo만 사용
  - workflow: oracle demo 수집 → BC pretrain → PPO fine-tune
  - 저장: `runs/demos/{stage}/{ep_id}.npz` (`runs/`는 `.gitignore`됨). 필드: obs, action, terminated, success_flag, cfg hash, seed
- **BC → PPO 전이 시 필수 사항 (가장 자주 깨지는 곳):**
  1. Network 공유 — BC와 PPO가 **동일한 `ActorCritic` 클래스** 사용
  2. Action logstd 초기값 작게 — `actor_logstd = -1.0` (std ≈ 0.37). 크면 BC weight 곧장 망가짐
  3. Critic warm-up — PPO 첫 10~20 iter는 **actor freeze + critic만** 학습 (garbage advantage가 actor 망치는 것 방지)
  4. Learning rate 낮게 — BC 직후 PPO LR = **1e-4** (평소 3e-4 대비)
  5. Observation normalization — **BC 데이터로 계산한 mean/std를 PPO에서 그대로 사용**. Running mean update X
  6. (선택) reward shaping을 쓰는 경우 초기 dense weight 강하게 → 학습 진행되며 점진 감소. Phase 1은 sparse only라 해당 없음
- **Goal-space curriculum (cube 시작 분포):**
  - 단계: goal 주변 정사각형 한 변 길이 `5 cm → 10 cm → 15 cm → 20 cm`
  - 확장 조건: 100 eval episode에서 success rate ≥ 70%
  - 정체 판정: 단계별 1 M env step 이내 도달 못 하면 정체로 보고 진단
- 두 항목 모두 `env_cfg.curriculum`, `train_cfg.bc`에 단일 소스로 둠

### 8. 가드 (Phase 1에서 작동 필수)
- `eval/video_sampler.py` — N step마다 rollout 영상 자동 저장
- `eval/obs_alignment.py` — env reset obs와 정책 학습 분포 일치 검증
- **success 5조건 개별 logging** — episode마다 5조건 각각의 만족률, lift event count, rolling event count를 별도 metric으로 기록 (성공률만 보다 또 reward hacking을 놓치지 않기 위함)
- env 상수 단일 소스 — sim2real wrapper / reward 어디에도 hardcode 금지
- **Isaac Lab API 변경성 경고** — Isaac Lab API는 버전마다 빠르게 변함. `ActuatorCfg` / `JointPositionActionCfg` / `RewTerm` / `EventTermCfg` 등의 정확한 인자 이름·시그니처를 **기억으로 작성하지 말 것** — 항상 현재 설치된 Isaac Lab 버전 docs 또는 소스 코드를 직접 확인. 의심스러우면 사용자에게 현재 Isaac Lab 버전을 물어볼 것

## Method B (v7) pivot 결정 사항 (2026-05-23 ~)

F15에서 stage 2 PnP 69% 도달 후, joint-delta + PPO/SAC/BC 경로의 reward hacking 위험 + sim2real 신호 일관성 문제를 막기 위해 **EE-delta + env 내부 IK + 순수 BC**로 전환. v1~v7 6회 codex 검토 + 자가 검토 누적. 단일 진실 소스는 `docs/method_b_design.md`. **M0.5 + M1 + M2 완료 (2026-05-23). action_size=4, obs_total=29. 기존 BC v0~v3 / F15 ckpt + runs/demos/stage* 폐기. oracle.py / collect_demos.py / train_*.py는 6-D action 출력이라 일시 비활성 (M3에서 oracle 재작성).**

### 핵심 결정 (위 Phase 1 결정 사항 #2/#4를 덮어씀, M2 이후 활성)
- **Action space**: **4-D** `[Δx, Δy, Δz, g]` top-down 고정 (5-DOF arm의 6-D task over-determined 회피). orientation은 home pose + `wrist_roll hard clamp (q_target[4] = 0.0)`로 강제
- **IK 위치**: env 내부 `env._ik_step()`에서 매 policy step (20 Hz) 호출. oracle은 IK 호출 X
- **IK algorithm**: position-only DLS (3-D task) + null-space bias toward `q_home`. 수식은 `oracle.py:295-340` 이식
- **Jacobian 소스**: **Pinocchio** (`pin.computeJointJacobians` + `pin.getFrameJacobian(LOCAL_WORLD_ALIGNED)`). PhysX X (real path 호환 + standalone test). M0.5에서 PhysX와 임의 q 비교로 regression test 영구화
- **Obs**: `last_action` 6→4, 총 obs **31→29**. 나머지 (`joint_pos`, `joint_vel`, `ee_pose_base`, `cube_xyz_base`, `target_delta_base`, `gripper_state`, `normalized_t`)는 유지
- **Trainer**: BC only 1차. 정체 시 chunked BC 또는 DAgger. RL 결합 보류

### 핵심 정의 (Step A 확정 — 변경 시 docs/method_b_design.md §2 갱신)
- `commanded_ee_delta_actual(t) = FK(q_target(t)) - FK(q_arm(t))` — post-IK-clamp 기대 delta (DLS/clamp 반영 후)
- `EE_jitter_residual(t) = ||EE_pos(t) - (EE_pos(t-1) + commanded_ee_delta_actual(t-1))||` — 명령 빼고 남는 진동만
- `tracking_error(t) = ||EE_pos(t) - FK(q_target(t-1))||` — jitter와 분리 로깅
- `q_dot(t) = (q(t) - q(t-1))/dt`  단위 rad/s, dt=0.05s
- `q_jerk(t) = (q_dot(t) - q_dot(t-1))/dt`  단위 rad/s²

### 다층 IK 발산 가드
- **즉시 AND**: `q_dot_step.abs().max() > 0.3 rad` AND `||x_err|| > 0.05 m` → fallback
- **즉시 OR**: `jitter_residual_p95 > T_jitter` OR `q_jerk_p95 > T_jerk` over 10-step window → fallback. T값은 M0.5에서 calibrate (calibration set ≠ validation set)
- **누적**: `consecutive_ik_fails ≥ 10` (0.5 s) → episode truncate
- **느린 발산**: `||x_err||` rolling 20-step 단조 증가 → warning
- **Fail reason 5종 분류**: `joint_limit / singular (det(JJT)<1e-4) / unreachable (||x_err||>0.1) / large_qdot / jitter`

### Reward-hacking sub-check (success.py 보강, 5조건 안에 박힘)
- Transit phase는 rolling tolerance (visibility 3-step / gripper 5-step / cube_orientation 3-step), **terminal window 마지막 20 step (=1 s)은 strict**
- 추가 검증: cube up-axis vs world up `< 30°` (tipped/edge 차단), release 시 fingertip↔cube `≥ 1 cm` (contact separation)
- IK fail masking: `consecutive ≥ 5` OR `ratio > 5%` (transit) + **terminal window 1 step도 차단**

### 가드 trip 분리 (audit 카테고리)
- **Hard exploit trip**: success=False 차단 → 100 ep 전체 **0건** 요구
- **Audited warning trip**: 평균 ≤ 1/ep, p95 ≤ 2/ep
- **Terminal-adjacent warning**: success=True ep의 마지막 40 step 안 → **0건**
- **Category rate > 1%**: 자동 audit 플래그

### 마일스톤 (요약)
- **M0.5**: 정의 박기 + Jacobian regression test + Negative scenario 12개 합성 + EE 속도 측정 + threshold calibration/validation split (`runs/calibration/ik_thresholds.json`) + home pose 검증 (2 d)
- **M1**: standalone IK 단위 테스트, 4 case (reachable/unreachable/limit-near/singular) (1.5 d)
- **M2**: env.step EE-delta 전환, action_space (4,), obs 31→29, fail-fast ckpt shape mismatch (0.5 d)
- **M3**: oracle EE-delta 출력 + 100 ep 평가 (Hard exploit 0, Warning 평균 ≤ 1/ep, terminal IK fail 0) (0.5 d)
- **M4**: stage 0 1000 demo 재수집, 기존 seed 재사용, 가드 trip 0 확인 (0.5 d auto)
- **M5**: BC 학습. Pass = stage 0/1 각 100 ep success ≥ 70% + 가드 만족. Stretch = 3 seed × 100 ep success ≥ 80% (0.5 d + stretch 1~2 d)
- **M6**: stage 1 demo / 정체 시 chunked BC + chunk delta cap 또는 DAgger

### 폐기 / 보관 / Dormant
- **폐기**: 기존 BC v0~v3, F15 PnP ckpt, `runs/demos/stage*` (action space 호환 X)
- **보관**: oracle IK 수식, Pinocchio FK setup, `success.py` (sub-check 추가본), `video_sampler.py`, `obs_alignment.py`, `demo_buffer.py` 포맷
- **Dormant** (자리 유지, 삭제 X): `training/{ppo,sac,sac_network,her}.py`, `envs/cube_lift/reward_dense.py`

### cfg 위치
신규 dataclass `EEControlCfg`, `SuccessGuardCfg`, `AuditCfg`는 `src/khj_rl/envs/cube_lift/cfg.py`에 정의 + `CubeLiftEnvCfg`에 wire. M0.5/M1 동안 dormant. M2에서 env 코드가 읽기 시작.

### 회귀 자산 (M0.5 산출)
- `tests/test_ik_jacobian_regression.py` — Pinocchio vs PhysX vs finite-diff 3-way 비교, 9 pose + 10 random
- `tests/test_reward_hacking_negative.py` — 12 negative scenario, 모두 success=False로 거부돼야 통과
- `runs/calibration/ik_thresholds.json` — T_jitter, T_jerk 결정 근거 + validation trip rate

## 관련 메모

상위 `/home/j-k14d101/.claude/projects/-home-j-k14d101-jabis-sim-v2/memory/`에 다음이 있음 (이 repo의 Claude는 별도 메모 폴더를 만들 것):

- `khj_rl_project.md` — 이 repo의 목적
- `project_rl_personal.md` — 개인 트랙임을 명시
- `pnp_reward_hacking.md` — 재발 방지 동기
- `oracle_lift_environment_limit.md` — Oracle 12cm 한계

## 외부 참고 링크

- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
- Isaac Sim — Closed-Loop Structures (4-bar 모델링 참고): https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/rig_closed_loop_structures.html
- Isaac Sim — Mimic Joint Rigging: https://docs.isaacsim.omniverse.nvidia.com/4.2.0/advanced_tutorials/tutorial_advanced_rigging_complex_structures.html
- PincOpen 본 레포: https://github.com/pollen-robotics/PincOpen
- CNURobotics PincOpen ROS2 드라이버 (URDF 출처, 4-bar는 simplified): https://github.com/CNURobotics/pinc_open_driver
- SO-ARM101: https://github.com/TheRobotStudio/SO-ARM100
- CleanRL PPO 참고 구현: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
