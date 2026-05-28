# Jabis RL 재시작 — Lessons Learned (Phase 0 + 1-1)

세션: 2026-05-09

## 1. Plan 가정 결함

`oracle_policy.py` 는 driver 가 아니라 **클래스 정의 모듈** (`OraclePolicy` class, 342 lines, `if __name__` 없음).
- User plan 의 명령어 `uv run python oracle_policy.py --task ... --max_episodes ... --visualize` 는 작동 불가
- 실제 driver 는 `collect_demos.py` 인데 demo 저장이 강제 (`--output` required)
- **신규 driver `validate_oracle.py` 작성** (~/jabis_sim/sim2real/oracle/validate_oracle.py, 230 lines)
- collect_demos.py 의 import/argparse/main 패턴 그대로 따름
- demo 저장 제거, success rate + state-at-done 히스토그램 + saturation 측정만

## 2. Headless 강제

`echo $DISPLAY` 비어있음 → jupyter07 은 X11 forwarding 안 되어 있음. visualize 모드 모두 불가.
- collect_demos.py / validate_oracle.py 둘 다 `AppLauncher.add_app_launcher_args(parser)` 가 있어서 `--headless` 자동 지원.
- 다음 세션부터는 plan 단계에서 visualize 옵션 자체를 빼는 게 맞음.

## 3. Oracle 의 Systemic 결함 (Phase 1-1 핵심 발견)

100 episode (16 env 병렬, 38초) 결과:

| 지표 | 값 |
|------|-----|
| Success rate | **0.00%** (0/100) |
| 종료 state | **DESCEND 100/100** (다른 state 진입 0건) |
| z_max | 0.015~0.030 (큐브 거의 안 움직임) |
| Per-joint saturation | 0% 모두 |

1env-1ep trace (debug_first_steps=200) 분석:
- step 118~199 동안 state=DESCEND 고착, ee z=0.038 → 0.036 (81 step 동안 2mm)
- target z=0.017 (큐브 옆면 + descend_dz=0.005), 실제 ee z=0.038 → |dz|=0.021 → `descend_z_dist=0.005` threshold 절대 통과 못 함
- `max_ee_step=0.02` 인데 실제 ee 이동량 step 당 ~0.0001m (200배 작음)

## 4. Actuator 미설정 경고 — mimic joint 비활성 의심

```
[Warning] [isaaclab.assets.articulation.articulation] Not all actuators are configured!
Total number of actuated joints not equal to number of joints available: 6 != 10.
```

10 joint 중 6 개만 actuator 설정. 4개 joint 비활성. 형태로 봐서 그리퍼 jaw 의 distal/proximal mirror (양쪽 2개씩 = 4개) 가 mimic 비활성으로 보임.

→ `~/jabis_sim/usd/` 의 mimic 활성 USD (`*.bak.20260508` 백업본) 가 실제로 환경에 로드됐는지 의심. 다음 세션 1순위 점검.

## 5. 다음 세션 디버깅 우선순위

1. **USD mimic joint** — 환경이 실제로 mimic 활성 USD 를 쓰는지 확인. so_arm101.py / lift_env_cfg.py 에서 USD path 추적.
2. **DifferentialIKController step size** — ee step 당 0.0001m 만 움직이는 원인. max_ee_step=0.02 가 무시되는지, action scaling 문제인지, IK gain 문제인지.
3. **그리퍼 geometry collision** — ee z=0.038 에서 정지하는 이유가 그리퍼 collider 가 큐브에 닿아서인지. collider 시각화 또는 contact force 로그.

## 6. `train.py --bc_init` 미구현

`~/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py` (218 lines, 2026-05-07 마지막 수정).
- `bc_init`, `bc_actor`, `bc_checkpoint`, `warmstart`, `load_state_dict` 모두 grep 0 hits
- **다음 세션에서 BC warmstart 로직 추가 필요** (이번 세션 범위 외)
- BC checkpoint payload 호환성은 확인됨 (smoke_test #5 에서 `state_dict keys match rsl_rl actor`, Linear 36→256→128→64→6)

## 7. 작업 패턴

- 신규 driver 작성 후 codex:review 로 검증 → 1개 functional issue (`args_cli.device` 미전파) 발견 → 수정. 작은 fix 하나가 멀티-GPU 환경에서 충돌 가능성 차단.
- 1ep sanity → 100ep 순서가 시간 절약에 효과적이었음 (1ep 6초로 systemic 결함 즉시 발견, 100ep 38초는 분포 확증용).

---

# 세션 2 — Oracle 0% 원인 진단 결과 (2026-05-09)

## 가설별 PASS/FAIL 판정

### H1 (mimic dead): **PASS** ✓

USD 변환 후 left_distal/right_proximal/right_distal 에 PhysX MimicJoint API 가 cosmetically 적용됐지만 **`ref` 값이 `"rotX"` (axis label)** 으로 잘못됨 — joint path 가 아니라 axis 이름이라 reference 가 무효. 결과적으로 mimic coupling 안 동작.

증거:
- D.1: BAK (변환 전) USD 에는 mimic 없음. CURRENT 에는 3개 finger 에 mimic API 가 추가됐으나 ref="rotX" — self-referential 또는 무효
- A.4: l_prox 와 l_dist 가 정확한 mirror motion (linear ±0.030/step) 은 4-bar 기하의 자연 결과 (passive 4-bar linkage 는 active joint 에 따라 자동 추종). 하지만 좌우 비대칭 (r_prox 시작값 0.155 vs l_prox 0) — mimic 이 right side 를 left 와 동기화하지 못함
- A.3: `gripper` joint 는 mimic 도 actuator 도 없어서 완전 DEAD (total_delta 2.3e-6)

### H2 (action squash): **PASS** ✓

여러 단계의 squash 누적으로 oracle 의 ee target 명령이 실제 ee 이동의 35배 (target/actual ratio=0.028).

증거:
- B.1: target_delta 평균 0.049m → actual_delta 평균 0.001m, ratio 2.8% (≈1/35.7), std 0.96% — 매우 systematic
- B.2: 첫 번째 squash 단계 = `oracle.scale=1.5` vs `arm_action.scale=0.5` → 3× squash. 즉 oracle 이 arm_raw=0.1 으로 명령해도 env 는 default+0.1×0.5=0.05rad 만 변경 명령
- B.2: PD stiffness 50~200 + decimation=2 (env step 0.02s) + effort_limit 1.9 → joint 가 명령을 부분만 추적
- 추가 ~12× squash (mimic dynamics + PD 미달) 로 누적 35×

### H3 (actuator mismatch): **PASS** ✓

ArticulationCfg actuator 정의가 6 joint 만 커버. 4 joint (`gripper`, `right_proximal`, `left_distal`, `right_distal`) passive.

증거:
- C.3: so_arm101.py 의 actuator dict 에 "arm" (5 joint) + "gripper" (left_proximal 1 joint) = 6 actuator. USD 의 10 joint 중 4개 빠짐
- A.3: passive joint 4개 중 `gripper` 는 완전 DEAD (자유 운동도 없음). 나머지 3개는 자유 운동 (mimic 가짜라 4-bar 기하만 따라감)
- 경고 로그: "Total number of actuated joints not equal to number of joints available: 6 != 10"

## H1 / H2 / H3 의 상호작용

이 3 가설은 독립이 아니라 같은 USD 변환 사고의 다른 측면:
- 변환이 mimic 을 deactivate → H1
- ArticulationCfg 가 변환 전 가정 (mimic 으로 자동 동기화) 으로 작성됨 → 변환 후 actuator 만으로는 4 joint 부족 → H3
- 명령 squash 는 별도 cfg 미스매치 (`oracle.scale=1.5` vs `env.scale=0.5`) + mimic 깨진 finger dynamics 흡수 → H2

## 다음 세션 (수리) 입력 요약 — 이번 세션은 진단까지

(수정 제안 작성 금지 규칙 준수 — 이건 사실 기록만)

- 객관적 사실로 확정된 것들:
  - USD diff 결과 (D.1)
  - Actuator cfg 의 joint_names_expr (C.3)
  - `oracle.scale=1.5`, `env.scale=0.5` 미스매치 (B.2)
  - 6 actuator vs 10 joint 미스매치 (`Not all actuators are configured!` 경고)

## 진단 도구

새 진단 패턴 — 다음 세션부터 재사용 가능:
- `tasks/diagnose_joints.py` (Isaac driver, CSV writer)
- `tasks/diagnose_analyze.py` (CSV analyzer, no Isaac)
- `tasks/diagnose_usd.py` (pxr.Usd 기반 USD diff, AppLauncher 필요)

USD diff 의 함정:
- pxr.Usd.Stage.Open 이 `.bak.20260508` 같은 비표준 확장자를 못 인식. 표준 `.usd` 만 받음.
- USD 가 reference/payload 구조면 root 파일이 1KB 정도, 실제 데이터는 `configuration/*.usd` 같은 sub-USD 에. diff 는 변경된 sub-USD 에서 직접 비교.

## Codex Review caveats (2026-05-09 마무리 검토)

PASS 판정의 약점 명시 — 다음 세션 입력으로 활용:

1. **H1 PASS 보강 가능**: PhysX MimicJoint 는 `referenceJoint` (joint path) + `referenceJointAxis` + `gearing` 3요소 필요. ref="rotX" 는 axis label 이라 joint path 누락 → coupling 무효 판정 타당. 단 "cosmetic" 단정 강화하려면 **PhysX 런타임 로그** (mimic API 가 무효로 무시됐다는 경고가 나오는지) 확인 필요.

2. **left mirror motion (l_prox/l_dist ±0.030/step) 은 4-bar 기하 자연 결과**: left_distal mimic API 가 ref 무효라 실제로는 4-bar 물리 구속이 동기화 만든 것. mimic 작동 증거 아님 — H1 보조 증거 맞음.

3. **H2 가 가장 약함**: 35× squash 분해 (`3× cfg + ~12× 기타`) 의 12× 출처 분리 안 됨. 동등한 후보들:
   - effort_limit_sim=1.9 포화
   - arm_action.scale 의미 혼동 (joint rad 단위 vs 정규화)
   - decimation=2 효과
   - IK residual
   - mimic dynamics 흡수
   - PD stiffness 부족
   현재 데이터로 분리 불가.

4. **다음 세션 추가 진단 권고** (수정 제안 아니라 데이터 수집 권고):
   - `pre_action → joint_target → applied_torque → joint_pos` step 별 로깅 → 35× 가 어디서 누락되는지 분리 가능
   - PhysX 런타임 mimic 경고 grep → H1 cosmetic 단정 보강
   - actuator list 직접 출력 (`robot.actuators`) → H3 보강

---

# 세션 3 — H1+H3 fix 결과 + 세션 2 정정 (2026-05-09)

## 1. 세션 2 의 H2 분석 정정 (중요)

세션 2 lessons.md 의 "H2 PASS — 3× cfg + ~12× 기타 = 35×" 분해는 **잘못된 cfg 파일을 읽은 결과**.

`joint_pos_env_cfg.py` 에는 **두 개의 robot 가족 cfg** 가 있음:
- `SoArm100LiftCubeEnvCfg` (line 32-94): scale=0.5, gripper=["gripper"]
- `SoArm101LiftCubeEnvCfg` (line 109-178): **scale=1.5, gripper=["left_proximal"]**

Task `Isaac-SO-ARM101-Lift-Cube-Play-v0` 는 `SoArm101LiftCubeEnvCfg_PLAY` (101 production 상속) 를 사용. 즉 scale=1.5 가 oracle.scale=1.5 와 이미 매치.

→ **35× squash 의 cfg scale 분해는 무효**. 35× 의 진짜 원인은 다른 메커니즘 (PD/decimation/contact/IK).

H2 PASS 자체는 유효 (35× 이 정량적으로 측정됨). 단 분해/원인 attribution 이 잘못됨.

## 2. Plan A 무익 — IsaacLab UrdfConverter 버그

`config.yaml` 의 `convert_mimic_joints_to_normal_joints`:
- `false`: mimic 무시 → 일반 joint, mimic API 없음 (May 6 BAK USD 결과)
- `true`: PhysX MimicJoint API 추가 → 그러나 ref="rotX" (axis label) 만 채움 (May 8 CURRENT USD 결과)

→ 두 옵션 모두 정상 mimic 결과 안 됨. IsaacLab UrdfConverter 의 mimic 변환 로직에 ref 채우기 버그가 있는 것으로 강력히 추정.

→ **USD 재변환 (Plan A) 은 H1 fix 에 무익**. cfg 만으로 fix 가능.

## 3. URDF mimic 정의 (정확한 ground truth)

URDF `~/jabis_sim/urdf/so101/so101_pincopen_gripper.urdf`:
- `left_distal: <mimic joint="left_proximal" multiplier="-1"/>` → l_dist = -l_prox
- `right_proximal: <mimic joint="left_proximal" multiplier="-1"/>` → r_prox = -l_prox
- `right_distal: <mimic joint="right_proximal" multiplier="+1"/>` → r_dist = r_prox = -l_prox
- `gripper`: master 없음 (그리퍼 회전 base, mimic 없음)
- joint limit: 모두 ±0.77 rad

→ close 명령 시:
- l_prox = -0.6 → l_dist = +0.6, r_prox = +0.6, r_dist = +0.6

세션 2 A.4 데이터 (l_prox -0.030 → l_dist +0.030 등) 가 URDF mimic 정의와 일치 (단 right side 는 mimic broken 으로 자유 운동).

## 4. cfg 변경 적용했으나 거동 변화 0

so_arm101.py 의 gripper actuator 에 right_proximal 추가, joint_pos_env_cfg.py 의 binary action 도 좌우 두 finger 명령. 적용 결과:
- actuator 경고 6 != 10 → 7 != 10 (cfg 변경 효과 확인)
- 1ep joint trace **거의 비트 단위 동일** (세션 2 vs 세션 3 비교)
- 100ep success rate 0% → 0% (변화 없음)

## 5. 0% 가 그대로인 이유 가설

**이번 세션 데이터로 확정 불가**. 가설들:
- PD stiffness=60 + damping=20 이 4-bar dynamics 를 못 이김 (right_proximal 의 자유 운동 trajectory 가 우연히 actuator-driven 결과와 비슷)
- USD 의 깨진 mimic API (ref="rotX") 가 right_proximal 을 어떤 식으로 간섭
- DESCEND 고착의 진짜 원인이 그리퍼 / 4-bar 가 아니라 **arm 의 ee 이동 자체** (target 4.9cm 의도하지만 actual 1.2mm — 35× squash). 즉 그리퍼를 fix 해도 ee 가 큐브 옆면에 못 닿으면 큐브를 잡을 수 없음.

세 번째 가설이 가장 그럴듯. 이 경우 **H2 (action squash) 가 실패의 근본 원인이고, H1/H3 는 부수 결함**.

## 6. 다음 세션 plan 권고 (codex 정정 반영)

**우선순위 1: cfg 변경이 정말 무효인지 확인** (codex caveat: H2 진단보다 우선)
- `arm_action_term.cfg.joint_names` / `grip_action_term.cfg.joint_names` 출력 → runtime 에 실제 들어간 joint 목록 확인
- `term.action_dim` 출력 → 1-dim 인지 2-dim 인지 (binary action 의 input shape)
- `robot.actuators.keys()` + `robot.actuators["gripper"].joint_names` 출력 → actuator 가 정말 right_proximal 까지 driven 하는지
- 만약 right_proximal 이 actuator 에 없으면 (cache/install 문제) → 새로운 fix 필요

**우선순위 2: H2 force-tracing (cfg 가 effective 하다고 확인된 후)**
1. `tasks/diagnose_h2_force.py` 실행 → pre_action / clipped / applied_target / applied_torque / jpos / jvel / ee 분리
2. **codex 권고 분해 순서**: action term output → joint target → torque saturation → EE 실제 이동
3. 35× squash 의 핵심 의심: **torque saturation > decimation > IK residual** (codex 권고)
4. effort_limit_sim=1.9 의 포화 빈도가 가장 우선 의심
5. contact force / 큐브 collision sanity (init z 멀리 두고 ee 자유 운동 확인)

이 순서로 진행 후 H2 fix 결정.

---

# 세션 4 — H4/H5/H6 PASS/FAIL + 35× 진앙 식별 (2026-05-09)

## 1. H4 FAIL (cfg 미반영 가설 기각)

세션 3 cfg 변경이 runtime 에 100% 반영됨. introspect 결과:
- `env.cfg = SoArm101LiftCubeEnvCfg_PLAY` (parsed from 우리가 수정한 파일)
- `gripper` actuator: `joint_names=['left_proximal', 'right_proximal']`, `joint_indices=[6, 7]`
- gripper_action term: `cfg.joint_names`/`cfg.close_command_expr`/`_joint_ids`/`processed_actions.shape=(1,2)` 모두 의도대로

→ pycache stale, import path mismatch, inheritance override 모두 FAIL.

## 2. H5 PASS (fix 자체 부족)

5 finger joint 중 **2 개만 actuator + action term 에 들어감**. 3 개 (`left_distal`, `right_distal`, `gripper`) passive.

이유: 세션 3 에서 codex 권고로 "4-bar 과구속 위험" 회피하려고 distal 제외했음. 그러나 데이터로 보니:
- 4-bar 자연 동기화는 **mimic 깨진 USD** 에서는 일부만 발생 (left side 만 일관, right side 불완전)
- `gripper` joint 는 mimic 도 actuator 도 없어 완전 dead (total_delta 1.16e-9 rad)

→ distal 2개 또는 gripper joint 도 actuator 추가가 필요한 것으로 보임. 단 codex 의 과구속 우려도 여전히 유효 — 작은 stiffness 로 보수적 시도 필요.

## 3. H6 PASS (effort_limit 포화 — 35× squash 의 진앙)

35× squash 의 stage 별 분해 (APPROACH 49 step):

| Stage | Ratio | 의미 |
|---|---|---|
| (1) clip | 1.000 | 영향 0 (action 이미 [-1,1] 내) |
| (2) scale → joint_target | 0.451 | use_default_offset 측정 정의 차이 (squash 아님) |
| **(3) PD tracking** | **0.063** | **PD 가 target 의 6.3% 만 추적 — 주범** |
| (4) ee total | 0.028 | 종합 (= 0.451 × 0.063, 정확 매치) |

PD 추적 미달의 원인 = effort_limit 포화:
- `shoulder_pan/shoulder_lift/elbow_flex` 가 effort_limit=1.9 N⋅m 에서 **40% 이상 saturated**
- `right_proximal` 은 effort_limit=2.5 에서 **94% saturated** (close=+0.6 명령을 강제 driving)
- `wrist_flex/wrist_roll` 은 saturation 없음 (이미 작은 명령)

사용자 메모 "STS3215 effort_limit=1.9" 와 정확히 일치. 실모터 한계 model 인데 sim 의 stiffness=200 이 무거운 link 들어올리려면 더 큰 토크 필요.

## 4. ee 가 step 당 1mm 만 이동하는 이유 (정량 confirmed)

- target_ee_delta_3d 평균 4.9cm (oracle 의도)
- joint_target_delta 평균 = 4.9 × 0.451 ≈ 2.2cm
- jpos_delta 평균 = 2.2 × 0.063 ≈ 1.4mm
- actual_ee_delta_3d 평균 = 1.2mm (관측값)

→ **stage (3) 의 PD tracking 미달이 1mm 의 직접 원인**. effort_limit=1.9 이 토크 부족으로 PD 가 명령을 못 따라감.

## 5. 다음 세션 (실수정) 입력 자료

이번 세션은 진단까지. 다음 세션에서 결정해야 할 사항 (cfg 수정 또는 새 strategy):

**H5 fix 옵션:**
- (a) `gripper` actuator 의 joint_names_expr 에 `left_distal`, `right_distal`, `gripper` 추가 — codex 우려: 4-bar 과구속 위험. 작은 stiffness (10~20) 로 시작.
- (b) Python action wrapper 로 left_proximal 명령을 5 finger 모두에 broadcast (manual mimic emulation).

**H6 mitigation 옵션:**
- (a) `effort_limit_sim` 1.9 → 3.0 또는 5.0 (sim2real gap 발생 — STS3215 실한계 1.9 위반)
- (b) `effort_limit_sim` 유지 + arm stiffness 200→100 (slower target tracking, but no saturation)
- (c) sim.dt 0.01 → 0.005 (더 자주 step, 매 step 작은 movement, fidelity 향상하지만 학습 throughput 절반)
- (d) 그대로 두고 학습이 실모터 한계 안에서만 cube lift 하도록 강제 (현실적, 하지만 oracle 도 이 한계로 lift 못 하므로 strategy 재설계 필요)

**Demo 수집 게이트: NOT YET** — Oracle 자체가 0% 라 demos 없음.

## 6. 작업 패턴

- "거의 동일" 같은 정성 표현은 정량화 필요. CSV diff (max/mean abs diff + nonzero count) 가 결정적.
- runtime cfg introspection (env.cfg.mro, inspect.getfile, action_manager terms 의 internal buffers) 이 cfg 적용 여부 판정의 황금 표준. python -c 로 외부에서 import 하면 omni.log 등 의존성 fail — AppLauncher 거쳐야.
- diagnose_h2_force.py 의 MimicTee 가 fileno() 누락으로 Isaac Sim 의 faulthandler.enable() 와 충돌. file-like wrapper 만들 때 fileno/isatty/__getattr__ 모두 delegate 필수.
- PhysX 가 mimic API 의 잘못된 ref="rotX" 에 대해 경고 출력 안 함 (silent fail). 이는 IsaacSim 4.5 의 mimic 처리 한계.

## 7. Codex 마무리 caveats (2026-05-09)

**H6 PASS 단정 보류** (codex 권고):
- stage 3 ratio 0.063 의 std 가 매우 작음 (0.001). saturation 만이 원인이면 step 별 std 더 커야 함 (saturation 시 작고 non-saturation 시 큼).
- 매우 일정한 0.063 은 effort_limit 단독보다 **PD gain/damping/env_dt/decimation 의 선형 저역통과 합산** 가능성도 큼.
- 정확한 표현: "**포화가 관여 (한 component)**" 까지가 안전. 단독 진앙 단정은 추가 데이터 필요 (예: stiffness 변경 시 ratio 변화 실험).

**Stage 2 의 0.451 해석 보류**: target buffer 1-step 지연 / offset 적용 위치의 가능성도 반증 대상. use_default_offset 측정 정의로 단정 확정 보류.

**right_proximal 94% saturation 은 부호 충돌 가능성** ⭐:
- close=+0.6 명령인데 4-bar 가 -0.6 으로 끌어당기는 상황이면 actuator 가 full force 로 싸우는 상태일 수 있음
- 데이터: r_prox 시작 0 → 0.155 까지 자유 운동 (4-bar 가 + 방향) → force_close 후 +0.6 target 명령 (같은 방향) → saturation 94%
- 같은 방향이면 saturation 불필요할 텐데 94% 는 "PD 가 6.3% 만 추적" 결과 — 의도 close 가 +0.6 인데 실제 0.244 까지만 가서 PD 가 계속 +0.6 으로 끌어당기느라 saturation
- **다음 세션 H5 수정 전 부호 검증 필요** (특히 distal joint 의 부호도)

## 8. Demo 수집 게이트 (정량 기준)

다음 세션 fix 후 다음 둘 다 충족해야 demo 수집 진입:
- (a) Oracle 100ep success rate **≥ 60%**
- (b) APPROACH state 의 actual_delta / target_delta ratio **≥ 0.5** (현재 0.028 → 18 배 개선 필요)

둘 중 하나라도 미달 시 추가 fix 또는 strategy 재설계.

## 7. 작업 패턴

- Codex 의 보수적 권고 (proximal 2 개만 driven) 가 안전했음. distal/gripper 까지 actuator 추가 안 한 게 과구속/진동 위험 회피.
- cfg 파일 두 가족 (SoArm100 vs SoArm101) 혼동은 grep 으로 line 만 보고 결정 시 실수 가능. **클래스 hierarchy 까지 확인**해야 정확 (task ID → cfg class entry_point → __post_init__ chain).
- "fix 적용 + 데이터 동일" 결과를 솔직히 기록. fix 가 무효임이 다음 세션의 의사결정에 critical.
