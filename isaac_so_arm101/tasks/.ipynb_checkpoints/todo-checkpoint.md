# Jabis RL 재시작 — Phase 0 & 1-1 (Sanity Check + Oracle 검증)

세션 시작: 2026-05-09
목표: IL 인프라 상태 점검 + Oracle policy 자체의 lift 성공률 검증
범위 외: demo 수집, BC 학습, PPO 시작, `--bc_init` 코드 추가

> 모든 학습/inference 명령은 `CUDA_VISIBLE_DEVICES=1` prefix 필수 (jupyter07 공유 GPU 0 침범 금지)
> 환경: `conda activate isaaclab`

## Phase 0 — 인프라 상태 점검

### 0.1 tmux 세션 정리
- [ ] `tmux ls` 결과 보고 (현재 활성: `jabis-day4`)
- [ ] `tmux capture-pane -t jabis-day4 -p | tail -50` 으로 학습 여부 + 마지막 iteration 캡처
- [ ] 학습 세션이면 사용자에게 kill 승인 요청, 비학습이면 그대로 보존

### 0.2 IL smoke test
- [ ] `cd ~/jabis_sim/sim2real/oracle && CUDA_VISIBLE_DEVICES=1 uv run python smoke_test.py`
- [ ] 7/7 통과 확인. 실패 시 어느 테스트인지 보고하고 중단

## Phase 1-1 — Oracle 단독 검증

### 1-1.0 환경 + headless 사전 점검 (NEW)
- [ ] `python ~/isaac_so_arm101/scripts/list_envs.py | grep -i lift` 로 정확한 task ID 확인
- [ ] `echo "DISPLAY=$DISPLAY"` 로 GUI 가능 여부 판별
  - DISPLAY 비어있음 → 1-1.1 visualize 단계는 `--headless` fallback 필수
  - DISPLAY 설정됨 → visualize 그대로 진행

### 1-1.1 1env-1ep mimic/IK sanity (NEW, codex 권고)
100ep 돌리기 전 짧은 검증으로 USD mimic joint + DifferentialIKController 정상 동작 확인.

- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python oracle_policy.py --task <ID> --num_envs 1 --max_episodes 1` (visualize 가능하면 추가)
- [ ] 그리퍼 mimic joint 가 양쪽 다 움직이는지 (한쪽만 닫히면 USD 백업본 미반영)
- [ ] IK 가 NaN/발산 없이 수렴하는지 로그 확인
- [ ] 통과 시에만 1-1.2 로 진행

### 1-1.2 Oracle visualize 10 episode + 5-state 관측
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python oracle_policy.py --task <ID> --num_envs 1 --max_episodes 10 --visualize`
  - headless 면 `--visualize` 제거 + `--headless` 추가
- [ ] 5-state 가 로그에 모두 나타나는지:
  - APPROACH (큐브 상공 5~8cm)
  - DESCEND (큐브 옆면 높이, 위누름 X)
  - CLOSE (그리퍼 닫힘)
  - LIFT (z축 5~10cm 상승)
  - MOVE_TO_GOAL (goal 이동)
- [ ] 각 episode 의 종료 state 기록

### 1-1.3 Oracle 100 episode rollout
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python oracle_policy.py --task <ID> --num_envs 16 --max_episodes 100 --no-visualize 2>&1 | tee ~/isaac_so_arm101/tasks/oracle_100ep_$(date +%Y%m%d_%H%M).log`
- [ ] **성공 기준**: `cube_z_max >= 0.10` (oracle_policy.py 의 출력 또는 자체 측정)
- [ ] 산출물: 위 tee log 파일 경로
- [ ] 기록: 총 success/fail 카운트, success rate, 실패 episode 의 종료 state 분포

### 1-1.4 진단 보고 (성공률 분기)

| Success rate | 진단 | 다음 단계 |
|--------------|------|-----------|
| ≥ 60% | 정상 | Phase 1-2 (demo 수집) 진행 가능 |
| 30~60% | state machine 튜닝 필요 | 어느 state 에서 가장 많이 실패하는지 보고 |
| < 30% | oracle 또는 환경 문제 | DifferentialIKController, USD mimic, observation 정합성 점검 |

- [ ] 위 표 기준 진단 + 다음 단계 권고 작성

## Phase 0.3 (이동) — `train.py --bc_init` 구현 상태 진단
> codex 권고로 Oracle 검증 후로 이동. 1-1 결과와 무관하게 보고만 하고 코드 수정은 금지.

- [ ] `~/isaac_so_arm101/scripts/rsl_rl/train.py` 위치 확인
- [ ] `--bc_init` 인자 파싱 grep
- [ ] BC actor weights `load_state_dict` 로직 grep
- [ ] 결과 보고: 구현됨 (핵심 5~10줄 인용) / 미구현 (빠진 부분 명시)

## Validation Checklist (세션 종료 전)

- [ ] tmux 학습 세션 처리 완료 (kill 또는 보존 결정)
- [ ] smoke_test.py 7/7 통과
- [ ] 환경 ID + DISPLAY 확인 완료
- [ ] 1env-1ep sanity 통과
- [ ] Oracle visualize 5-state 관찰 또는 막힘 지점 보고
- [ ] Oracle 100ep 성공률 숫자 기록 + 로그 파일 경로
- [ ] 실패 패턴 state 별 분포 기록
- [ ] train.py --bc_init 구현 상태 명확히 보고
- [ ] tasks/todo.md 결과 요약, tasks/lessons.md 발견사항 기록

## Cleanup Whitelist (절대 수정/삭제 금지)

- `~/jabis_sim/usd/*.usd`, `*.bak.20260508`
- `~/jabis_sim/sim2real/policy_inference.py`, `action_decoder.py`
- `~/jabis_sim/sim2real/oracle/{oracle_policy,collect_demos,train_bc}.py`
- v1~v7 학습 로그/체크포인트 (`logs/rsl_rl/v*`)
- `~/S14P31D101/embedded/training/lerobot_gpu/outputs/`

## Codex Review (2026-05-09)
- 0.3 → 1-1 뒤로 이동 (선행조건 아님)
- 1-1.0 환경/DISPLAY 사전 점검 + 1-1.1 1env-1ep sanity 추가
- 모든 명령에 `CUDA_VISIBLE_DEVICES=1` prefix
- 성공 기준 + 로그 산출물 경로 명시

## Session Result Summary (2026-05-09 종료)

### 완료
- ✅ Phase 0.1 — tmux 세션 정리 (`jabis-day4` kill, 학습 세션 아니었음. 마지막 활동 5/6, GLFW 실패로 idle)
- ✅ Phase 0.2 — IL smoke_test 7/7 PASS (BC train loss 0.046→0.003, BC actor state_dict ↔ rsl_rl actor 호환 확인)
- ✅ Phase 1-1.0 — DISPLAY 비어있음 (headless 강제), 환경 ID `Isaac-SO-ARM101-Lift-Cube-Play-v0` 정상
- ✅ **Plan 결함 우회**: `oracle_policy.py` 가 driver 아님 → 신규 driver `~/jabis_sim/sim2real/oracle/validate_oracle.py` (230 lines, codex review 거침) 작성
- ✅ Phase 1-1.1 — 1env-1ep sanity: success=0/1, DESCEND 고착, ee z=0.038 정지
- ✅ Phase 1-1.3 — 100ep rollout: **success_rate = 0.00% (0/100), 100/100 DESCEND 고착**
  - log: `~/isaac_so_arm101/tasks/oracle_100ep_20260509_1532.log`
- ✅ Phase 1-1.4 — 진단 분기 결정: **<30% (oracle/환경 문제)**
- ✅ Phase 0.3 — `train.py --bc_init` **미구현** 확정 (grep 0 hits)
- ✅ tasks/lessons.md 작성

### 건너뜀 (Plan 가정 결함으로 불가)
- ❌ Phase 1-1.2 (visualize 10ep) — DISPLAY 없음 + oracle_policy.py 가 driver 아니어서 visualize 인자 자체 부재. validate_oracle.py 의 `--debug_first_steps 200` trace 로 대체.

### 다음 세션 우선순위 (Phase 2 디버깅)
1. **USD mimic joint 확인** — actuator 6 != 10 경고. so_arm101.py / lift_env_cfg.py 의 USD path 추적, `~/jabis_sim/usd/*.bak.20260508` 가 활성 USD 인지 검증
2. **DifferentialIKController step size 진단** — ee 가 step 당 0.0001m 만 이동 (max_ee_step=0.02 의 1/200). action 스케일링 / IK gain 점검
3. **그리퍼 collision** — ee z=0.038 에서 정지하는 원인이 그리퍼 collider 와 큐브 contact 인지 확인
4. **위 3건 해결 후** → `train.py --bc_init` 추가 → demo 수집 → BC 학습 → PPO warmstart 순서

### 작업 환경 정리
- Cleanup Whitelist 항목 모두 보존 ✓ (USD, oracle_policy.py, collect_demos.py, train_bc.py, v1~v7 로그 모두 unchanged)
- 신규 파일 1개: `~/jabis_sim/sim2real/oracle/validate_oracle.py` (whitelist 외부)
- 신규 로그 1개: `~/isaac_so_arm101/tasks/oracle_100ep_20260509_1532.log`

---

# 세션 2 — Oracle 0% 원인 진단 (2026-05-09)

> 진단 전용. 모든 USD/cfg/oracle 파일 read-only. 신규 파일은 `tasks/diagnose_*.py` 만 허용. 가설 H1/H2/H3 을 데이터로 PASS/FAIL 만 판정, 수정 제안 작성 금지.

## 가설

- **H1**: PincOpen 4-bar mimic 변환 후 coupling 사라져서 left_proximal 만 닫히고 left_distal/right_proximal/right_distal/gripper 는 0 고정
- **H2**: IK action scale 또는 clip_actions 문제로 ee 가 step 당 0.0001m 만 이동
- **H3**: ArticulationCfg actuator 6개 정의와 USD 10 joint 사이 미스매치, 4개 finger joint 가 passive/0 고정

H1 과 H3 는 같은 원인의 다른 측면일 수 있음.

## 환경 자료

- USD 현재: `~/jabis_sim/usd/so101_pincopen.usd`
- USD 백업: `~/jabis_sim/usd/so101_pincopen.usd.bak.20260508`
- ArticulationCfg: `~/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`
- 검증 driver baseline (read-only): `~/jabis_sim/sim2real/oracle/validate_oracle.py`
- env cfg: `~/isaac_so_arm101/src/isaac_so_arm101/tasks/lift/joint_pos_env_cfg.py`, `lift_env_cfg.py`

## 진단 스크립트 (신규, tasks/ 에 작성)

1. **`tasks/diagnose_joints.py`** — validate_oracle.py 패턴 기반 Isaac driver. 수정 금지된 파일들 import 만 함. step 별 10개 joint pos/vel + command action + target/current ee + cube pos + state 를 CSV 저장. 1ep, max 200 step.
2. **`tasks/diagnose_actions.py`** — CSV 분석 (no Isaac). APPROACH state 의 raw_action / target_delta / actual_delta / ratio 통계.
3. **`tasks/diagnose_usd.py`** — pxr.Usd 로 두 USD 의 joint 목록 + PhysicsMimicJointAPI 추출. so_arm101.py grep 으로 actuator 정의 파싱. C.3 + D.1 표 출력.

## A — Joint 실시간 모니터링

- [x] A.1 `diagnose_joints.py` 작성
- [x] A.2 1env-1ep, max 200 step 실행 → `tasks/diagnose_joints_<ts>.csv`
- [x] A.3 Joint Activation Table 출력 (10 joint 별 min_pos / max_pos / total_delta / first_movement_step)
- [x] A.4 CLOSE state 진입 직후 5 step finger 표 (left_proximal / left_distal / right_proximal / right_distal / gripper / command_grip)

## B — IK action → ee delta

- [x] B.1 APPROACH state 의 |raw_action| / |target_delta| / |actual_delta| / ratio mean/min/max/std 표
- [x] B.2 cfg 값 표: action_scale, clip_actions (env+runner), decimation, dt, IK step_size/damping/scaling

## C — Actuator vs Joint 미스매치

- [x] C.1 ArticulationCfg / actuator 정의 grep + 인용
- [x] C.2 USD revolute/prismatic joint name 10개 추출
- [x] C.3 미스매치 표: usd_joint_name | in_actuator_cfg | actuator_type | stiffness | damping

## D — USD mimic diff

- [x] D.1 두 USD 의 mimic 메타데이터 비교 표: joint_name | before (bak) | after (current)

## 종합

- [x] 6개 표 todo.md 에 부착
- [x] lessons.md 에 H1/H2/H3 PASS/FAIL 판정 + 증거 명시 (수정 제안 금지)

## 진행 흐름

1. plan codex review
2. diagnose_joints.py 작성 → codex review (Isaac 의존 driver 라 가장 위험)
3. A 실행
4. diagnose_actions.py 작성 + B 실행
5. diagnose_usd.py 작성 + C/D 실행
6. 결과 부착 + 가설 판정 + lessons.md 업데이트

## Session 2 Result Summary

### 산출물
- `~/isaac_so_arm101/tasks/diagnose_joints.py` (driver, 197 lines, codex review 거침)
- `~/isaac_so_arm101/tasks/diagnose_analyze.py` (CSV 분석, 130 lines)
- `~/isaac_so_arm101/tasks/diagnose_usd.py` (USD diff, 140 lines)
- `~/isaac_so_arm101/tasks/diagnose_joints_20260509_1546.csv` (200 rows joint trace)

### A.3 Joint Activation Table (200 step, 1 env, force_close from step 100)

```
joint_name         | min_pos    | max_pos    | total_delta  | first_move | DEAD?
shoulder_pan       | 0.000000   | 0.232517   | 0.237682     | 1          |
shoulder_lift      | 0.000000   | 0.716706   | 0.836574     | 1          |
elbow_flex         | 0.000000   | 0.306920   | 0.422423     | 1          |
wrist_flex         | 1.120878   | 1.623964   | 0.557051     | 1          |
wrist_roll         | 0.000000   | 0.002732   | 0.004807     | 33         |
gripper            | 0.000000   | 0.000000   | 2.300e-06    | (none)     | DEAD
left_proximal      | -0.598355  | 0.000000   | 0.598373     | 101        |
right_proximal     | 0.000000   | 0.561170   | 0.602369     | 1          |
left_distal        | 0.000000   | 0.578012   | 0.580391     | 15         |
right_distal       | -0.613725  | 0.004949   | 0.825610     | 1          |
```

핵심: `gripper` 완전히 DEAD (total_delta 2.3e-6, 사실상 0). `left_proximal` 은 force_close 직후 (step 101) 부터 처음 움직임 — 그 전엔 OPEN 명령(+1) 이라 안 닫힘. 다른 3 finger 는 step 1 또는 15 부터 이미 자유 운동.

### A.4 CLOSE state finger response (force_close=1, cmd_grip=-1)

```
step | l_prox     | l_dist     | r_prox     | r_dist     | gripper    | cmd_grip   | state    | force
100  | -1.225e-05 | 6.389e-04  | 0.155273   | 0.002303   | 0.000000   | -1.000000  | DESCEND  | 1
101  | -0.030011  | 0.030629   | 0.162999   | -0.006270  | 0.000000   | -1.000000  | DESCEND  | 1
102  | -0.060011  | 0.060617   | 0.188162   | -0.033141  | 0.000000   | -1.000000  | DESCEND  | 1
103  | -0.090010  | 0.090606   | 0.216923   | -0.063718  | 0.000000   | -1.000000  | DESCEND  | 1
104  | -0.119597  | 0.120181   | 0.245180   | -0.093756  | 0.000000   | -1.000000  | DESCEND  | 1
```

핵심:
- **`gripper` 는 close 명령 5 step 동안 0 고정 (DEAD).**
- l_prox 와 l_dist 가 정확히 mirror (선형 ±0.030/step, 4-bar 기하 자연 결과) — **그러나 left_proximal 만 actuator 로 driven, l_dist 는 자유 운동 (mimic 없음)**.
- r_prox 가 시작값 0.155 (l_prox 0 과 비대칭) — 우측 finger 가 좌측을 따라가지 않음. mimic broken 의 직접 증거.

### B.1 Arm Action / EE Delta Statistics (APPROACH state, 49 rows)

```
metric                   | mean         | min          | max          | std
|action_0| (arm)         | 0.097923     | 0.041411     | 0.165519     | 0.036939
|action_1| (arm)         | 0.085478     | 0.017560     | 0.149635     | 0.039740
|action_2| (arm)         | 0.107709     | 0.053277     | 0.149540     | 0.027885
|action_3| (arm)         | 0.035757     | 0.023026     | 0.042756     | 0.005177
|action_4| (arm)         | 4.897e-04    | 0.000000     | 9.385e-04    | 2.758e-04
|target_delta_3d|        | 0.049424     | 0.021026     | 0.081076     | 0.017167
|actual_delta_3d|        | 0.001241     | 0.001155     | 0.001331     | 5.107e-05
ratio (actual/target)    | 0.027878     | 0.015328     | 0.052126     | 0.009619
```

핵심: **target ee delta 평균 4.9cm, actual 1.2mm, ratio 2.8% (≈1/35).** ratio std 0.0096 매우 tight → systematic squash. wrist_roll (action_4) 명령은 거의 0 (정상).

### B.2 cfg 값 표

```
Setting                              | Value         | Source
sim.dt                               | 0.01          | lift_env_cfg.py:246  (100 Hz)
decimation                           | 2             | lift_env_cfg.py:242
env_dt (derived)                     | 0.02          | (50 Hz)
arm_action.scale (env)               | 0.5           | joint_pos_env_cfg.py:45
oracle.scale                         | 1.5           | oracle_policy.py / collect_demos.py default
scale ratio (env / oracle)           | 0.333         | env이 oracle 의 1/3 만 명령으로 해석
clip_actions (rsl_rl)                | 1.0           | rsl_rl_ppo_cfg.py:27
arm stiffness (shoulder_pan~wrist_roll) | 200/170/120/80/50 | so_arm101.py:44-50
arm damping (shoulder_pan~wrist_roll)| 80/65/45/30/20 | so_arm101.py:51-57
arm effort_limit_sim                 | 1.9           | so_arm101.py:42
arm velocity_limit_sim               | 1.5           | so_arm101.py:43
gripper actuator stiffness           | 60.0          | so_arm101.py:63
gripper actuator damping             | 20.0          | so_arm101.py:64
gripper joint_names_expr             | ["left_proximal"] | so_arm101.py:60 ⚠️
oracle max_ee_step                   | 0.02          | oracle CLI default
```

⚠️ 핵심: `arm_action.scale=0.5` vs `oracle.scale=1.5` → **3× 명령 squash**. 거기에 PD 추적 미달 + mimic dynamics 흡수가 추가로 ~12× → 합 ~35× 가 actual/target ratio 2.8% 와 일치.

### C.3 Actuator vs Joint 미스매치

```
usd_joint_name    | in_actuator_cfg | actuator_name | actuator_type    | stiffness          | damping
shoulder_pan      | YES             | "arm"         | ImplicitActuator | 200.0              | 80.0
shoulder_lift     | YES             | "arm"         | ImplicitActuator | 170.0              | 65.0
elbow_flex        | YES             | "arm"         | ImplicitActuator | 120.0              | 45.0
wrist_flex        | YES             | "arm"         | ImplicitActuator | 80.0               | 30.0
wrist_roll        | YES             | "arm"         | ImplicitActuator | 50.0               | 20.0
gripper           | NO              | -             | (passive)        | -                  | -
left_proximal     | YES             | "gripper"     | ImplicitActuator | 60.0               | 20.0
right_proximal    | NO              | -             | (passive)        | -                  | -
left_distal       | NO              | -             | (passive)        | -                  | -
right_distal      | NO              | -             | (passive)        | -                  | -
```

`gripper` actuator 의 `joint_names_expr=["left_proximal"]` 만 정의 → 6 actuator. 4 joint passive (gripper, right_proximal, left_distal, right_distal). 이게 "Total number of actuated joints not equal to number of joints available: 6 != 10" 경고의 정확한 원인.

### D.1 USD Mimic Diff (configuration/so101_pincopen_physics.usd)

```
joint           | BAK (변환 전, May 6)            | CURRENT (변환 후, May 8)
shoulder_pan    | RevoluteJoint                  | RevoluteJoint
shoulder_lift   | RevoluteJoint                  | RevoluteJoint
elbow_flex      | RevoluteJoint                  | RevoluteJoint
wrist_flex      | RevoluteJoint                  | RevoluteJoint
wrist_roll      | RevoluteJoint                  | RevoluteJoint
gripper         | RevoluteJoint (mimic 없음)      | RevoluteJoint (mimic 없음)
left_proximal   | RevoluteJoint (mimic 없음)      | RevoluteJoint (mimic 없음)
left_distal     | RevoluteJoint (no mimic)        | RevoluteJoint + MIMIC-API ref=rotX gearing=+1.0 offset=0
right_proximal  | RevoluteJoint (no mimic)        | RevoluteJoint + MIMIC-API ref=rotX gearing=+1.0 offset=0
right_distal    | RevoluteJoint (no mimic)        | RevoluteJoint + MIMIC-API ref=rotX gearing=-1.0 offset=0
                +PhysX mimic params: dampingRatio=0.005, naturalFrequency=25.0 Hz
```

**핵심 발견:** mimic API 의 `ref` 값이 `"rotX"` (axis label) 임. 일반적 referenceJoint 는 "/path/to/joint" 형태인데, axis label 만 적힘 → reference 가 무효 또는 self-referential. cosmetic mimic, 실제 coupling 없음. `gripper` 와 `left_proximal` 에는 mimic API 자체가 없음.

---

# 세션 3 — H1+H3 fix + 재진단 (2026-05-09)

## 환경 자료 사전 점검 결과

- URDF: `~/jabis_sim/urdf/so101/so101_pincopen_gripper.urdf` (변환 source)
- 변환 스크립트: `~/jabis_sim/day5/scripts/reconvert_pincopen.py` (line 35 hardcoded `convert_mimic_joints_to_normal_joints=True`)
- config.yaml: 옵션 `convert_mimic_joints_to_normal_joints` 가 `true` (현재) vs `false` (BAK)
- **역설**: `true` 가 mimic API 추가하지만 IsaacLab 버그로 ref 가 axis label 로 깨짐. `false` 가 mimic 무시한 일반 joint.
- → Plan A 로 정상 mimic 확보 불가 (둘 다 broken 결과)

## 추가 발견 (Plan 외)

`joint_pos_env_cfg.py:48-53` — Binary gripper action 이 `joint_names=["gripper"]` 인데 `gripper` joint 는 그리퍼 회전축이지 finger 가 아님. 진짜 그리퍼는 left/right_proximal 이 닫는 4-bar. 즉 binary action 자체가 잘못된 joint 를 명령 중. 이건 H1+H3 와 별도의 큰 결함.

## 작업 1 — H1 fix 접근법 ✅

### 1.1 USD 재변환 평가 — Plan A 결론 = **불가능**
- 변환은 단순 (한 줄): `cd ~/jabis_sim/day5/scripts && CUDA_VISIBLE_DEVICES=1 uv run python reconvert_pincopen.py --headless`
- 그러나 IsaacLab 의 mimic→PhysX 변환 자체가 ref 를 axis 로만 채우는 버그 존재 (D.1 증거)
- `true` 또는 `false` 둘 다 정상 mimic 결과 안 됨
- → **Plan B (cfg 확장 wrapper) 채택 권고**

### 1.2 Plan B (선택) — cfg 만 수정으로 fix 가능

가장 단순한 형태:
1. **so_arm101.py**: gripper actuator 의 `joint_names_expr = ["left_proximal"]` → 5 finger 모두 포함 (`["left_proximal", "right_proximal", "left_distal", "right_distal", "gripper"]` 또는 정규식)
2. **joint_pos_env_cfg.py:48-53**: Binary gripper action 의 `joint_names`/`open/close_command_expr` 를 실제 닫는 joint 들로 확장

이렇게 하면 binary action 1 채널이 5 finger 모두에게 같은 target 을 broadcast → 4-bar 기하 + actuator 대칭 → 좌우 동기화 회복.

USD 재변환 불필요. `tasks/fix_*.py` 신규 스크립트도 불필요.

## 작업 2 — H3 fix ✅

작업 1.2 안에서 actuator 추가가 자동으로 H3 fix 도 해결. 별도 작업 불필요.

## 작업 3~6 결과

- ✅ 작업 3: 환경 로딩 sanity — 경고 6 != 10 → **7 != 10** (right_proximal 추가 효과 확인)
- ✅ 작업 4: 1ep joint trace 재진단 — 세션 2 와 거의 비트단위 동일 (cfg 변경이 1ep 거동에 0 영향)
- ✅ 작업 5: 100ep 재측정 — **0% (변화 없음)**, 100/100 DESCEND. 진단 분기 < 10%
- ✅ 작업 6: `tasks/diagnose_h2_force.py` 작성 (210 lines, py_compile OK, codex hasattr fallback 추가)

## Session 3 Result Summary

### 적용된 cfg 변경

**`src/isaac_so_arm101/robots/trs_so101/so_arm101.py`** (line 60):
```python
"gripper": ImplicitActuatorCfg(
-   joint_names_expr=["left_proximal"],
+   joint_names_expr=["left_proximal", "right_proximal"],
    effort_limit_sim=2.5, ...
)
```

**`src/isaac_so_arm101/tasks/lift/joint_pos_env_cfg.py`** (line 127-132, `SoArm101LiftCubeEnvCfg`):
```python
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
-   joint_names=["left_proximal"],
-   open_command_expr={"left_proximal": 0.0},
-   close_command_expr={"left_proximal": -0.6},
+   joint_names=["left_proximal", "right_proximal"],
+   open_command_expr={"left_proximal": 0.0, "right_proximal": 0.0},
+   close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6},
)
```

### 결과 요약

| 지표 | 세션 1 | 세션 3 |
|------|--------|--------|
| Actuator 경고 | 6 != 10 | **7 != 10** (right_proximal 추가됨, distal/gripper 는 의도적으로 passive) |
| 100ep success rate | 0% | **0%** (변화 없음) |
| 100/100 종료 state | DESCEND | DESCEND |
| 1ep joint trace | (세션 2 측정) | **거의 비트 단위 동일** (cfg 변경이 거동에 영향 0) |

### 진단 분기: < 10% → "H1+H3 fix 불완전 또는 새 문제"

cfg 변경은 파일에 정상 저장 (git diff 확인) 되었고 actuator 경고도 7 로 변화. 그러나 100ep/1ep 거동이 거의 동일.

### 세션 2 분석의 정정 (중요)

세션 2 의 H2 분석에서 "scale 0.5 vs 1.5 = 3× squash" 라고 한 부분은 **잘못된 cfg (SoArm100, line 42-53) 를 읽은 결과**. 실제 우리 task `Isaac-SO-ARM101-Lift-Cube-Play-v0` 는 `SoArm101LiftCubeEnvCfg` (line 109-178) 를 사용하며 **scale=1.5 로 oracle 과 이미 일치**.

→ 35× squash 의 진짜 원인은 cfg scale 미스매치가 아닌, PD stiffness/decimation/contact force/IK step 등. 다음 세션 H2 진단 (`diagnose_h2_force.py`) 으로 분리.

### 신규 산출물

- `~/isaac_so_arm101/tasks/diagnose_joints_session3.csv` (200 row, fix 후 1ep trace)
- `~/isaac_so_arm101/tasks/oracle_100ep_session3_20260509_1613.log`
- `~/isaac_so_arm101/tasks/diagnose_h2_force.py` (실행 X, syntax check OK)

### Cleanup Whitelist 보존 ✓
- USD 파일 unchanged
- oracle_policy.py / collect_demos.py / train_bc.py / validate_oracle.py unchanged
- 세션 2 진단 스크립트 / CSV unchanged
- v1~v7 학습 로그 unchanged
- 수정된 파일: so_arm101.py / joint_pos_env_cfg.py 만 (사용자 승인)

---

# 세션 4 — cfg 반영 검증 + H2 진단 (2026-05-09)

## 가설

- **H4**: 세션 3 cfg 변경이 runtime 에 부분/전혀 미반영
  - H4a: cfg inheritance/override
  - H4b: import 경로 mismatch
  - H4c: bytecode cache stale (`__pycache__`)
- **H5**: cfg 반영됐으나 fix 부족 (예: gripper 명령이 5 finger 전부에 안 감)
- **H6**: 35× squash 의 진짜 원인 = effort_limit 포화 (1.9)

## Pre-flight 사실

- 두 cfg 파일 mtime: 2026-05-09 16:09-16:10 (세션 3 수정 시점) ✓
- editable install 경로: `/home/j-k14d101/isaac_so_arm101/src/isaac_so_arm101` ✓
- `__pycache__` 디렉토리 다수: `src/isaac_so_arm101/{robots,tasks}/{trs_so101,lift}/__pycache__/` 등

## 작업 1 — 세션 2 vs 세션 3 trace diff 정량화

- [ ] `tasks/diff_session3.py` 작성: 두 CSV 의 컬럼별 max/mean abs diff + nonzero step 수
- [ ] 핵심 판정:
  - right_proximal pos 에 0 이 아닌 차이 — 있으면 actuator 추가 반영
  - left_distal/right_distal/gripper pos 차이 — 없으면 broadcast 안 됨
  - command_action[5] 차이 — 없으면 cfg 안 read

## 작업 2 — Runtime cfg introspection (H4)

- [ ] `tasks/introspect_runtime.py` 작성, AppLauncher 거쳐 실행
- [ ] env.cfg mro / source file
- [ ] robot.actuators (key, joint_names, effort/stiffness)
- [ ] action_manager terms (joint_names, scale, command_expr, action_dim, _joint_ids)
- [ ] runtime 항목별 ✓/✗ 판정

## 작업 2.3 — H4 PASS (cfg 미반영) 시

- [ ] 같은 source file → __pycache__ 삭제 후 재실행
- [ ] 다른 source → 진짜 위치 보고, 사용자 확인 후 수정

## 작업 2.4 — H4 FAIL (cfg 반영됐으나 결과 변화 없음) 시

- [ ] 5 finger joint 모두 action term 에 들어있나 확인 → H5 evidence

## 작업 3 — Oracle 재측정 (cfg fix 확정 후만)

- [ ] 100ep + 1ep trace
- [ ] 세션 3 vs 4 trace diff (이번엔 의미 있게 다른지)

## 작업 4 — H2/H6 진단 (cfg fix 확정 후만)

- [ ] 4.1 `diagnose_h2_force.py` 실행 (1ep, 200 step)
- [ ] 4.2 `tasks/analyze_h2_decomp.py` 작성, 35× decomposition 표 (clip / scale / IK / tracking / decimation)
- [ ] 4.3 H6 torque saturation: per joint mean/max |torque|, saturation % vs effort_limit=1.9
- [ ] 4.4 PhysX mimic 경고 grep (stderr "mimic" log)

## 작업 5 — 결론 보고

- [ ] H4a/H4b/H4c/H5/H6 PASS/FAIL 표
- [ ] 35× squash 주범 stage 명시
- [ ] demo 수집 게이트 판정

## Session 4 Result Summary

### 산출물
- `tasks/diff_session3.py` (CSV diff 분석, 60 lines)
- `tasks/introspect_runtime.py` (runtime cfg/actuator/term introspect, 130 lines)
- `tasks/analyze_h2_decomp.py` (35× decomposition + H6 saturation, 160 lines)
- `tasks/diagnose_h2_force_session4.csv` (200 rows, force tracing)
- `tasks/h2_stderr_session4.log` (0 bytes — PhysX mimic 경고 없음, ref="rotX" 가 silently 무시되는 결정적 단서)

### 작업 1 — 세션 2 vs 세션 3 trace diff (정량)

```
column                       | max_abs_diff | nonzero_steps
jpos_right_proximal          | 1.001e-3 rad | 89/200    *  ← actuator 추가 효과
jpos_right_distal            | 9.36e-4 rad  | 93/200    *
jpos_left_proximal           | 2.4e-7       | 0
jpos_gripper                 | 1.16e-9      | 0  (완전 dead 그대로)
jpos_left_distal             | 3.76e-6      | 35
jvel_*proximal/*distal       | 0.020 rad/s  | 96~97
action_5 (gripper command)   | 0.000        | 0  (oracle 동일 명령)
current_ee_*                 | < 1e-6 m     | 0  (ee 동일)
```

→ cfg 효과 일부 있음 (right_proximal 1mm), 하지만 ee 거동 사실상 변화 없음.

### 작업 2 — Runtime introspection (H4)

```
[1] env.cfg = SoArm101LiftCubeEnvCfg_PLAY
    source: /home/j-k14d101/isaac_so_arm101/src/isaac_so_arm101/tasks/lift/joint_pos_env_cfg.py ✓
[2] Robot actuators:
    'arm':     joint_names=[shoulder_pan~wrist_roll], effort=1.9, stiffness 50~200
    'gripper': joint_names=['left_proximal', 'right_proximal'] ✓ (세션 3 의도)
               effort_limit=2.5, stiffness=60, damping=20
[3] gripper_action term:
    cfg.joint_names = ['left_proximal', 'right_proximal'] ✓
    cfg.close_command_expr = {'left_proximal': -0.6, 'right_proximal': 0.6} ✓
    _joint_ids = [6, 7] ✓
    processed_actions shape = (1, 2) ✓ broadcast 됨
    _close_command shape = (2,)
[4] env.action_space = Box(-1, 1, (1, 6))
```

→ **모든 항목 ✓. cfg 100% 정상 반영.**

### 작업 4 — H2 35× decomposition (APPROACH 49 step)

```
stage                                              | mean ratio | 해석
(1) |clipped| / |pre_action|                       | 1.000      | clip 영향 0
(2) |joint_target_delta| / (|clipped|×1.5)         | 0.451      | use_default_offset 측정 정의
(3) |jpos_delta| / |joint_target_delta| (tracking) | 0.063      | ⭐ PD 추적 미달, 주범
(4) |actual_ee_delta| / |target_ee_delta| (ee)     | 0.028      | 종합 (= 0.451 × 0.063 정확)
```

### 작업 4 — H6 torque saturation

```
joint           | effort_limit | mean|t| | max|t| | sat% (>0.95×limit)
shoulder_pan    | 1.9          | 0.826   | 1.900  | 40.0%
shoulder_lift   | 1.9          | 0.904   | 1.900  | 41.0%
elbow_flex      | 1.9          | 0.789   | 1.900  | 41.0%
wrist_flex      | 1.9          | 0.263   | 1.401  | 0%
wrist_roll      | 1.9          | 0.373   | 1.032  | 0%
left_proximal   | 2.5          | 2.150   | 2.500  | 54%
right_proximal  | 2.5          | 2.416   | 2.500  | 94%
gripper / left_distal / right_distal | (passive) | 0 | 0 | -
```

### 가설 PASS/FAIL 표

```
Hypothesis                          | Result | Evidence
H4a inheritance override            | FAIL   | env.cfg.mro 정상 (101 → 101 PLAY)
H4b import path mismatch            | FAIL   | inspect.getfile 우리 파일 정확히 가리킴
H4c bytecode cache stale            | FAIL   | runtime cfg 100% 반영됨 (cache 안 stale)
H5 fix 자체 부족 (joint 누락)        | PASS   | action term _joint_ids=[6,7] (left/right proximal만), distal 2개+gripper passive
H6 effort_limit saturation          | PASS   | shoulder pan/lift/elbow 40%+, right_proximal 94%
```

### 세션 결론

- 세션 3 cfg fix 는 정상 적용 (H4 FAIL)
- 거동 변화 미미한 이유: **(a) H5 — distal 2개 + gripper joint 가 actuator 없어서 4-bar 자연 동기화 안 됨, (b) H6 — arm 토크가 이미 saturated 상태**
- **35× squash 의 정확한 진앙: PD tracking residual (stage 3, 0.063 ratio)**, 그 원인은 H6 effort_limit 1.9 포화
- ee 가 step 당 ~1mm 만 이동하는 이유 = arm 모터가 명령된 토크를 다 못 냄

### Demo 수집 게이트 판정: **NOT YET**
- Oracle 100ep 0%, DESCEND 고착 그대로
- 다음 세션에서 H5 fix (distal/gripper actuator 추가) + H6 mitigation (effort_limit 또는 stiffness 조정 또는 decimation 조정) 필요
- 사용자 결정 사항: STS3215 실모터 한계 1.9 를 sim 에서도 강제할지 (sim2real gap 우선) vs 더 크게 두고 학습부터 가능하게 할지 (학습 우선)

---

# 세션 5 — 부호 검증 → H5 fix → H6 fix → 게이트 (2026-05-09)

## 단계별 게이트

각 단계 게이트 통과 후만 다음 진행. 한 번에 한 후보만 적용.

## 작업 0 — 한글 인코딩 점검

- [ ] todo.md / lessons.md / *.log 의 file 출력
- [ ] 깨진 파일 발견 시 사용자 보고

## 작업 1 — 부호 검증 probe (Codex 권고)

- [ ] `tasks/probe_gripper_sign.py` 작성: arm 정지, gripper 채널 -1 (50step) → +1 (50step)
- [ ] 5 finger 별 direction (close/open/static) 표
- [ ] 시나리오 판정 (모두 일치 / 부호 충돌 / passive 확정)

## 작업 2 — H5 fix (1.3 결과 따라)

- [ ] 시나리오 A/B/C 중 결정 + 사용자 승인
- [ ] git diff 사전 보고 후 cfg 수정
- [ ] introspect 재확인 (모든 5 finger _joint_ids 포함)
- [ ] probe 재실행, close phase 5 finger 모두 close 방향 게이트

## 작업 3 — H6 mitigation (1 후보)

- [ ] 현재 effort/stiffness/damping/dt/decimation 확인
- [ ] 후보 A (effort_limit↑) / B (damping↓) / C (stiffness↑) / D (decimation↓) 중 사용자와 결정
- [ ] git diff 후 적용

## 작업 4 — Trace 재측정 + decomp 비교 + ratio 게이트

- [ ] diagnose_joints / h2_force 재실행
- [ ] H2 decomp 세션 4 vs 5 비교 표
- [ ] **게이트: PD tracking ratio ≥ 0.5**

## 작업 5 — Oracle 100ep 게이트

- [ ] ratio 게이트 통과 후만 실행
- [ ] **Demo 진입 게이트: success rate ≥ 60% AND ratio ≥ 0.5 (둘 다)**
- [ ] 미통과 시 사용자 결정 (추가 fix vs RL 종료)

## 작업 6 — 결과 정리

- [ ] todo.md / lessons.md 업데이트

## Session 5 Result Summary

### 적용된 cfg 변경 (`so_arm101.py`)
```diff
         "arm": ImplicitActuatorCfg(
             joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
-            effort_limit_sim=1.9,
+            effort_limit_sim=2.5,    # H6 후보 A: sim2real gap +25%
         ...
         "gripper": ... (그대로)
+        "finger_distal": ImplicitActuatorCfg(   # H5 시나리오 C
+            joint_names_expr=["left_distal", "right_distal"],
+            effort_limit_sim=2.5, velocity_limit_sim=1.5,
+            stiffness=5.0, damping=2.0,        # 극보수 stiffness (codex 보다 더 작게)
+        ),
```

### 작업 1 — 부호 검증 probe (CRITICAL)
- close 명령 (-1) 50 step → tip distance **0.0909 → 0.0502** (40mm 가까워짐, 정상 grasp)
- open 명령 (+1) 50 step → tip distance **0.0502 → 0.0832** (33mm 멀어짐, 정상 release)
- 부호 패턴 (l_prox-, l_dist+, r_prox+, r_dist-) = 4-bar 좌우 대칭 grasp 의 자연 결과
- gripper joint 만 static (wrist 다음 회전축, grasp 무관 — 의도대로)
- → **codex 부호 충돌 의심 기각**, 4-bar grasp 자체는 정상 동작

### 작업 2 — H5 fix 적용 후 introspect
```
actuators keys: ['arm', 'gripper', 'finger_distal']
  arm:           joint_indices=[0,1,2,3,4],  effort=2.5
  gripper:       joint_indices=[6,7],        effort=2.5
  finger_distal: joint_indices=[8,9],        effort=2.5  stiffness=5  damping=2
경고: 9 != 10 (남은 1 = gripper joint, 의도대로)
```
post-H5 probe: 4 finger 모두 close 방향 ✓, tip distance 동일 (40mm grasp), distal mean torque 0 → 2.45 (effort 2.5 의 98%, 4-bar 와 PD 가 싸우는 상태)

### 작업 4 — H2 Decomposition 비교 (★ 주요 결과)

| Stage | 세션 4 | 세션 5 | 변화 |
|---|---|---|---|
| (1) clip | 1.000 | 1.000 | 동일 |
| (2) scale → joint_target | 0.451 | 0.451 | 동일 |
| **(3) PD tracking** | **0.063** | **0.0629** | **거의 변화 없음** |
| (4) ee total | 0.028 | 0.0279 | 거의 동일 |

H6 가 mean torque 0.83→1.07 (25% 증가) 시켰지만 **PD tracking ratio 변화 없음** — codex 권고 정확히 적중 ("effort_limit 단독 아닌 PD/decimation 합산 의심").

### 작업 4 — joint trace diff (세션 3 vs 5)
- jpos_right_proximal max 1mm → 16mm (16배 변화), distal 6~14mm
- arm joint max 0.0014 rad
- ee position max 0.2mm (여전히 미미)

### 작업 5 — 100ep 게이트 결과

```
[DONE] episodes=100  steps=2800  elapsed=38s
[DONE] success=0 fail=100 success_rate=0.00%
[STATE HISTOGRAM]
  DESCEND       success=  0  fail=100  (전부 DESCEND 고착)
```

### 게이트 판정

| 게이트 | 임계 | 측정 | 판정 |
|---|---|---|---|
| PD tracking ratio | ≥ 0.5 | 0.0629 | **FAIL** |
| PD tracking ratio (보조) | ≥ 0.3 | 0.0629 | **FAIL** |
| 100ep success rate | ≥ 60% | 0.00% | **FAIL** |

→ **Demo 수집 진입 불가**. Critical decision point 도달.

### 다음 단계 — 사용자 결정 사항

1. **추가 H6 후보 적용 (1세션 더)**: 후보 D (decimation 2→1) 또는 B (damping↓) 또는 동시
   - 위험: D 는 throughput 절반, B 는 oscillation. 효과 보장 없음
   - 시간: D-day 정보 없어 일정 영향 평가 불가
2. **RL 트랙 종료 검토**: 진단 4세션 + fix 1세션 시도 후 ratio 8× 개선 필요한 격차. 메인 IL 트랙 합류 검토
   - Claude 단독 결정 금지 (plan 명시)
   - 5세션의 진단 데이터는 RL 트랙 종료 보고서 입력으로 활용 가능

### Cleanup Whitelist 보존 ✓
- USD/oracle/v1~v7 unchanged
- so_arm101.py 만 수정 (사용자 승인)

---

# 세션 6 — Oracle 우회 (옵션 4) + Sweep (옵션 5) 병행 (2026-05-09)

## 전략 변경 근거

5세션 누적 결과: 환경 cfg fix (effort_limit, distal actuator 등) 효과 0.
- ratio 0.0629 (8× 개선 필요), 100ep 0% 변화 없음.
- Oracle 측에서 우회 + 백그라운드 sweep 으로 환경 원인 데이터 확보 병행.
- RL 종료 결정은 김현준 명시 승인만 (양팔 시연 한 팔 RL 트랙).

## Oracle interface 분석 (사전 조사)

`OraclePolicy.__init__` 인자 = `params` dict 에 저장:
- `reach_dist=0.02` (APPROACH→DESCEND 전이 threshold)
- `descend_z_dist=0.005` (DESCEND→CLOSE 전이)
- `max_ee_step=0.02` (한 step ee delta cap)
- `reach_above_dz=0.10`, `descend_dz=0.005`, `lift_dz=0.10`, `success_z=0.10`

→ threshold 완화 (옵션 4.C) 는 **wrapper 없이 validate_oracle.py CLI 인자만** 으로 가능.

세션 5 1ep trace 데이터:
- DESCEND state ee z=0.038, target z=0.017, |dz|=0.021 → threshold 0.005 무한 timeout
- 200 step 에 ee z 2mm 변화 (step 당 0.01mm) → max_ee_step 늘려도 PD ratio cap 0.063 효과 X
- Episode time limit 400 step → 100/100 DESCEND timeout

## 옵션 4 후보 비교

| 후보 | 메커니즘 | 예상 효과 | wrapper 필요 |
|---|---|---|---|
| A. Dwell time | action 캐싱 N step 반복 | oracle 이 이미 같은 state 에서 같은 명령 보내므로 효과 작을 듯 | 필요 |
| B. Sub-step decomposition | max_ee_step ↓, 더 자주 IK | PD ratio cap 이 dominant 라 효과 X | 부분 (cli 만으로 가능) |
| **C. Threshold 완화** | reach_above_dz/descend_z_dist ↑ | DESCEND timeout 회피 + ee 가 cube 옆 닿기 전 CLOSE 가능 | 불필요 |

→ **C 가 가장 직접적이고 wrapper 불필요**. 그러나 plan 명시 1차 조합 (dwell=10, relax=1.0) 따를지 user 결정 필요.

## 작업

### 옵션 4 (메인 트랙)
- [ ] 4.1 1차 후보 + 인자 user 결정
- [ ] 4.2 wrapper (필요시) `tasks/oracle_bypass_v1.py`
- [ ] 4.3 driver 작성 또는 validate_oracle.py 인자만 사용
- [ ] 4.4 1차/2차/3차 한 번에 1 조합 측정. 매 조합 후 결과 보고

### 옵션 5 (백그라운드)
- [ ] 5.1 `tasks/sweep_pd_decimation.py` — PD stiffness×decimation×damping grid 18 조합, 1ep ratio 측정. so_arm101.py 백업+임시+원복
- [ ] 5.2 tmux khj_sweep 백그라운드 실행, 약 9 분 예상
- [ ] 5.3 ratio 정렬 표

### 결과 통합
- [ ] 옵션 4 best success rate
- [ ] 옵션 5 best ratio + config
- [ ] 게이트 기반 권고 (Stage B / 옵션 1 / 옵션 2)

## Session 6 Result Summary

### 옵션 4.C 결과 — state machine 진행 풀음, lift 한계

**1차 (4.C v1)**: `--reach_dist 0.04 --descend_z_dist 0.025 --reach_above_dz 0.05 --descend_dz 0.020`
```
[DONE] success=0 fail=100 success_rate=0.00% (sz=0.10)
[STATE HISTOGRAM] LIFT success=0 fail=100  (이전 세션: DESCEND fail=100)
```
세션 1/3/5 와 비교:
- 이전: 100/100 DESCEND 고착
- **4.C v1: 100/100 LIFT 도달** — APPROACH→DESCEND→CLOSE→LIFT state machine 정상 진행

**1차 success_z 완화 측정 (sz=0.05)**:
```
success=2 fail=98 (2%)
```
last z_max sample: `[0.030, 0.026, 0.033, 0.051, 0.016, 0.012, 0.025, 0.018, 0.012, 0.012]`
- 일부 (~30%) cube 1~5cm 들림 (grasp 부분 성공 + lift 부족)
- 다수 (~60%) cube 시작 z 그대로 (grasp 자체 실패)

**2차 (4.C v2 극단 완화)**: `--descend_z_dist 0.05 --reach_above_dz 0.02`
```
success=0 fail=100 (sz=0.10)
last z_max 분포: [0.030, 0.026, 0.033, 0.051, ...] = 1차와 완전 동일
```
→ **threshold 추가 완화 효과 0** (1차에서 이미 한계). 진짜 bottleneck = LIFT 단계 PD ratio cap.

### 옵션 4 결론
- threshold 완화는 DESCEND 고착만 해소
- 35× squash (LIFT 단계) 가 dominant — 이 부분은 환경 fix 영역
- success rate 0% 변화 없음

### 옵션 5 sweep — 실행 실패 (cfg 손실로 중단)

`tasks/sweep_pd_decimation.py` 작성, tmux 백그라운드 실행 시도.

**1차 시도 실패 — uv build error**:
- subprocess 가 `uv run python` 사용. ~/jabis_sim/sim2real/oracle 에서 isaac_so_arm101 dependency 빌드 실패 (flatdict, 세션 1 동일 issue).

**2차 시도 실패 — patch 함수 정규식 버그**:
- 초기 patch 가 init_state.joint_pos dict 의 `"shoulder_pan": 0.0` 등도 매치해서 0.0 → 0.000000 등 wrong target 변경.
- 해결 1: cfg 원복 → `git checkout -- src/...` 실행
- **부작용**: git checkout 이 세션 3-5 누적 cfg 변경 (proximal actuator 추가, effort 2.5, finger_distal stiffness=5) 도 모두 원복. so_arm101.py 가 원본 URDF (`urdf/so_arm101.urdf`) 사용 상태로 돌아감.
- 해결 2: patch 함수 정규식 수정 (`stiffness={...}` / `damping={...}` dict 만 타겟). 단위 테스트 통과 확인.

**3차 시도 실패 — robot articulation 6 joint**:
- cfg 원복 후 so_arm101.py 가 원본 URDF 사용 → robot 6 joint (`shoulder_*, elbow_flex, wrist_*, gripper`)
- env cfg 의 `gripper_action.joint_names=["left_proximal", "right_proximal"]` 가 robot 에 없는 joint 매치 시도 → ValueError
- sweep 종료. 

**현재 cfg 상태**: git HEAD = 세션 3 fix 적용 전 원본. URDF 도 원본 (PincOpen 4-bar 없음). robot 6 joint.

### 옵션 5 sweep 데이터: **수집 실패**
다음 세션에서 cfg 재구성 후 재시도 필요.

### Cleanup Whitelist 보존
- USD/oracle/v1~v7 unchanged
- so_arm101.py + lift_env_cfg.py: 세션 3-5 누적 변경이 git checkout 으로 원복됨 (sweep buggy patch 정리 시). 4.C 측정은 세션 5 cfg 상태에서 진행 (effort 2.5, finger_distal). sweep 결과는 git HEAD baseline.
- 다음 세션 결정에 따라 cfg 재적용 또는 새 fix.

---

# 세션 7 — Baseline 재측정 + lift_dz sweep (2026-05-09)

## 사고 방지 원칙
- git checkout 사용 금지
- cfg 수정 전 `cp <file> <file>.session7_pre` 백업 필수
- 한 번에 한 변경
- worktree-격리 sweep 인프라 다음 세션

## 사전 점검 결과
- **branch: `khj-rl-track`** (사용자 새 브랜치)
- **git status 비어있음** — 세션 3-6 누적 cfg 변경 모두 commit 된 상태. lost 아님.
- bypass wrapper 신규 만들 필요 없음 (validate_oracle.py CLI 가 `--lift_dz` 등 모두 노출)
- **lift_dz sweep wrapper 없이 가능** ← 큰 simplification

## 작업 1 — introspect 검증 (cfg 의도대로인지)

- [ ] `tasks/introspect_runtime.py` 실행, log 저장
- [ ] 체크리스트:
  - actuators 키에 `arm`, `gripper`, `finger_distal` (세션 5 신규)
  - num_joints 10
  - gripper actuator joint_indices [6, 7]
  - finger_distal joint_indices [8, 9], stiffness=5
  - arm effort_limit_sim=2.5
  - gripper_action term cfg.joint_names ['left_proximal', 'right_proximal']
  - gripper_action close_command_expr / open_command_expr 정의
- [ ] X 항목 있으면 사용자 보고 (cfg 일부 lost?)

## 작업 2 — 4.C v1 baseline 재측정

- [ ] `validate_oracle.py --reach_dist 0.04 --descend_z_dist 0.025 --reach_above_dz 0.05 --descend_dz 0.020` 100ep
- [ ] 세션 6 vs 7 비교 표 (success rate sz=0.10/0.05, LIFT 도달률, z_max 분포)
- [ ] 게이트: ±10% 내 재현 → 작업 3 진행 / 큰 차이 → 사용자 보고

## 작업 3 — lift_dz sweep (5 조합)

baseline 재현 통과 후만:
- [ ] lift_dz [0.05, 0.08, 0.10, 0.15, 0.20] 5 조합 100ep 각각 (4.C threshold 동일 유지)
- [ ] 결과 표 + best success rate
- [ ] 게이트 분기:
  - ≥60% → demo 진입
  - 30~60% → 사용자 결정
  - 10~30% → dwell sweep 추가
  - <10% → 사용자 결정 (옵션 1 vs 옵션 2)

## 작업 4 — lessons.md §7 + 권고

## Session 7 Result Summary

### 사전 검증 (introspect)

**부분 cfg lost 발견 + 재구성**:
- `git status` 비어있어 cfg 안정 보였으나 introspect 로 확인 시 ValueError ("left_proximal not found")
- `so_arm101.py` 가 `UrdfFileCfg` (원본 URDF, 6-joint) 사용. env cfg 만 세션 3 변경 잔존 → mismatch
- → **lessons.md §6.1 cfg 재구성 (cp .session7_pre 백업 + Write 적용)** — 사용자 승인
- 재 introspect: actuators ['arm', 'gripper', 'finger_distal'] ✓, num_joints 10, action_dim 1 / processed_actions (1,2) ✓, 9 != 10 (gripper joint 의도적 passive)

### 4.C v1 Baseline 재측정 — **0% → 15%** (큰 진전)

```
[DONE] success=15 fail=85 success_rate=15.00%
[STATE HISTOGRAM] APPROACH succ=1 fail=1 / LIFT succ=14 fail=84
last z_max sample: [0.078, 0.012, 0.012, 0.108, 0.012, 0.012, 0.128, 0.050, 0.080, 0.062]
range 0.012~0.128 (이전 세션 6 max 0.051 의 2.5×)
ep length 500 (이전 400)
```

**원인 식별** — `git log -p` 로 commit `15f3bb3` 에서 발견:
```
src/isaac_so_arm101/tasks/lift/lift_env_cfg.py:
-    self.decimation = 2
+    self.decimation = 1
```

→ 사용자가 codex 의 세션 5 권고 "후보 D (decimation ↓)" 를 적용 commit. **이게 35× squash 의 진짜 mitigation 의 핵심**. 세션 6 의 측정은 (이 commit 전 또는 sweep 잔존 영향) inconsistent 상태였을 가능성. 세션 7 이 진짜 baseline.

### lift_dz Sweep — 효과 X

| lift_dz | success rate | LIFT s/f | APPROACH s/f |
|---|---|---|---|
| 0.05 | **15%** | 14/84 | 1/1 |
| 0.08 | 15% | 14/84 | 1/1 |
| 0.10 | 15% | 14/84 | 1/1 |
| 0.15 | 15% | 14/84 | 1/1 |
| 0.20 | 15% | 14/84 | 1/1 |

비트 단위 동일 trajectory. **lift_dz 효과 0**. oracle 의 `max_ee_step=0.02` cap 또는 grasp 형성 한계가 dominant. lift_dz 명령 magnitude 무관.

### 🎯 Demo 수집 — **102 demos 성공**

```
[DONE] succ=102 finished=608 rate=16.78% steps=19000 elapsed=171s
saved → tasks/demos_session7_target100.pt (8.6 MB)
```

| 항목 | 값 |
|---|---|
| n demos | 102 |
| obs_dim / act_dim | 36 / 6 |
| z_max min / mean / max | 0.101 / 0.131 / 0.189 |
| ep length | 모두 500 step |
| final_state | 101 LIFT + 1 APPROACH |
| source | oracle_v1 |

→ **Stage B 진입 자료 확보**. BC actor 호환 (smoke_test #5 의 architecture matches).

### 다음 세션 권고

1. **BC 학습** (`train_bc.py`) — 102 demos 로 actor MLP (36→256→128→64→6) 학습
2. **`train.py --bc_init` 추가** (이번 세션 절대 금지였으나 Stage B 진입 시 필요)
3. **PPO warmstart** with BC init weights

진정한 게이트 통과 — RL 트랙 종료 검토 → **Demo 진입 진행으로 전환**.

### Cleanup Whitelist 보존
- USD/oracle_policy.py/collect_demos.py/train_bc.py/validate_oracle.py unchanged
- so_arm101.py: `cp .session7_pre` 백업 후 lessons.md §6.1 cfg 재구성 (사용자 승인)
- 신규: tasks/demos_session7_target100.pt, oracle_100ep_session7_*.log, introspect_session7.log

### 진정한 게이트 결과 (codex 정의)
- 100ep success rate ≥ 60%: ❌ (15%)
- ratio ≥ 0.5: ❌ (decimation=1 효과로 baseline 0% → 15% 입증, 정량은 다음 세션 측정)
- 그러나 demo 수집 102 성공 → BC track 진입 가능

---

# 세션 8 — Demo 확장 + BC 학습 + 단독 평가 + 영상 protocol (2026-05-09)

## 사전 점검
- 카메라 cfg **없음** (lift task) → 추가 필요 (사용자 승인 사항)
- TiledCameraCfg 가용
- imageio 2.37.3 ✓
- validate_oracle.py `--seed` 인자 ✓
- 세션 7 demo 파일: `tasks/demos_session7_target100.pt` (102 demos, 8.6 MB) — **삭제/덮어쓰기 절대 금지**

## 사고 방지 원칙 (세션 8 추가)
- git checkout 금지 (세션 6 사고 재발 방지)
- cfg 수정 전 cp .session8_pre 백업
- demos_session7 파일 보호
- 학습 launch 는 BC 만, PPO 는 다음 세션
- jupyter07 headless — viewport capture 불가, TiledCamera 가 유일한 영상 source

## User plan 의 driver 가정 정정
- `oracle_bypass_v1.py` 미존재 (세션 7 에서 wrapper 불필요로 안 만들었음)
- 대신 사용:
  - oracle 100ep 평가 = `validate_oracle.py` (CLI 인자 그대로)
  - oracle demo 수집 = `collect_demos.py` (CLI 인자 그대로)
  - Render 통합 wrapper = 신규 `tasks/render_policy.py`
  - BC 단독 평가 = 신규 `tasks/eval_bc_session8.py`
  - BC 학습 50Hz wrapper = 신규 `tasks/train_bc_50hz.py`

## 작업 0 — 카메라 cfg + render 인프라

- [ ] 0.1 카메라 추가 위치 + 디자인 user 승인 (옵션 A/B/C)
- [ ] 0.2 cfg 적용 (cp .session8_pre 백업) + introspect 검증
- [ ] 0.3 `tasks/render_policy.py` 작성 (oracle/BC/PPO 통합)
- [ ] 0.4 세션 7 oracle 영상 3 seed 사후 생성

## 작업 1 — Baseline 3 seed robustness

- [ ] validate_oracle.py --seed 1/2/3 100ep, 4.C threshold 동일
- [ ] mean/std 표
- [ ] 게이트: mean ≥10% AND std <5pp → 작업 2 진행

## 작업 2 — Demo 확장 (target 400)

- [ ] collect_demos.py 추가 300 수집
- [ ] python merge 스크립트로 102+300 = 400 demos 병합
- [ ] z_max/length/final_state 분포 표

## 작업 3 — BC 학습 (action repeat 2)

- [ ] `tasks/train_bc_50hz.py` 작성 (train_bc.py 호출, action repeat dataset wrapper)
- [ ] tmux khj_bc 백그라운드 실행 (30 epoch)
- [ ] loss curve

## 작업 4 — BC 단독 평가 + 영상

- [ ] `tasks/eval_bc_session8.py` 작성
- [ ] 100ep success rate (action repeat 2)
- [ ] BC 영상 3 seed
- [ ] 게이트: ≥30% / 15-30% / 5-15% / <5%

## 작업 5 — commit + lessons.md §8

## Session 8 Result Summary

### 작업 0 — 카메라 cfg + render 인프라
- `SoArm101LiftCubeEnvCfg_VIDEO` 신규 (joint_pos_env_cfg.py, num_envs=1, top-down TiledCamera 320×240)
- task ID `Isaac-SO-ARM101-Lift-Cube-Video-v0` 등록 (`tasks/lift/__init__.py`)
- `tasks/render_policy.py` 작성 (oracle/bc 통합 wrapper, imageio mp4)
- 세션 7 oracle 영상 3 seed 사후 생성: `videos/session7/oracle_4c_v1_seed{0,1,2}.mp4` (각 ~190 KB, 500 frames)

### 작업 1 — 3 seed robustness 게이트 통과

| Seed | success rate |
|---|---|
| (default, 세션 7) | 15% |
| 1 | 7% |
| 2 | 14% |
| 3 | 16% |
| **mean** | **13%** ≥ 10% ✓ |
| **std** | **4.0pp** < 5pp ✓ |

### 작업 2 — Demo 확장 (102 + 300 = 402)
- `collect_demos.py --target_episodes 300 --seed 7` 추가 수집
- 300 demos / 1824 ep / 510 sec / rate 16.45%
- `tasks/merge_demos.py` 작성 (codex 권고 metadata 정합성 check)
- 병합 결과: `tasks/demos_session8_total400.pt` (34 MB, **402 demos**)
- z_max: min 0.100 / mean 0.129 / max 0.189
- ep length 모두 500 (timeout to LIFT)
- final_state: 399 LIFT (3) + 3 APPROACH (0)

### 작업 3 — BC 학습 (50Hz dataset, codex #3)
- `tasks/downsample_demos_50hz.py` 작성, stride=2 → `demos_session8_total400_50hz.pt` (17 MB)
- 100,500 transitions (402 demos × 250 step)
- `train_bc.py --epochs 30 --batch_size 256 --lr 1e-3` (in conda python, GPU)
- train loss: 0.00489 → 0.00112 (4× 감소)
- val loss: 0.00137 → 0.00107 (1.3× 감소)
- saved: `tasks/bc_actor_session8_v1.pt`

### 작업 4 — BC 단독 평가 + 영상

`tasks/eval_bc_session8.py` 작성. action repeat 2 적용.

```
[DONE] BC episodes=100 steps=3500 elapsed=25s
[DONE] success=14 fail=86 success_rate=14.00%
[DONE] z_max: min=0.012 q25=0.012 median=0.042 q75=0.088 mean=0.051 max=0.194
```

**게이트 비교:**
- Plan 원안 ≥30%: ❌ FAIL (14%)
- Codex 권고 ≥10~15%: ✓ 통과 (14% in range)
- Oracle 4 seed mean 13% 와 동등 — BC 가 oracle 비효율 그대로 학습 (codex 예상)

BC 영상 3 seed: `videos/session8/bc_v1_seed{0,1,2}.mp4` (각 ~150-200 KB)

### Cleanup Whitelist 보존 ✓
- USD/oracle_policy.py/collect_demos.py/train_bc.py/validate_oracle.py 모두 unchanged
- demos_session7_target100.pt 보존 (병합 source)
- 수정: joint_pos_env_cfg.py + tasks/lift/__init__.py (사용자 승인 후 카메라 variant + task 등록)
- 백업: *.session8_pre

### 신규 산출물
- `tasks/render_policy.py`, `tasks/merge_demos.py`, `tasks/downsample_demos_50hz.py`, `tasks/eval_bc_session8.py`
- `tasks/demos_session8_extra300.pt` (300 demos)
- `tasks/demos_session8_total400.pt` (402 demos, 병합본)
- `tasks/demos_session8_total400_50hz.pt` (50Hz downsample)
- `tasks/bc_actor_session8_v1.pt`
- `tasks/bc_train_session8.log`, `tasks/bc_eval_session8.log`
- `videos/session7/oracle_4c_v1_seed{0,1,2}.mp4`
- `videos/session8/bc_v1_seed{0,1,2}.mp4`

### 다음 세션 (Stage B+ )
1. **`train.py --bc_init` 추가** — rsl_rl actor 가 BC actor weights 로 초기화 (smoke_test #5 호환 이미 확인)
2. **PPO warmstart** with BC init weights + 4.C threshold + decimation=1 cfg
3. PPO 결과 영상 녹화 (videos/session9/)
4. 추가 후보: max_ee_step sweep (codex #3 권고), demos 200~500 확장 (codex #2 권고)

---

# 세션 9 — PPO Warmstart + Iter 게이트 + 영상 재녹화 (2026-05-09)

## 사전 점검 결과
- branch khj-rl-track, git clean
- BC actor: tasks/bc_actor_session8_v1.pt 보존
- demos_session{7,8}*.pt 보존
- train.py: line 196-202 의 resume 분기 옆에 BC init 분기 추가 가능
- **PPO cfg 현재 값 (user plan 가정과 다름)**:
  - learning_rate = 1e-4 (user plan 1e-3 잘못 추정)
  - entropy_coef = 0.01 (변경 권고 0.02)
  - desired_kl = 0.01 ✓ (이미 적용)
  - save_interval = 50 ✓ (이미 적용)
  - schedule = "adaptive" ✓
- cli_args 가 `--experiment_name` 이미 노출 ✓

## 사고 방지 원칙
- git checkout 금지
- 모든 cfg 수정 전 cp tasks/backups_session9/<file>.pre
- 학습 launch 후 cfg 수정 X
- view 첫 frame 사용자 확인 후만 영상 녹화

## 작업

### 1. train.py --bc_init 추가
- [ ] tasks/backups_session9/ 생성
- [ ] cp backups
- [ ] argparse --bc_init 인자 추가 (line 33 다음)
- [ ] runner.load(resume) 옆에 elif --bc_init: load actor only (strict=False)
- [ ] git diff 사용자 승인

### 2. PPO cfg 보수화
- [ ] cp 백업
- [ ] entropy_coef 0.01 → 0.02 (user plan 권고)
- [ ] lr 1e-4 그대로 (user plan 가정 정정)
- [ ] save_interval 50 ✓ (이미)
- [ ] desired_kl 0.01 ✓ (이미)
- [ ] git diff 사용자 승인

### 3. PPO 학습 launch (tmux khj_ppo)
- [ ] `train.py --task Isaac-SO-ARM101-Lift-Cube-v0 --num_envs 4096 --headless --max_iterations 1000 --bc_init bc_actor_session8_v1.pt --experiment_name session9_v1`
- [ ] tee tasks/ppo_train_session9.log
- [ ] 1~2 시간 예상

### 4. 카메라 view 재설정 (학습 도중 병행 — GPU 충돌 시 학습 후로)
- [ ] cp 백업
- [ ] joint_pos_env_cfg.py 의 SoArm101LiftCubeEnvCfg_VIDEO 카메라 좌표 수정
- [ ] view 1 (top-down 멀게), view 2 (diag_front)
- [ ] render 첫 frame 1 step → 사용자 확인 (max 2-3 반복)
- [ ] 만족 시 작업 6 영상 녹화 진행

### 5. iter 50/100/200 평가 + 게이트
- [ ] tasks/eval_ppo_session9.py 작성
- [ ] checkpoint 자동 평가 (success rate, z_max, KL, entropy, state-at-done)
- [ ] iter 200 게이트 ≥25% → 1000 까지 / 미통과 → 사용자 결정

### 6. 영상 + 최종 평가
- [ ] PPO checkpoint 영상 (3 iter × 2 view × 3 seed)
- [ ] 세션 7/8 영상 archive 후 새 view 재녹화
- [ ] 최종 100ep 평가
- [ ] oracle/BC/PPO 비교 표

### 7. commit + lessons §9

## Session 9 Result Summary

### 작업 1+2 — train.py --bc_init + PPO cfg
- `tasks/backups_session9/{train.py.pre, rsl_rl_ppo_cfg.py.pre, joint_pos_env_cfg.py.pre}` 백업
- train.py: `--bc_init` 인자 추가 (line 33+), runner 생성 후 `runner.alg.policy.actor.load_state_dict(strict=False)` (rsl-rl 2.3.x 패턴, `actor_critic` 아니라 `policy`)
- PPO cfg: entropy_coef 0.01 → 0.02 (codex 권고)
- lr 1e-4 그대로 (user plan 1e-3 가정 잘못 정정)

### 작업 3 — PPO 학습 (15 min, 1000 iter)
```
[BC init] loaded 8 actor params; missing=[]; unexpected=[]
1000 iter: Total timesteps 98M, elapsed 14:46
final reward: 0.93, ep_length 500 (timeout), action_std 54.92
```
checkpoint: `~/jabis_sim/sim2real/oracle/logs/rsl_rl/lift/2026-05-09_20-01-15/`
saved every 50 iter (model_0/50/100/.../999.pt)

### 작업 5 — iter 별 평가 (게이트 ❌ FAIL)

| iter | success | z_max max | entropy(std) | action MSE vs BC |
|---|---|---|---|---|
| 0 (BC init) | **14%** | 0.189 | 1.01 | 0.002 (BC와 거의 동일 ✓) |
| 50 | 15% | 0.219 | 3.24 | 0.92 |
| 100 | 15% | **0.341** ⭐ | 7.85 | 20.7 |
| 200 | 13% | 0.189 | 21.8 | 540 |
| 500 | 13% | 0.189 | 32.7 | 9300 |
| 999 | **10%** | 0.149 (감소) | **54.9** | 46866 |

**Codex caveat 1 (saddle point) + caveat 2 (entropy 0.02 BC 흐릴 위험) 정확히 적중**:
- iter 0 의 14% 가 BC 단독 14% 와 정확 일치 → **BC init load 검증 완벽**
- entropy 0.02 가 너무 강함 → action_std explode (1 → 55, 54×)
- iter 100 까지 일부 explore success (z_max max 0.34, BC 0.19 의 1.8×)
- iter 200+ 부터 BC weights 손상 + 발산
- **PPO 가 BC baseline 못 넘음**: 14% → 10% (오히려 감소)

**게이트 결과:**
- iter 200 ≥ 25%: ❌ FAIL (13%)
- iter 500 ≥ 40%: ❌ FAIL (13%)
- iter 1000 ≥ 50%: ❌ FAIL (10%)

### 작업 4+6 — 카메라 view 재설정 + 영상 9개

카메라 변경 (cp .pre 백업):
- pos z 0.6 → 1.2 (거리 멀게)
- focal_length 24 → 18 (wider FOV)
- clipping 5.0 → 10.0

영상 archive + 새 view 재녹화:
- archive_old_view/: 세션 7+8 의 기존 6 mp4 (좁은 view)
- session7/: oracle_4c_v1_seed{0,1,2}.mp4 (3개)
- session8/: bc_v1_seed{0,1,2}.mp4 (3개)
- session9/: ppo_iter{100,999}_seed{0,1,2}.mp4 (6개)

총 12 mp4 (3 oracle + 3 BC + 6 PPO).

render_policy.py 에 `--policy ppo --ppo_ckpt` 모드 추가.

### oracle / BC / PPO 비교 표

| metric | oracle | BC | PPO@iter100 | PPO@iter999 |
|---|---|---|---|---|
| success rate | 13% (4 seed mean) | 14% | 15% | **10%** |
| z_max max | 0.189 | 0.194 | **0.341** | 0.149 |
| z_max mean | 0.129 | 0.051 | (n/a) | 0.047 |

PPO 가 BC saddle point 못 넘고 entropy 발산으로 손상됨.

### Cleanup Whitelist 보존 ✓
- USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged
- 수정: train.py + rsl_rl_ppo_cfg.py + joint_pos_env_cfg.py (백업 후 사용자 승인)

### 다음 세션 권고
1. **entropy 조정**: 0.02 → 0.005 또는 0.001 (action_std 발산 방지)
2. **또는 noise_std fix** (학습 X)
3. **또는 BC actor frozen + critic only burn-in** (separate phase)
4. **또는 demos stratify** (codex caveat 4) — cube init 위치 별 BC 성공률 분석
5. iter 100 시점 (z_max max 0.341) 의 hyperparam 영역에서 학습 멈추는 early-stop 또는 KL-bound 도입

---

# 세션 10 — 카메라 정리 + Entropy Ablation (2026-05-09)

## 사전 점검
- 현재 카메라 1개 (`tiled_camera`), ROS convention + rot=(0,0,1,0). user 보고 "옆 누운 시점" → quaternion 또는 convention 잘못
- render_policy.py 의 grab_rgb 가 `tiled_camera` 단일만 access — multi-view 지원 X
- rsl-rl ActorCritic `self.std = nn.Parameter(...)` (학습됨). 시도 3 (fixed std) 는 monkey patch 필요

## 사고 방지
- git checkout 금지
- cp .session10_pre 백업 필수
- 첫 frame 검증 후만 학습
- 시도 순차 (GPU 1번 충돌 회피)

## 작업

### 0. 카메라 view 정리
- [ ] cp 백업
- [ ] joint_pos_env_cfg.py: top_camera + diag_camera 2개. OpenGL convention 시도
- [ ] render_policy.py grab_rgb: --views 인자, 카메라별 mp4
- [ ] 첫 frame test → 사용자 확인 (max 3 반복)

### 1. 시도 1 (entropy 0.005)
- [ ] PPO cfg 백업 + entropy 0.02→0.005, lr 1e-4 유지
- [ ] 학습 launch (tmux khj_ppo_v1, experiment_name session10_v1_ent005)
- [ ] iter 0/100/200/500/999 평가
- [ ] 게이트: iter999 ≥ 25% 또는 action_std < 10

### 2. (조건부) 시도 2 (entropy 0.001)
- [ ] 시도 1 미통과시. lr 3e-4 도 함께
- [ ] experiment_name session10_v2_ent001

### 3. (조건부) 시도 3 (noise_std fixed)
- [ ] 시도 1/2 미통과시. entropy=0, init_noise_std=0.5, std parameter requires_grad=False (monkey patch)
- [ ] experiment_name session10_v3_fixed_std

### 4. Best 영상 + 7/8 재녹화
- [ ] 채택 시도 best ckpt PPO 영상 (top+diag, 3 seed)
- [ ] 세션 7/8 영상 archive 후 새 view 재녹화 (oracle, BC)

### 5. 학습 곡선 plot
- [ ] success rate / action_std / MSE vs BC 추이 PNG

### 6. commit + lessons §10
- [ ] 시연 strategy 사용자 권고 (PPO best ≥30% / 20-30% / <20% 분기)

## Session 10 Result Summary

### 작업 0 — 카메라 view 2개 + render multi-view
- cp 백업: backups_session10/{joint_pos_env_cfg, rsl_rl_ppo_cfg, render_policy}.session10_pre
- joint_pos_env_cfg.py SoArm101LiftCubeEnvCfg_VIDEO: top_camera (OpenGL identity, pos z=1.2) + diag_camera (look_at quat, pos=(0.7,0.5,0.5))
- render_policy.py: --views 인자, sensor_name='<view>_camera' dispatch, mp4 별도 저장
- 첫 frame test → user 확인 (diag 처음 누운 시점 → numpy R→quat 정확 계산 (0.3354, 0.1841, 0.4446, 0.8099) → OK)

### 작업 1 — v1 (entropy 0.005)
- entropy 0.02 → 0.005, lr 1e-4 유지
- 학습 15:40, std 발산 회피 (max 2.57 vs v0 54.92, **21× 작음**)
- reward 1.98 (v0 0.93의 2.1×)
- 평가: iter 0=14% / 100=13 / 200=14 / 500=13 / 999=**13%**
- → BC saddle 못 넘음, 게이트 25% FAIL

### 작업 2 — v2 (entropy 0.001 + lr 3e-4)
- entropy 0.005 → 0.001, lr 1e-4 → 3e-4
- 학습 15:00, std **0.15** (v0 54.92의 366× 작음, 완전 안정)
- reward 4.54 (v0의 4.9×)
- 평가: iter 0=14 / 100=13 / 200=13 / 500=13 / 999=**13%**
- → 여전히 BC saddle, 게이트 25% FAIL

### 작업 4 — 영상 (3 정책 × 2 view × 3 seed = 18 mp4)
- archive_session_9_view/: 세션 9 구 view 12 mp4
- session7/: oracle_4c_v1_{top,diag}_seed{0,1,2}.mp4 (6)
- session8/: bc_v1_{top,diag}_seed{0,1,2}.mp4 (6)
- session10/: ppo_v2_iter999_{top,diag}_seed{0,1,2}.mp4 (6)

### 작업 5 — 학습 곡선 plot
`tasks/ablation_session10_curves.png` (155 KB):
- 1×3 figure: success rate / action_std (log) / MSE vs BC (log)
- v0 vs v1 vs v2 비교
- BC baseline 14% + gate 25% reference line

| iter | v0 succ | v0 std | v1 succ | v1 std | v2 succ | v2 std |
|---|---|---|---|---|---|---|
| 0 | 14% | 1.01 | 14% | 1.00 | 14% | 1.00 |
| 100 | 15 | 7.85 | 13 | 1.22 | 13 | 0.69 |
| 200 | 13 | 21.8 | 14 | 1.61 | 13 | 0.50 |
| 500 | 13 | 32.7 | 13 | 2.37 | 13 | 0.21 |
| 999 | **10%** | 54.9 | **13%** | 2.57 | **13%** | **0.15** |

### 핵심 결론
- v0 의 entropy 발산 (54×) 정확히 회피됨 (v2 std 0.15)
- reward shape 정상화 (v0 0.93 → v2 4.54, **4.9× 개선**)
- **그러나 success rate 13~14% plateau — BC saddle exit 가 entropy 문제 아님 확정**
- → codex caveat 4 **stratify (demo coverage)** 우선순위 입증

### 시연 strategy 권고 (사용자 결정 사항)
- PPO best 13% < 20% → **Demo path 권고**:
  - 주력 시연: BC 14% (lift 0.04~0.19m 분포) + Oracle 4.C 영상
  - PPO 영상: ablation 자료
- 다음 세션: demos stratify (cube init 위치별 BC perf 분석) + 시도 3 (fixed_std)

### Cleanup Whitelist 보존 ✓
- USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged
- 수정: rsl_rl_ppo_cfg.py (entropy 0.005→0.001, lr 1e-4→3e-4) + joint_pos_env_cfg.py 카메라 + render_policy.py 멀티뷰 (사용자 승인 + cp 백업)
- 세션 9 backup 유지

### 다음 세션 권고
1. **demos stratify** (codex caveat 4): cube init xy/z 별 BC 14% 분포 분석. saddle 의 coverage 부족 진단.
2. **시도 3 (noise_std fixed)** — entropy=0 + std requires_grad=False monkey patch
3. demos 추가 수집 — cube init 다양화 또는 oracle hyperparam 변경으로 더 다양한 trajectory
4. 시연 video 자료 충분 (18 mp4, top+diag), 발표 path 결정 자료 OK

---

# 세션 11 — Reward Shape Fix (lifting↑, joint_vel↓) (2026-05-09)

## 사전 점검 결과

`lift_env_cfg.py:157 RewardsCfg`:
- `reaching_object` weight=1.0, mdp.object_ee_distance(std=0.05)
- `lifting_object` weight=15.0, mdp.object_is_lifted(minimal_height=0.025)
- `object_goal_tracking` weight=16.0
- `action_rate` weight=-1e-4 → -1e-1 (curriculum 10k steps)
- `joint_vel` weight=-1e-4 → -1e-1 (curriculum 10k steps)

학습 log mean reward (세션 10 v2):
- reaching 0.70 / lifting 0.16 / action_rate -0.07 / joint_vel -0.39 / total 4.54

→ lifting weight 15 임에도 lifted (z≥0.025) step 비중 작아 평균 reward 작음. weight 늘리거나 minimal_height 완화.

## 사고 방지
- git checkout 금지
- cp .session11_pre 백업 (lift_env_cfg.py)
- 학습 launch 후 cfg 수정 X
- 시도 1 결과 < 18% → 시도 2 진행
- 시도 1+2 < 18% → 옵션 1 (Demo path) 즉시 전환

## 작업

### 1. reward 진단 + 사용자 승인
- [ ] 시도 1 cfg: lifting 15→45, joint_vel -1e-1→-3e-2 (curriculum 변경)

### 2. 시도 1 학습 + 평가
- [ ] cp 백업, edit, git diff 사용자 승인
- [ ] 학습 launch (1000 iter, 15 min, experiment_name session11_v1_reward1)
- [ ] iter 0/200/500/999 평가
- [ ] reward decomposition v2 vs v1 비교

### 3. 게이트
- ≥25% → 채택, 작업 4 (영상)
- 18~24% → 시도 2
- <18% → 시도 2

### 4. (조건부) 시도 2: reaching 1.0→0.5 + lifting 15→75
### 5. (조건부 ≥25%) 영상 + 비교
### 6. commit + lessons §11 + session11_report.md
### 7. 시연 path 결정 (Path A/B/C, 사용자)

## Session 11 Result Summary

### 작업 1+2 — Reward fix v1 (lifting 3×, joint_vel 0.3×)
- cp 백업: backups_session11/lift_env_cfg.py.session11_pre
- lift_env_cfg.py: lifting 15→45, joint_vel curriculum -1e-1→-3e-2
- 학습 15:40, std 0.39, mean reward 2.02
- reward decomp: reaching 0.70→0.020 (35× ↓), lifting 0.16→0.470 (3× ↑ 정확)
- 평가: iter 0=14%, 200=12, 500=13, 999=**13%**
- → 게이트 18% FAIL → 시도 2 진행

### 작업 3 — Reward fix v2 (reaching 0.5×, lifting 5×)
- 추가 변경: reaching 1.0→0.5, lifting 45→75 (15→75 = 5×)
- 학습 15:13, std 0.48, mean reward 3.38
- reward decomp: reaching 0.020→**0.001** (20× ↓), lifting 0.470→**0.817** (1.7× ↑ 정확)
- 평가: iter 0/200/500/999 모두 **14%** (BC 정확 동등)
- → 게이트 18% FAIL

### 핵심 결론
**Reward shape 문제 아님 확정**. lifting 가 reaching 의 800× reward 였음에도 success 14% plateau.

| 가설 | 세션 | 결과 |
|---|---|---|
| Entropy 발산 | 9 (0.02) | 10% |
| Entropy 보수 | 10 v1 (0.005) | 13% |
| Entropy 극보수 | 10 v2 (0.001+lr3e-4) | 13% |
| Reward shape v1 | 11 v1 | 13% |
| Reward shape v2 | 11 v2 | 14% |

**3 가설 (entropy / reward shape / fixed_std) 모두 무효** → **codex caveat 4 (demos coverage)** 만 남음.

### 작업 4 — v2 영상 (best 아니라 ablation 자료)
- 6 mp4 (~/jabis_sim/day5/videos/session11/ppo_v2_reward2_{top,diag}_seed{0,1,2}.mp4)
- 메시지: "reward 압도적으로 lift 보상 → PPO 가 lift 시도 적극 → 그러나 saddle 못 넘음"

### 시연 strategy (Codex 권고대로 사용자 결정)
**Path C (Demo path) 권고**:
- Oracle 13% (state machine 가능성 입증)
- BC 14% (재현 가능한 부분 lift)
- PPO ablation (entropy + reward 진단 자료)

### Cleanup Whitelist 보존 ✓
- USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged
- 수정: lift_env_cfg.py (사용자 승인 + cp 백업)

### 다음 세션 권고 (saddle exit 의 진짜 원인)
1. **Demos stratify** (codex caveat 4) — cube init 별 BC 성공률 분석
2. **Demos 재수집 with cube_pos meta** — collect_demos.py wrapper (사용자 승인 필요)
3. **사용자 결정**: stratify 1세션 추가 vs Demo path 확정 RL 트랙 종료

---

# 세션 12 — Demos Coverage 분석 + RL 트랙 종료 확정 (2026-05-10)

## 사전 점검 — Plan 가정 invalid 발견

`tasks/demos_session8_total400.pt` 의 cube init 분포 분석:

| 축 | mean | std | range |
|---|---|---|---|
| cube_x (obs idx 20) | 0.1994 | **0.061** (6.1cm) | [0.10, 0.30] |
| cube_y (obs idx 21) | -0.0086 | **0.134** (13.4cm) | [-0.20, +0.20] |
| cube_z (obs idx 22) | 0.012 | 0 | constant |

→ user plan 가정 (±2cm 좁은 분포) **invalid**. 이미 광범위 randomize:
- cube_x: ±10cm 분포 (default 0.20 ± 5cm or more)
- cube_y: ±20cm 분포 (default 0.0 ± 13cm)

z_max distribution: range [0.100, 0.189], mean 0.129 (success_z=0.10 통과).
final_state: 399 LIFT + 3 APPROACH (99.3% LIFT).

`tasks/demos_session8_coverage.png` (산포도 + 히스토그램, 발표 자료).

## 사용자 결정 — RL 트랙 종료 + sim2real 전환 (Path C 확정)

**근거**:
- 5세션 ablation (entropy / reward / fixed_std) 모두 13~14% plateau
- demos coverage 광범위 — saddle 의 진짜 원인 아님 확정
- 추가 RL 시도의 효과 보장 X
- 양팔 시연 한 팔 RL 트랙 자료 충분 (Oracle 13% / BC 14% / PPO ablation)

**작업 2~5 (cube init 확장 + demos 재수집 + BC 재학습 + PPO) skip**.

## 작업

- [x] 1. demos coverage 분석 (`analyze_demos_coverage.py`, PNG 생성)
- [ ] 6+7. session12_report.md + lessons §12 + commit

## Session 12 Result Summary

### 핵심 발견
402 demos 의 cube init 이미 ±10~20cm 범위 randomize. **demos coverage 가 BC saddle 의 원인 아님** 확정. → **5세션 ablation 의 마지막 가설 (codex caveat 4) 도 무효**.

### RL 트랙 5세션 누적 ablation 결론

| 가설 | 결과 |
|---|---|
| H1 mimic dead | PASS (sess 2) |
| H3 actuator 6 vs 10 | PASS |
| **decimation 2→1** | **TRUE FIX** (0% → 15%, sess 7) |
| H7 entropy 발산 (0.02) | 10% (sess 9) |
| H7 entropy 보수 (0.005, 0.001) | 13% (sess 10) |
| H8 reward shape (lift 75×) | 14% (sess 11) |
| H9 demos coverage | 무효 (이미 광범위, sess 12) |

→ **15% ceiling 의 진짜 원인은 진단 5세션 안 풀림**. 환경 fix 14pp (0%→14%) 후 추가 ablation 효과 X.

### 시연 strategy — Path C 확정

- **Oracle 4.C threshold + decimation=1**: 13% (state machine + 환경 fix 입증)
- **BC 14%** (메인 시연 자료): 재현 가능한 부분 lift, z_max 0.10~0.19m 분포
- **PPO ablation v9~v11**: entropy 발산/회피 + reward shape 정확 분리 (5세션 진단 자료)

### 다음 세션 (RL 종료 후)
1. **Sim2real 진입** — BC actor + Oracle policy 를 실 SO-ARM101 hw 로 deploy
2. action_repeat 2 (sim 100Hz → hw 50Hz) 검증
3. 양팔 시연 리허설

---

# 세션 13 — Policy Behavior 진단 (2026-05-10 재개)

## 핵심 발견 — saddle 진짜 원인 식별
**ee_to_cube_min < 5cm**: Oracle 86% / **BC 2% / PPO 1%** → BC/PPO 가 cube 에 못 가서 14% 만 우연 catch.

## 가설 PASS/FAIL
| 가설 | 결과 |
|---|---|
| 1 도달 못 함 | **PASS ⭐** |
| 4 좁은 영역만 성공 | FAIL (분포 동일) |
| **5 BC trajectory drift** (NEW) | **유력** |

## 작업 결과 ✓
- diagnose_policy.py + 100ep × 3 정책
- Policy behavior 비교 표
- Success vs Fail scatter (session13_success_vs_fail.png)
- 12 mp4 (BC + Oracle behavior 비교, videos/session13/)
- session13_report.md, lessons §13
- commit `ab340fb`

## 사용자 결정 권고 (RL 종료 재검토)
- (a) Fix 3 (reward distance std 0.05→0.02) 1세션 ablation
- (b) Path C 확정 유지 — sim2real
- (c) DAgger 1~2세션

---

# 세션 14 (2026-05-10) 결과 — Reward distance std 0.05→0.02 ablation

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. cfg 백업 + std 0.05→0.02 | ✓ (`tasks/backups_session14/lift_env_cfg.py.session14_pre`) |
| 1.3 reward landscape plot | ✓ `tasks/reward_landscape.png` |
| 2. PPO 학습 (1000 iter, 15분) | ✓ ckpt `~/jabis_sim/sim2real/oracle/logs/rsl_rl/lift/2026-05-10_01-35-37/` |
| 3+4. iter 평가 + 비교 표 | ✓ 9 iter (0/100/150/200/250/300/400/500/999) |
| 5. 영상 (iter 200, iter 999) | ✓ 12 mp4 `~/jabis_sim/day5/videos/session14/` |
| 6. 시연 path 결정 | **사용자 결정 보류** (success 14% → Path B/C 영역) |
| 7. commit + report | ✓ |

## 핵심 결과

- **iter 200 sweet spot**: ee<5cm 도달율 **31%** (s13 BC 2% / PPO 1% 대비 15× 개선)
- **iter 999 collapse**: ee<5cm 2%, close 시도 0%
- **success rate 14% 일정** (도달 spike 시점에도 close timing 무너짐)
- **핵심 게이트 ≥50% 미달** → fix 1 단독 부족 확정. 그러나 **방향 옳음 입증**

## 새 가설 (세션 14)

- 가설 6: sharper reward → PPO exploration 망가짐
- 가설 7: lifting reward 가 close 직접 보상 안 함 (가장 promising)
- 가설 8: entropy 가 후반에 close 행동 잊음

## 사용자 결정 사항

- 옵션 A: Fix 7 (close reward 직접 추가) + std=0.02 유지 — 1세션
- 옵션 B: DAgger 1~2세션
- 옵션 C: Path B 확정 (BC 메인, PPO iter 200 보조) → sim2real
- 옵션 D: Path C 확정 (RL 종료) → sim2real

---

# 세션 15 (2026-05-10) 결과 — Close Reward + Early Stop

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. cfg 백업 + grasp_object RewTerm 추가 | ✓ (기존 gripper_closure_near_object 활용) |
| 2. PPO 200 iter 학습 | ✓ ckpt `~/jabis_sim/sim2real/oracle/logs/rsl_rl/lift/2026-05-10_02-29-45/` |
| 3+4. iter 평가 (5 ckpt) + 비교 표 | ✓ tasks/diagnose_session15_iter{0,50,100,150,199}.csv |
| 5. 영상 (iter 100, 199) | ✓ 12 mp4 ~/jabis_sim/day5/videos/session15/ |
| 6. 시연 path 결정 | **Path C 확정** (모든 게이트 미달) |
| 7. commit + report | ✓ |

## 핵심 결과

- s15 iter100 sweet spot: 도달율 21%, success 15%
- s14 iter200 (31%) 보다 **낮음** — close reward 가 reaching 학습 방해
- close% 활성 (99~100) 그러나 timing 잘못 (close_dist 18cm)
- 모든 핵심 게이트 미달 → Path C 확정

## 다음 세션 — Path C Sim2Real

옵션 A (권장): BC actor (s8 v1) 메인 + Oracle 백업
옵션 B: + s14 iter200 PPO 보조 (도달 31% 시연 가치)
옵션 C: RL 추가 (lifting weight ablation) — 시간 추가

---

# 세션 16 (2026-05-10) 결과 — PathOn-AI Baseline 평가 시도 (실패)

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. model_1950 평가 | ❌ obs 36 vs 28 mismatch |
| 2. model_2100 평가 | ❌ 동일 |
| 3. obs strip last_action | ❌ 30 vs 28 mismatch |
| 4. action dim 확인 | ❌ PathOn 6 vs 우리 8 (gripper 부재) |
| 5. 시나리오 분기 | **평가 불가, Path C 유지** |
| 6. tasks/pathon_ai_comparison.md | ✓ |
| 7. lessons.md §16 | ✓ |

## 핵심 발견

- 동일 task name 이지만 **fundamentally 다른 환경**
- PathOn = 6-DOF arm only (gripper 없음)
- 우리 = 6 arm + gripper binary 양손 mirrored (8 action)
- "high success rate" 주장은 다른 task 환경에서 측정

## 다음 단계

- **Path C 확정 유지** (5/12 sim2real)
- BC actor (s8 v1) + Oracle 백업 시연 자산
- 선택 (시간 있으면): gripper-less 환경 ablation — saddle 이 gripper 학습 자체 문제인지 검증

---

# 세션 17 (2026-05-10) 결과 — Gripper-less Ablation + Presentation 정리

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. gripperless cfg variant + 백업 | ✓ |
| 2. PPO 2000 iter 학습 (32분) | ✓ ckpt `~/jabis_sim/sim2real/oracle/logs/rsl_rl/lift/2026-05-10_03-25-32/` |
| 3+4. iter 평가 (6 ckpt) + 비교 표 | ✓ |
| 5. gripperless 영상 (iter 1999, 6 mp4) | ✓ |
| 6. presentation 큐레이션 (7 mp4 + 3 plot) | ✓ |
| 7. 시연 path 결정 | **Path C 확정** (gripperless 도 14%) |
| 8. report + commit | ✓ |

## 핵심 결과

- **가설 FAIL — gripper 무관** (gripperless final 14%, BC saddle 동일)
- gripperless 도달율 max 5% (s14 iter200 31% 보다 낮음 — gripper 학습이 reaching incentive 제공)
- **Saddle = 환경/task 자체 한계** (cube random init 우연 catch)
- 4세션 ablation (13~17) 종합 → Path C 확정

## 다음 세션 (5/12) — Sim2Real

- BC actor `tasks/bc_actor_session8_v1.pt` 메인
- Oracle 백업
- action_repeat 2
- 두 팔 시연 rehearsal

## Presentation 자산

- 7 mp4 (Oracle / BC / s14 iter200 / s14 iter999 / s17 gripperless 등)
- 3 plot (ablation_evolution / reward_landscape / session13_success_vs_fail)

---

# 세션 18 (2026-05-10) 결과 — Task Spec Ablation (Oracle 14% 진짜 원인)

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. 백업 (lift_env_cfg / joint_pos_env_cfg) | ✓ `tasks/backups_session18/` |
| 2. Ablation 1 (episode 5s→10s) + Oracle 100ep | ✓ `tasks/diagnose_oracle_session18_a1_ep10s.csv` |
| 3. Ablation 2 (+y ±20cm→±10cm, A1 위 stack) + Oracle 100ep | ✓ `tasks/diagnose_oracle_session18_a2_ep10s_y10cm.csv` |
| 4. env 원복 + Ablation 3 (success_z 0.10→0.05) + Oracle 100ep | ✓ `tasks/diagnose_oracle_session18_a3_z5cm.csv` |
| 5. 비교 표 + saddle 원인 식별 | ✓ session18_report.md §5–6 |
| 6. report + lessons + commit | ✓ |

## 핵심 결과

| ablation | success | 변화 vs baseline |
|---|---|---|
| baseline (s13) | 15.0% | - |
| A1 (ep 10s) | 15.0% | **0** (episode 길이 무관) |
| A2 (+y 10cm) | 2.0% | -13 (역효과) |
| **A3 (z=0.05m)** | **44.0%** | **+29** (최대) |

**진짜 saddle 원인 (codex 검증, bimodal)**:
- A. **mid-lift 26/85 (31%)**: Oracle LIFT가 z=5–10cm까지 도달하나 임계 10cm 미달 → success_z 5cm 완화 시 +29%p (A3 = 44%) 회복
- B. **no-lift 49/85 (58%)**: grip 후 cube가 z<2.5cm → 시간/임계 무관, 메커니즘 미식별 (다음 세션 진단)

→ **Oracle 자체가 14% saddle 만든 주범**. RL/BC 알고리즘 결함 아님. Saddle은 두 개의 분리된 문제로 분해됨.

## 5세션 ablation의 진짜 교훈

s13–s17 동안 Oracle baseline 의심 안 함. Oracle z_max **distribution** 안 보고 aggregate metric만 봄. **episode-level fail mode 분리**가 saddle 진단의 핵심.

## 다음 세션 19 권고

- 옵션 A (권장): Oracle `--lift_dz 0.15` 검증 (30분). 현실적 ceiling ~41% (mid-lift 26 회복). no-lift 49는 별도 진단 필요
- 옵션 A2: no-lift 49 메커니즘 진단 (force/IK/contact)
- 옵션 B: success metric 5cm로 시연 정의 명확화
- 옵션 C: Path C 유지 (현재 가용 — BC s8 v1 + Oracle 백업, 5/12 sim2real)

## env cfg 최종 상태

baseline 원복 (5s / y±20cm) — A1/A2는 효과 없거나 역효과, A3는 env 무영향 (측정 인자만).

---

# 세션 19 (2026-05-10) 결과 — success_z=0.05 Demos 재수집 + BC/PPO 재학습

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. env cfg 변경 | **skip** (옵션 B 선택, CLI arg만 사용) |
| 2. Oracle 재검증 | skip (s18 A3 = 44% 재사용) |
| 3. Demos 200 재수집 (success_z=0.05) | ✓ `tasks/demos_session19_z5cm.pt` (219 demos) |
| 4. BC 학습 (30 epoch, lr 1e-3) | ✓ `tasks/bc_actor_session19_z5cm.pt` |
| 5. BC 평가 (100ep) | ✓ `tasks/diagnose_bc_session19_z5cm.csv` (40% @ z=0.05) |
| 6. PPO 1500 iter (BC warmstart) | ✓ `~/isaac_so_arm101/logs/rsl_rl/lift/2026-05-10_15-44-36/` |
| 7. PPO 5 checkpoint 평가 | ✓ `tasks/diagnose_ppo_session19_iter{0,300,600,1000,1499}.csv` |
| 8. report + commit | ✓ |

## 핵심 결과

| 정책 | @ z=0.05 | @ z=0.10 (post-hoc) |
|---|---|---|
| Oracle s18 A3 | 44% | 15% |
| **BC v19** | **40%** | 11% |
| **PPO v19 iter300 (best)** | **43%** | 15% |
| PPO v19 iter1499 (final) | 40% | 12% |

## Honest Caveat

z=0.05 기준 ≥40% (Path A 게이트 통과)이지만 z=0.10 기준 historical saddle (~14%) 그대로. **향상의 원천 = threshold 완화** (Mode A 31% mid-lift 자연스럽게 success 분류). 실제 정책 capability 개선 미미. PPO learning gain 사실상 0 (no-lift fail 50/60 그대로).

## 시연 path 결정 (사용자 결정 사항)

- **Path A** (z=0.05 채택): PPO v19 iter300 + BC v19, "5cm lift = success" framing → sim2real 5/12
- **Path C** (z=0.10 유지): BC v8 + Oracle 백업, "saddle = task threshold 문제" 메시지 → sim2real 5/12

## 다음 세션 20 후보

- Mode B (no-lift 50/60) 메커니즘 진단 (gripper-cube force / kinematic / oracle lift_dz 0.15)
- 또는 sim2real 즉시 진입 (path 결정 후)

---

# 세션 20 (2026-05-10) 결과 — Mode B (no-lift 48%) 환경 한계 확정

## 작업 결과

| 작업 | 상태 |
|---|---|
| 1. Mode B 진단 (csv 분석) | ✓ Oracle Mode B = grasp 실패 (ee_at_close 4.91cm) |
| 2. 가설 1 (lift_dz=0.15) | ✓ FAIL (44%/15% 동일) |
| 3. 가설 2 (friction) | skip (이미 1.5 적용 중) |
| 4. 가설 3 (close timing tighter, H3+H4) | ✓ FAIL (Mode B 48 동일) |
| 5. BC/PPO 재학습 | skip (fix 없음) |
| 6. report + commit | ✓ |

## 핵심 결과

| variant | @z=0.05 | @z=0.10 | Mode B | ee_at_close |
|---|---|---|---|---|
| baseline | 44% | 15% | 48 | 4.91cm |
| H1 lift_dz=0.15 | 44% | 15% | 48 | 4.91cm |
| H3 reach=0.025 | 44% | 15% | 48 | 3.56cm |
| H4 reach=0.015 close=16 | 42% | 13% | 49 | 2.76cm |

→ 모든 Oracle 파라미터 변경에서 Mode B 48±1. ee_at_close 4.91→2.76cm로 가까워져도 회복 안 됨.

## 결론

**Mode B = gripper-cube physical grasp 실패** (geometry/contact dynamics). 환경 USD 수준 변경 필요, Oracle 파라미터 무관.

**9세션 ablation 종합**: RL 학습 알고리즘 정상, 환경 자체가 ceiling. Path C 확정.

## 시연 path 최종 권고

**Path C** — BC v19 + Oracle 백업, 5/12 sim2real 즉시 진입. RL 트랙 종료.

---

# 세션 21 (2026-05-10) 결과 — Mode B USD-수준 ablation FAIL, 환경 ceiling 정직 확정

## 작업 결과

| 작업 | 상태 |
|---|---|
| 0. hang process kill, USD baseline 복구 | ✓ |
| Wrapper script `tasks/usd_convert_session21.py` 작성 | ✓ |
| 1. 옵션 3 (collision_from_visuals) 10ep | ✓ FAIL (20% @z=0.05, Mode B 7/10) |
| 2. 옵션 2 (convex_decomposition) 100ep | ✓ null effect (44%/14%, Mode B 48 동일) |
| 3. 옵션 1 (URDF explicit collision) 100ep | ✓ FAIL (24%/8%, Mode B 71/100) |
| 4. BC/PPO 재학습 | skip (fix 없음) |
| 5. report + commit | ✓ |
| 6. URDF + USD baseline 원복 | ✓ |

## 핵심 결과

| 옵션 | @z=0.10 | Mode B | 평가 |
|---|---|---|---|
| baseline | 15% | 48 | ref |
| Opt 3 collision_from_visuals | 10% (10ep) | 7/10 | 악화 |
| Opt 2 convex_decomposition | 14% | 48 | null |
| Opt 1 URDF explicit collision | **8%** | **71** | 큰 악화 |

→ 모든 USD collision 변경 fail 또는 악화. **Counter-intuitive**: 더 정확한 collision = 더 안 좋음. baseline의 IsaacLab "collision 누락 자동 처리"에 의존.

## 발견

1. `config.yaml`은 UrdfConverter 출력 (입력 아님). sed 편집 무효.
2. `convert_mimic_joints_to_normal_joints` default `false` → s20 working은 `true` 빌드됨. 무지한 재변환 시 gripper 부정합 → hang.
3. URDF에 5개 gripper 링크 collision 누락 발견 (이번 세션 진단). 그러나 추가가 fix가 아닌 **regression**.

## 12세션 ablation 마침표 (s9~s21)

RL 알고리즘 / task spec / Oracle params / USD collision 모두 ablation 완료.
- Mode A 31% mid-lift: success_z=0.05로 회복 (s19)
- Mode B 48% no-lift: 환경 USD ceiling 입증 (s20+s21)
- 진짜 fix path = gripper geometry 재설계 (이번 시연 범위 외)

## 시연 path 최종 — Path C 확정

- BC v19 (`tasks/bc_actor_session19_z5cm.pt`) — 40% @z=0.05, 메인
- PPO v19 iter300 (`logs/rsl_rl/lift/2026-05-10_15-44-36/model_300.pt`) — 43%, 보조
- Oracle (state machine) — 44%, 백업
- sim2real 시작: 5/12 (월)
- 발표 메시지: "12세션 ablation으로 환경 한계 정직 입증, Path C 시연"

---

# Session 22 — Oracle 코드 리뷰 + Sim2Real Interface Contract

작성일: 2026-05-11
목적: 5/14(수) deploy 결정 시점 전, Oracle 코드의 sim 의존성과 perception 인터페이스를 명확히 분리할 spec 작성. RL/Oracle 한 팔 sim2real 트랙 (대안, 메인 IL과 별개).

## 입력 파일 경로 정정 (확인 필요)

- task description: `tasks/oracle_policy.py` → **존재하지 않음**
- 실제 위치: `/home/j-k14d101/jabis_sim/sim2real/oracle/oracle_policy.py` (342 lines, 2026-05-08 작성)
- 호출 예시: `tasks/diagnose_h2_force.py:120-121` 에서 `sys.path.insert(0, "/home/j-k14d101/jabis_sim/sim2real/oracle")` 후 import
- 같은 폴더 보조 파일: `collect_demos.py`, `validate_oracle.py`, `smoke_test.py`, `train_bc.py`
- → **이 파일을 리뷰 대상으로 진행할지 사용자 확인 필요**

## 체크리스트

- [ ] **Step 0**: 입력 파일 경로 확정 (사용자 확인)
- [ ] **Step 1.a**: `oracle_policy.py` 전체 읽기 — 5-state machine
- [ ] **Step 1.b**: IK 호출부 (`DifferentialIKController` 또는 대체) 위치
- [ ] **Step 1.c**: gripper 제어부 (`left_proximal = -0.6` 등) 하드코딩 위치
- [ ] **Step 1.d**: target 계산부 (cube pose → IK target frame 변환)
- [ ] **Step 1.e**: 보조 — `collect_demos.py`에서 cube 좌표 source 확인 (Q1, Q5 답에 필요)
- [ ] **Step 2**: 5가지 질문 답변 (질문당 5~10줄, 라인 번호 인용 필수, 추측 금지)
  - [ ] Q1. 좌표 frame 경계 (world / base / camera)
  - [ ] Q2. Gripper 명령 추상화 (sim USD vs Feetech 매핑)
  - [ ] Q3. 실패/타임아웃 로직 (reset 의존 여부, 실물 fallback)
  - [ ] Q4. Goal pose 처리 (하드코딩 vs 외부 입력)
  - [ ] Q5. 추론 주기 가정 (stream vs trigger)
- [ ] **Step 3**: `tasks/oracle_interface_spec.md` 작성 (1페이지 분량)
  - [ ] Oracle 입력 schema (target_object_pose, goal_pose, …)
  - [ ] Oracle 출력 schema (joint_target, gripper_cmd, …)
  - [ ] Perception 책임 범위 (D456, YOLO, deprojection, extrinsic)
  - [ ] 미해결 이슈 ≥ 2개
- [ ] **Step 4 (선택)**: sim 의존성 / 실물 비호환 코드 위치 부록

## 제약 (재확인)

- 코드 수정 금지 (oracle_policy.py 건드리지 않음, 리뷰/문서 only)
- 추측 금지 (코드만으로 판단 불가 → "사용자 확인 필요"로 명시)
- 간결성: spec 1페이지, 답변 질문당 5~10줄
- 완료 기준: 미해결 이슈 ≥ 2개 (없으면 리뷰 부실)

## 진행 상황 (2026-05-11 업데이트)

- [x] **Step 0**: 입력 파일 경로 확정 → `/home/j-k14d101/jabis_sim/sim2real/oracle/oracle_policy.py`
- [x] **Step 1.a**: oracle_policy.py 전체 읽기 (342L)
- [x] **Step 1.b**: IK = `DifferentialIKController(ik_method="dls")` (oracle_policy.py:132-137)
- [x] **Step 1.c**: gripper hardcoding은 oracle 외부 — `joint_pos_env_cfg.py:48-52` `BinaryJointPositionActionCfg`
- [x] **Step 1.d**: target = base frame (oracle_policy.py:286-292)
- [x] **Step 1.e**: collect_demos.py 확인 — sim에서 `object_asset.data.root_pos_w` ground-truth 사용
- [x] **Step 2**: 5가지 질문 답변 (아래)
- [x] **Step 3**: `tasks/oracle_interface_spec.md` 작성 완료
- [x] **Step 4**: 부록(sim 의존성 표) spec에 포함

## Step 2 답변 — 5가지 질문

### Q1. 좌표 frame 경계
- 5-state target은 **robot base frame**. `compute_target_pos_b` (157-188)가 모두 base frame `_b` 변수로 동작.
- DifferentialIKController도 base frame 입력: `self._ik.set_command(command=target_capped, ee_quat=ee_quat_b)` (318), `self._ik.compute(ee_pos_b, ee_quat_b, jac, ...)` (319). Jacobian은 `base_rot_inv` 회전으로 base 변환 (312-314).
- sim에서 cube 좌표: `cube_pos_w = self.object_asset.data.root_pos_w` (290) → `cube_pos_b = cube_pos_w − root_pose_w[:, 0:3]` (292). **identity base rotation 가정** (291 주석). 실물에서는 base가 회전하면 깨짐 — base 고정이라 무시 가능, 사용자 확인 필요.

### Q2. Gripper 명령 추상화
- oracle_policy.py에 `left_proximal=-0.6` 같은 USD 직접값 **없음**. oracle은 binary `+1.0`(open)/`-1.0`(close)만 출력 (253-258, `gripper_open_mask = states < CLOSE`).
- 실제 USD joint 매핑은 task env config에 있음: `src/isaac_so_arm101/tasks/lift/joint_pos_env_cfg.py:48-52` — `open_command_expr={"left_proximal": 0.0, "right_proximal": 0.0}`, `close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6}`.
- → 실물 매핑 가능한 추상화는 이미 있음 (oracle 측은 binary). 실물에서는 `BinaryJointPositionActionCfg` 자리에 Feetech `goal_position` step 값 매핑 테이블만 끼우면 됨.

### Q3. 실패/타임아웃 로직
- **명시적 timeout 없음**. 전이 조건: APPROACH→DESCEND `dist3 < 0.02m` (210), DESCEND→CLOSE `dz_abs < 0.005m` (211), CLOSE→LIFT `close_step >= 8` (212), LIFT→MOVE_TO_GOAL `cube_z >= success_z` (213).
- APPROACH에서 cube를 놓치면 sim에서는 ground-truth라 불가능. 실물에서 stale pose면 oracle은 옛 위치로 무한 수렴 시도.
- IK 실패: `dls`는 항상 답을 주지만 saturate. `max_ee_step=0.02m` cap (305)이 jump 완화. action `clamp(-1, +1)` (252)로 강제.
- reset은 외부 호출: `oracle.reset(env_indices)` (139). collect_demos.py:208에서 env가 done 처리 시 호출. **실물에는 동등 mechanism 없음** → 미해결 이슈 #2로 spec에 명시.

### Q4. Goal pose 처리
- **외부 입력 가능 구조**: `target_pos_b_provider: Optional[Callable[[], torch.Tensor]]` (92, 106). `compute()` 매 step에서 pull (295-296).
- 미공급 시 fallback: `target[m_goal] = self.lift_target_pos_b[m_goal]` (184) — 즉 lift 위치 hold.
- collect_demos.py:99-101에서는 `command_manager.get_command("object_pose")[:, 0:3]`로 sim 의 `UniformPoseCommand`에서 가져옴. 실물에서는 같은 callback 자리에 쓰레기통 좌표를 상수 반환하는 lambda 끼우면 됨 (1줄).

### Q5. 추론 주기 가정
- `compute()`는 매 step `cube_pos_w`를 새로 읽음 (290) — **매 control step 갱신 가정**.
- collect_demos.py 메인 루프 (145-178): `action = oracle.compute()` → `env.step(action)` 동기 — env step rate (= control decimation 후) 가 oracle 추론 주기.
- 의미: perception이 **stream으로 (>= control rate) 보내야** 안전. trigger 1회로는 DESCEND~LIFT 동안 cube가 밀리면 못 따라감. perception rate < control rate면 직전 프레임 hold로 약하게 동작 가능 (단 stale 처리 별도 필요 — 미해결 이슈 #2).

## 산출물

- `tasks/oracle_interface_spec.md` (1페이지+부록, 미해결 이슈 6개)
- 본 todo.md Session 22 섹션 (체크리스트 + 답변)

## 다음 단계

- 사용자 확인 후 `/codex:review`로 spec 검토 (글로벌 지침)
- 검토 의견 반영 → 5/12(월) sim2real 시작 입력으로 사용

---

## 5/11 오후: D456 Perception 설계

작성일: 2026-05-11 오후
입력: `tasks/oracle_interface_spec.md` (오전 산출물, P0 6 / P1 5 / P2 4)
산출: `tasks/d456_perception_design.md` (2~3 페이지)

- [x] **Step 0**: spec sanity check 완료 (보강 사항 아래 기록)
- [x] **Step 1**: plan 사용자 확인 완료 ("진행")
- [x] **Step 2.0**: spec P0 #1, #2, #3을 d456 §1로 이동 + spec에 이동 표시 (stub 3줄로 축소)
- [x] **Step 2.1**: §1 Calibration 3종 (extrinsic / 영점 / EE link) — 방법/절차/합격 기준 수치
- [x] **Step 2.2**: §2 Perception Pipeline (RGBD → YOLO/HSV → deproject → base 변환)
- [x] **Step 2.3**: §3 Interface 표 — spec Oracle 입력 5종 매칭 (perception 책임 1종, 나머지 4종 = 로봇 컨트롤러 책임)
- [x] **Step 2.4**: §4 검증 절차 (10 위치 × 100 frame, mean ≤5mm/p95 ≤10mm)
- [x] **Step 2.5**: §5 perception 고유 미해결 이슈 5개 (≥3 충족)
- [x] **부록**: 가정/사용자 확인 필요 6개 (A1~A6) 명시

---

## 5/11 저녁: D456 Perception 설계 수정 (cube → 임의 물체 3종)

작성일: 2026-05-11 저녁
배경: 5/10 결정 — 시연 물체 3종 (phone 확정 / cube / 쓰레기 잠정 cube). Oracle 아키텍처 (c) 채택 — Oracle은 grip_pose만 받아 pick & lift, 물체별 grip pose는 perception/상위 로직 책임.
대상: `tasks/d456_perception_design.md` (수정). spec 파일은 형태 무관이라 미수정.

### Step 0 — 영향 분석 (cube 단일 가정 의존 위치)

| # | 위치 | 현재 | 변경 방향 |
|---|---|---|---|
| 1 | 서론 (line 3) | "SO-ARM101 한 팔 cube-lift" | "임의 물체 lift (phone/cube/trash)" + "5/10 결정 반영" 마크 |
| 2 | §1.1 합격 기준 (line 38) | "cube가 50 mm 폭 가정" 한 줄 | 삭제 또는 "물체별 §4에서 분리 측정" 일반화 |
| 3 | §1.2 / §1.3 | 형태 무관 | **변경 없음** (확인만) |
| 4 | §2 도식 (line 73) | `cube_pos_b` 단일 출력 | `D456 → YOLO multi-class → deproject → object_class 분류 → grip pose lookup → base 변환 → grip_pose_b` |
| 5 | §2 표 (b) (line 80) | YOLO **또는** HSV | YOLO multi-class only (HSV **폐기**) |
| 6 | §2 표 (d)(e) (lines 82-83) | `cube_pos_b` | `object_pos_b` + `object_class` + `grip_pose_b` |
| 7 | §2 결정 한 줄 (line 86) | HSV vs YOLO 선택 | YOLO 다중 클래스 학습/재활용 결정 → §5로 이동 |
| 8 | §2 lookup 신규 | (없음) | grip pose lookup table 정의: cube=측면, trash=cube 동일, phone=top-down |
| 9 | §3 표 (line 96) | cube_pos_b 1종 매칭 | grip_pose_b 1종 + object_class metadata 매칭 |
| 10 | §3 schema (line 110) | `{cube_pos_b, stamp, valid, depth_quality}` | `{object_class, object_pos_b, grip_pose_b{position,orientation}, confidence, timestamp}` |
| 11 | §3 책임 분리 문장 (line 102) | "cube_pos_b 1종" | "grip_pose_b 1종 (lookup 결과)" |
| 12 | §4 목적 (line 121) | cube 위치 오차 | 물체별 위치 오차 + grip pose 정확도 |
| 13 | §4 절차 (lines 127-131) | "큐브 1개, 10 위치 grid" | phone N 위치 / cube 10 위치 / trash (cube 통일 시 별도 측정 X) |
| 14 | §4 스크립트 (line 140) | `eval_static_cube.py` | `eval_static_objects.py` (또는 `--object` arg) |
| 15 | §4 추가 검증 (없음) | (없음) | grip pose lookup 정확도 — 물체별 N회 sim/실물 검증 |
| 16 | §5 #2 (line 162) | "Cube color/shape/size" 단일 | **제거** → 부록 A3a/A3b로 분리. §5에 신규 4개 추가 (아래) |
| 17 | §5 #3 (line 166) | HSV/YOLO 비교 | YOLO 중심으로 단순화 |
| 18 | §5 #4 detection fail | 단일 class | 다중 class fail 구분 (어느 class) |
| 19 | §5 신규 (4개) | (없음) | YOLO 다중 클래스 모델, cube↔쓰레기 형태 통일, phone grip 정밀도, **place-with-orientation Oracle 확장** |
| 20 | 부록 A3 (line 188) | 큐브 50mm 단색 정사각형 | **A3a** (cube/쓰레기) + **A3b** (phone, 김현준 측정 가능) |
| 21 | 부록 A6 (line 191) | cube workspace | 모든 물체 workspace |

### 체크리스트

- [x] **Step 0**: 영향 분석 (위 매트릭스 21개 항목)
- [x] **Step 1**: plan 사용자 확인 ("진행", phone=side-grip 결정, place issue=미해결 명시)
- [x] **Step 2.1**: §1 상단 "변경 없음" 명시 + §1.1 cube 50mm 참고 라인 일반화
- [x] **Step 2.2**: §2 Pipeline 수정 (HSV 폐기, YOLO multi-class, 7단계 표 + §2.1 lookup table 신규)
- [x] **Step 2.3**: §3 Interface schema 변경 (`grip_pose_b` + `object_class` + 필드별 책임)
- [x] **Step 2.4**: §4 검증 절차 §4.1 위치 / §4.2 orientation / §4.3 grip 성공률 3분할, 물체별 N + 합격 기준
- [x] **Step 2.5**: §5 미해결 #2 제거, 신규 4개 (YOLO 모델, 형태 통일, phone grip 정밀도, place-with-orientation), 기존 #3·#4 다중 class 반영. 총 8개.
- [x] **Step 2.6**: 부록 A3 → A3a (cube/쓰레기) + A3b (phone, 김현준 측정) 분리, A6 일반화
- [x] **Step 2.7**: 변경 부분 모두 "5/10 결정 반영" 마크 (서론, §1, §2, §2.1, §3, §4, §5, A3a, A3b)

### 제약 (재확인)

- spec 파일 (`oracle_interface_spec.md`) 건드리지 말 것 (형태 무관)
- oracle_policy.py 코드 수정 금지
- 추측 금지 (phone 사양, YOLO 재활용 가능성 등 → "사용자 확인 필요")
- §4 합격 기준 수치 유지 (mean ≤5mm / p95 ≤10mm / max ≤15mm / 성공률 ≥95%) — 단 물체별 분리 측정

### 진행 방식

1. 사용자 plan 확인 → Step 2 진입
2. Step 2 끝나면 결과 보고 + codex review 옵션 제시

### Step 0 보강 내역 (spec 변경)

`oracle_interface_spec.md` P0 섹션 헤더 아래 2줄 추가:
- 검증 위치 라벨 정의: `[HW]` / `[Code]` / `[HW+Code]`
- P0 의존 순서: **#2 (영점) → #3 (EE link) → #1 (extrinsic)**, #4·#5·#6은 독립

각 P0 항목 prefix에 라벨 부착:
- #1 Extrinsic `[HW]`, #2 영점 `[HW]`, #3 EE link `[HW+Code]`,
  #4 Gripper LUT `[HW]`, #5 Timeout `[Code]`, #6 E-stop `[HW+Code]`

### 제약 (재확인)

- spec P0 #1~#3은 d456_perception_design.md §1로 이동 (중복 제거, spec에는 "→ §1.x로 이동" 표시)
- 합격 기준은 수치로 (cm/mm/ms/% 단위)
- 추측 금지 (D456/SO-ARM101 사양 불명확하면 "사용자 확인 필요")
- 코드 수정 금지

### 진행 방식

1. 사용자 plan 확인 → Step 2 진입
2. Step 2 끝나면 결과 보고 (codex review 옵션 제시)

---

## 5/12 사전 준비 (5/11 밤)

목표: 내일 D456 + SO-ARM101 연결되면 즉시 테스트 가능하도록 ChArUco 검출 + 오른팔 π rotation wrapper를 미리 작성. 하드웨어 의존성 0.

좌표계 결정 (spec / 메모리 #23 반영):
- world 원점 = 책상 가운데
- 오른팔 base frame은 sim 대비 z축 π 회전 (rotation = π)
- 변환: position `(x,y,z) → (-x,-y,z)`, quaternion 은 `(0,0,1,0)` 을 sim quat에 **좌측에서** 곱 (parent-frame 회전 → `R_right = Rz(π) · R_sim`)

산출물 디렉토리:
```
~/jabis_sim/sim2real/perception/   (신규 디렉토리)
  ├── charuco_detector.py
  ├── test_charuco_detector.py
  └── README.md
~/jabis_sim/sim2real/oracle/       (기존 디렉토리)
  ├── sim_to_real_right_arm.py     (신규)
  └── test_sim_to_real.py          (신규)
```

### Codex 검토 반영 (5/11 plan revision)

5가지 지적 모두 반영:
1. **`board_to_world_transform` 제거** — identity stub은 5/12 실측 시 silent 오류 위험. 함수 자체 빼고 README에만 contract 문서화. (5/12 실측 후 별도 작성)
2. **OpenCV 4.7+ API**: `cv2.aruco.CharucoBoard((cols, rows), squareLength, markerLength, dictionary)` 형태로 사용 (4.6 이전 `CharucoBoard_create` 폐기됨). `cv2.aruco.ArucoDetector` 사용.
3. **Tolerance를 pixel reprojection 기반**: 합성 이미지 검출 정확도는 mm 단위가 아닌 **reprojection RMS ≤ 2 px** 로 검증.
4. **sim2real 핵심 테스트 추가**:
   - **nontrivial 회전 (예: x축 90°) × Rz(π) 행렬 합성** vs `sim_to_right_arm_quat` 결과 행렬 비교 → 좌측 곱 입증 (commute 안 되는 케이스라야 의미 있음)
   - 비단위 quaternion 정규화 / 부호 동치 (q == -q) / 영벡터 quat 처리
5. **charuco 테스트 추가**:
   - K shape 오류 (3×3 아닌 입력) → ValueError raise
   - dist shape 오류 → ValueError raise
   - grayscale vs RGB 입력 처리 명시 (BGR 가정 + 명시)
   - 마커 0개 / 임계값 직전 (코너 5개) → None
   - `BoardPose.to_4x4_matrix()` shape (4,4), bottom row `[0,0,0,1]` 검증
6. **opencv-contrib 확인**: `cv2.aruco` namespace 존재 여부 첫 import 시 명시적 체크.

### 체크리스트

- [x] perception/ 디렉토리 생성 (`__init__.py` 미사용 — sys.path.insert 로 sibling import)
- [x] charuco_detector.py 작성 (DICT_4X4_50 / 5×7 / checker 40mm / marker 30mm 가정값 상수화 + 입력 shape 검증)
- [x] test_charuco_detector.py: 11 케이스 (정상 검출 / 검은 / 부분 / 임계값 미만 / grayscale / K-shape / dist-shape / image-shape / None / `to_4x4_matrix()` SE(3) / dataclass fields)
- [x] sim_to_real_right_arm.py 작성 (pos / quat / pose + quaternion_multiply, 비단위 quat 정규화, zero-norm raise)
- [x] test_sim_to_real.py: 17 케이스 (position 6 / quat 5 / nontrivial 회전 행렬 합성 / multiply 3 / pose 2)
- [x] `pytest ~/jabis_sim/sim2real/{perception,oracle}/` **28/28 통과**
- [x] perception/README.md 1페이지 (보드 파라미터 / 사용 예 / `T_camera_board` contract 문서화 / 다음 단계 / 알려진 이슈)
- [x] TODO 주석 4건 표시 (K·dist 실측값 / marker size / dictionary / world extrinsic — 5/12 실측)

### 가정값 (인쇄 PDF 메타정보 검증 후 수정)

- DICTIONARY = `cv2.aruco.DICT_4X4_50`
- CHECKER_ROWS=5, CHECKER_COLS=7, CHECKER_SIZE=0.040 m, MARKER_SIZE=0.030 m
- OpenCV `CharucoBoard((cols, rows), ...)` 인자 순서 = **(cols=7, rows=5)** — Codex 지적 반영 (rows/cols 뒤집힘 주의)

### 제약

- OpenCV 4.7+ ArUco API (`cv2.aruco.ArucoDetector`, `cv2.aruco.CharucoBoard`) — 환경 4.11.0 확인됨
- 외부 의존성: numpy + opencv-python (또는 opencv-contrib-python) — scipy 금지
- 검출 실패 (코너 < 6) = None 반환 (예외 raise 금지)
- 입력 shape 오류 (K가 3×3 아님 등) = `ValueError` raise
- 코드 주석 한국어 OK / docstring 영어
- 모든 `np.allclose` assert 에 atol 명시 (position: 1e-9, quaternion: 1e-9, reprojection RMS: 2.0 px)
- 보드 파라미터 가정값은 상단 상수 + "TODO: PDF 메타정보 검증 후 수정"

### 진행 방식

1. ~~사용자 plan 확인~~ ✅ → ~~`/codex:review` plan 검토~~ ✅ (5가지 반영)
2. ~~Step 2~6 (디렉토리 생성 → 4 파일 → README)~~ ✅ — 28/28 pytest 통과
3. ~~코드 마무리 `/codex:review` 재검토~~ ✅ — RISK 4건 중 deploy 안전성 핵심 적용

### 구현 마무리 검토 결과 (5/11 밤)

Codex 구현 검토에서 RISK 4건 식별, 사용자 결정 "실조 deploy 안전성 우선" 으로 처리:

| 항목 | 분류 | 처리 |
|---|---|---|
| B4+E1: `n_corners_detected` 의미 모호 / `solvePnP` 예외 미흡수 / flag 미명시 / `obj_points`·`img_points` 길이 mismatch 미검증 | 코드 수정 | `n_corners_detected` 를 실제 PnP 사용 점수 (`len(obj_points)`) 로 변경, `try/except cv2.error` 로 예외 → None, `flags=cv2.SOLVEPNP_ITERATIVE` 명시, length mismatch 시 None |
| E2: BGR/RGB 호출부 책임 모호 | docstring + README | `detect_charuco` docstring 에 BGR 가정 / pyrealsense2 stream 설정 책임 명시, README "알려진 이슈" 강화 |
| B1: `quaternion_multiply` 정규화 안 함 — 외부 chain 시 위험 | docstring | docstring 에 "public wrapper 사용 권장 / chain 시 drift" 강조 |
| C1: `test_partial_detection` 약함 (`pose is None` 통과) | 미수정 | 의도적 (가림 마스크 geometry 가 OpenCV 버전마다 fragile). 향후 5/12 실측 데이터로 보강 |

A (수학 정확성), D (코드 스타일) 모두 OK.

### 산출물 위치

- `~/jabis_sim/sim2real/perception/charuco_detector.py`
- `~/jabis_sim/sim2real/perception/test_charuco_detector.py` (11 케이스)
- `~/jabis_sim/sim2real/perception/README.md`
- `~/jabis_sim/sim2real/oracle/sim_to_real_right_arm.py`
- `~/jabis_sim/sim2real/oracle/test_sim_to_real.py` (17 케이스)

### 5/12 deploy 시 수정해야 할 TODO 4건 (코드 내 주석으로 마킹)

1. `charuco_detector.py` 상단 — DICTIONARY/CHECKER/MARKER 가정값 인쇄 PDF 메타정보 검증 후 수정
2. ~~D456 intrinsics (K, dist) 실측값 — pyrealsense2 로 가져와 yaml 저장~~ ✅ `get_d456_intrinsics.py` 작성 (5/12 카메라 연결 후 실행)
3. Marker size 정확값 (현재 30mm 가정 — checker × 0.7~0.8 규약)
4. World frame extrinsic (`T_world_camera`) — 보드 placement 측정 후 별도 함수 작성 (charuco_detector.py 끝의 주석 참조)

### 추가: D456 intrinsics 추출 스크립트 (5/11 밤, 사용자 추가 요청)

산출물: `~/jabis_sim/sim2real/perception/get_d456_intrinsics.py`

- [x] pyrealsense2 RGB(+depth) intrinsics 추출 → JSON 저장
- [x] argparse `--output`, default = 같은 디렉토리 `d456_intrinsics.json`
- [x] depth + depth→color extrinsic 보너스 포함
- [x] 카메라 미연결 / pyrealsense2 미설치 → graceful exit 1 + 명확한 메시지
- [x] README 사용법 추가 + 주의사항 (해상도 / distortion model / row-major)
- [x] 회귀 0건 (28/28 기존 테스트 그대로 통과)

Codex 마무리 검토에서 3건 이슈 식별 + 수정:
1. **D2 (BUG)**: `rs2_extrinsics.rotation` 이 column-major 3x3 인데 row-major 로 저장 중 → 인덱스 변환으로 row-major 출력 (`r[0],r[3],r[6]` / `r[1],r[4],r[7]` / `r[2],r[5],r[8]`)
2. **B3 (BUG)**: depth 가 "보너스" 라고 주석은 달았지만 실제로는 항상 enable_stream 요청 → depth 실패 시 pipeline.start 통째로 실패. color-only fallback 흐름 추가.
3. **A3 (RISK)**: D456 color 가 종종 `inverse_brown_conrady` 모델로 노출. OpenCV `solvePnP` 는 forward Brown-Conrady 가정 → 의미 다름. docstring + README 에 model 필드 검증 + 5/12 실측 시 RMS 검증 안내.
4. **E1 (info)**: 640×480 K/dist 는 그 해상도에만 유효 — README 경고.

A1 (intrinsics 추출 흐름), A2 (extrinsic 방향), B2 (RuntimeError catch), C1 (coeffs 길이 5), C2 (JSON dtype), D1 (finally 흐름) 모두 OK.

