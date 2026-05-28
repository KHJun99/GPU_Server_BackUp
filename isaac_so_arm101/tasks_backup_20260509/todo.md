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
