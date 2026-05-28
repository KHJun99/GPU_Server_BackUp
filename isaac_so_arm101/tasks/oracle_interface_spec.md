# Oracle ↔ Perception Interface Spec (Sim2Real)

작성: 2026-05-11 · 대상: SO-ARM101 한 팔 cube-lift, 5/14 deploy 결정용
근거 코드: `/home/j-k14d101/jabis_sim/sim2real/oracle/oracle_policy.py` (342L), `collect_demos.py`, `src/isaac_so_arm101/tasks/lift/joint_pos_env_cfg.py`

## Oracle 입력 (perception/외부에서 공급)

| 이름 | shape / dtype | frame | 갱신 주기 | 비고 |
|---|---|---|---|---|
| `cube_pos_b` | `(N, 3)` float32 (m) | **robot base** | 매 control step | sim 현재: `object_asset.data.root_pos_w − robot.root_pos_w` (oracle_policy.py:290-292). orientation은 **identity 가정** (rot=(1,0,0,0), 라인 291 주석). 실물에서는 perception이 base frame 변환까지 책임. |
| `goal_pos_b` (선택) | `(N, 3)` float32 (m) | robot base | 매 step pull (callback) | `target_pos_b_provider: Callable[[], Tensor]`로 주입 (oracle_policy.py:92,106,295-296). 미공급 시 lift 위치 hold (185行). 실물에서는 쓰레기통 위치 등을 한 번 받아 상수 콜백으로 감싸도 됨. |
| `ee_pos_b`, `ee_quat_b` | `(N, 3)`, `(N, 4)` | robot base | 매 step | 현재 `robot.data.body_pose_w` + `subtract_frame_transforms` (286-289). 실물에서는 FK from joint encoder. |
| `joint_pos_arm` | `(N, 5)` rad | — | 매 step | 5 arm joint 현재값 (315). 실물 = 서보 인코더. |
| `jacobian` | `(N, 6, 5)` | base frame | 매 step | 현재 `robot.root_physx_view.get_jacobians()` (309-314). 실물에서는 URDF 기반 FK/Jacobian 라이브러리 필요. |

## Oracle 출력

| 이름 | shape / range | 의미 |
|---|---|---|
| `action` | `(N, 6)` float32, `[-1, +1]` clamp | PPO raw action. arm 5 + gripper 1. (oracle_policy.py:259, `to_action`) |
| `action[:, :5]` (arm) | `arm_raw = (joint_target − default_arm) / scale` (250) | 실물 변환: `joint_target = arm_raw * scale + default_arm` (rad). scale=1.5. |
| `action[:, 5]` (gripper) | `+1.0`=open, `-1.0`=close, binary (253-258) | sim에서는 `BinaryJointPositionActionCfg`가 `left_proximal/right_proximal` ±0.6 rad로 매핑 (`joint_pos_env_cfg.py:48-52`). **실물 매핑은 별도 정의 필요.** |
| `info["states"]` | `(N,)` int64 | 0=APPROACH, 1=DESCEND, 2=CLOSE, 3=LIFT, 4=MOVE_TO_GOAL — 시각화/로그용 |

## Perception 책임 범위 (D456)

1. RGBD 캡처 (D456, 30Hz 가정)
2. cube detection (YOLO 또는 색 기반)
3. depth deprojection → camera frame 3D point
4. **camera→robot_base extrinsic 변환** (calibration 필수)
5. 출력 schema:
   ```python
   {"cube_pos_b": np.ndarray(3,), "stamp": float, "valid": bool}
   ```
6. **stale/lost detection**: 직전 프레임에서 detection 실패 시 `valid=False`. Oracle 측은 `valid=False`인 경우 직전 프레임 cube_pos_b를 hold 또는 명시적 abort.

## 미해결 이슈 / 결정 필요 사항

우선순위: **P0** = deploy 차단 (해결 안 하면 동작 불가) · **P1** = 동작 품질/안정성 · **P2** = 추가 검증
카테고리: `[Cal]` calibration · `[Map]` sim↔실물 매핑 · `[Rob]` robustness · `[Saf]` safety · `[Tun]` 튜닝 · `[Iface]` 인터페이스 결정

### P0 — deploy 전 반드시 해결

검증 위치: `[HW]` 실물 하드웨어 필요 · `[Code]` oracle/URDF 검토만으로 가능 · `[HW+Code]` 둘 다 필요
의존 순서: **#2 (영점) → #3 (EE link) → #1 (extrinsic)** (영점 안 맞으면 EE link 물리 측정 무의미, EE link 미확정이면 hand-eye 의미 없음). #4·#5·#6은 독립적으로 병렬 진행 가능.

1. `[HW]` **`[Cal]` Extrinsic D456 ↔ robot_base** — → **`d456_perception_design.md §1.1`로 이동.** 본 spec에서는 "perception이 base frame으로 변환해 cube_pos_b를 공급" 인터페이스만 보장.
2. `[HW]` **`[Cal]` 서보 영점 ↔ sim `default_arm` 일치** — → **`d456_perception_design.md §1.2`로 이동.** 핵심 코드 위치: `arm_raw = (joint_target − default_arm) / scale` (oracle_policy.py:250), `default_arm` 정의 (120).
3. `[HW+Code]` **`[Cal]` EE link `gripper_link` 정의 일치** — → **`d456_perception_design.md §1.3`로 이동.** 핵심 코드 위치: `ee_body_name="gripper_link"` (107).
4. `[HW]` **`[Map]` Gripper binary ±1.0 → Feetech step** — oracle은 binary `±1.0`만 출력 (253-258). sim의 `BinaryJointPositionActionCfg` (open `left_proximal=0.0`, close `-0.6` rad, `joint_pos_env_cfg.py:48-52`)을 실물 Feetech `goal_position` (0~4095 step)으로 매핑할 LUT 부재. 결정: 실물 open/close 자세 측정.
5. `[Code]` **`[Rob]` 실패/타임아웃 fallback (최대 risk)** — oracle에 timeout/abort **전무**. 전이 조건은 거리 임계 (210-213) 뿐, perception이 stale pose 보내면 옛 위치로 무한 수렴. sim에서는 env가 reset (collect_demos.py:208), **실물에는 동등 mechanism 없음**. 결정: (a) state별 `max_step` timeout, (b) perception `valid=False` 가 K 프레임 연속이면 home 복귀.
6. `[HW+Code]` **`[Saf]` E-stop / 충돌 / 한계각 감지 부재** — oracle은 action `clamp(-1, +1)` (252) 외 안전 layer 없음. 실물에서 사람·테이블 충돌, 서보 한계각 초과 시 abort 부재. 결정: 외부 safety supervisor (ROS topic, soft 한계각, 또는 hardware E-stop) 별도 구현.

### P1 — 동작 품질/안정성

7. **`[Rob]` IK saturation 처리** — `ik_method="dls"` (135)는 항상 답을 주나 singularity 근처 jump 가능. `max_ee_step=0.02m` cap (305) + arm clamp가 완화. saturation 비율은 collect_demos.py:153에서 통계만. 결정: saturation rate가 임계 초과 시 안전정지할지.
8. **`[Tun]` sim-튠 hyperparameter 재튠 필요성** — `reach_above_dz=0.10`, `descend_dz=0.005`, `lift_dz=0.10`, `reach_dist=0.02`, `descend_z_dist=0.005`, `success_z=0.10`, `max_ee_step=0.02`, `close_steps=8`, `scale=1.5` (oracle_policy.py:54-66) 전부 sim 큐브·그리퍼·제어 주기에 맞춘 값. 실물 차이 시 재튠.
9. **`[Iface]` Goal callback 갱신 주기** — `target_pos_b_provider`가 매 step pull (296). 실물에서 쓰레기통 위치를 1회 측정 후 상수 lambda로 감쌀지, 매 프레임 perception 갱신할지 결정. 1회면 단순 (perception trigger 1회).
10. **`[Iface]` Control rate 명시** — oracle은 매 step compute, sim의 env step rate (decimation 후)가 곧 추론 주기. 실물 control loop rate가 다르면 IK delta·`max_ee_step` 이 의도와 다른 속도가 됨. 결정: sim env step rate 측정 후 실물 loop rate 통일 (Hz 명시).
11. **`[Iface]` Transport / IPC 형태** — perception ↔ oracle 통신 채널 (ROS2 topic / shared memory / ZMQ / pipe) 미정. 결정: 팀 표준에 맞춰 schema (이름/타입/timestamp 포함) 확정.

### P2 — 추가 검증

12. **`[Iface]` Cube orientation 무시 가정** — 현 oracle은 cube position만 사용 (quat 안 씀). 잡기 자세가 cube 회전에 무관해야 성립. 사용자 확인 필요 (큐브가 정사각형에 가까우면 OK, 길쭉하면 회전 정렬 필요).
13. **`[Cal]` Base orientation identity 가정** — `cube_pos_b = cube_pos_w − root_pose_w[:, 0:3]` (292) 는 base rotation을 무시. 주석에 "Identity base rotation per SO_ARM101_CFG (rot=(1,0,0,0))" (291). 실물 base가 world frame과 회전 없이 align됐는지 확인 필요 (보통 기둥 mount면 OK).
14. **`[Rob]` 첫 프레임 valid 보장** — `reset()` (139-152) 직후 첫 `compute()`가 cube_pos_b 수신을 가정. perception 부팅 직후 N프레임은 invalid일 가능성. 결정: 첫-프레임 valid streak K 충족까지 oracle 시작 대기.
15. **`[Rob]` Initial joint state 가정** — oracle은 reset 후 `default_arm` 자세 가정 없이 동작하지만, 임의 자세에서 시작 시 첫 step IK 답이 멀어 `max_ee_step` cap 동작이 여러 step 누적됨. 실물에서 항상 home 자세에서 시작하는지 확인.

---

## 부록 — sim 의존성 / 실물 비호환 코드 위치

| 위치 | 코드 | 실물 대체 |
|---|---|---|
| oracle_policy.py:132 | `from isaaclab.controllers import DifferentialIKController` | URDF/PyKDL/pinocchio 등 외부 IK 라이브러리 |
| oracle_policy.py:276-280 | `from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms` | scipy/numpy/transforms3d 자체 구현 |
| oracle_policy.py:284-292 | `robot.data.body_pose_w`, `object_asset.data.root_pos_w` (sim ground-truth) | FK from 인코더 + perception |
| oracle_policy.py:309-314 | `robot.root_physx_view.get_jacobians()` (PhysX) | URDF 기반 Jacobian (pinocchio 등) |
| collect_demos.py:208 | `oracle.reset(done_indices)` (env 자동 reset) | 실물에서는 사람 개입 또는 home 복귀 routine |
| joint_pos_env_cfg.py:48-52 | `close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6}` | Feetech 서보 step 값 매핑 테이블 (별도 measure 필요) |

## 추론 주기 결론

`compute()`는 **매 control step에서 cube_pos_b를 새로 읽음** → perception은 **stream(>=control rate)**으로 보내야 함. 한 번 trigger로는 불충분 (DESCEND~LIFT 동안 cube가 살짝 밀려도 따라가지 못함). 다만 perception rate가 control rate보다 낮아도 직전 프레임 hold로 동작은 가능 (단, 위 미해결 이슈 #2의 stale 처리가 전제).
