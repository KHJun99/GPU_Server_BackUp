# Phase 2 — Sim Inspection + Obs Builder

Phase 1 완료 (clip_actions=1.0 발견, raw actor output huge magnitude 정상).
Phase 2 는 sim obs 32 dim 을 실물 센서값으로 재구성. 단 ground truth 확보가 우선.

## Step 1 — Sim Inspection  (먼저, STOP 지점 있음)

### Deliverables
- `~/jabis_sim_v2/sim2real/inspect_sim.py`
- 실행 결과 `~/jabis_sim_v2/sim2real/inspection_raw.txt`
- 정리본 `~/jabis_sim_v2/sim2real/inspection_results.md`

### inspect_sim.py 가 출력해야 하는 ground truth
1. `robot.data.joint_names` 10 개 ordering + default_joint_pos 값
2. `arm_action` term 의 joint_names / joint_ids (sim 내부 ordering 기준)
3. `gripper_action` term 의 joint_names / joint_ids / open_command / close_command
4. `env.scene.env_origins[0]`, `robot.data.root_pos_w[0]`, `cube.data.root_pos_w[0]`
   → cube_pos obs 의 frame 확정 (env_origins == robot_root?)
5. `env.observation_manager.compute()['policy']` shape → (1, 32)
6. `env.action_manager.total_action_dim` → 6
7. step_dt (joint_vel finite diff 시 필요): `cfg.sim.dt * cfg.decimation`

### 실행
```
cd ~/jabis_sim_v2/sim2real
/home/j-k14d101/.conda/envs/jabis-sim/bin/python inspect_sim.py 2>&1 \
  | awk '/JABIS_INSPECT_BEGIN/{flag=1} flag; /JABIS_INSPECT_END/{flag=0}' \
  | tee inspection_raw.txt
```

### STOP — 사용자 보고
inspection_results.md 작성 후 다음 4 가지 요약 보고하고 멈춤:
- joint_names 10 개 ordering
- arm_action 5 joint 이름 + index (in joint_names)
- env_origins vs robot_root_pos_w 관계 (cube_pos frame)
- obs 32 / action 6 dim 검증

사용자 승인 전 Step 2 시작 금지.

### 막히면 (STOP)
- Isaac Sim 부팅 실패
- joint count != 10
- env_origins != robot_root (frame 가설 깨짐)
- obs shape != (1, 32) 또는 action_dim != 6
→ 환경 / conda 절대 변경하지 말고 raw 에러 그대로 보고

## Step 2 — obs_builder.py  (Step 1 STOP 통과 후)

### Deliverable
`~/jabis_sim_v2/sim2real/obs_builder.py` — 실물 센서값 → sim obs 32 dim.

### 좌표 변환 (right-arm base @ world (0.55, 0, 0.12), rotation π about z)
```
cube_x_sim = -(cube_x_world - 0.55)
cube_y_sim = -(cube_y_world - 0.0)
cube_z_sim =   cube_z_world - 0.12
```
(env_origins == robot_root 검증된 경우 한정)

### ObservationBuilder 인터페이스
- `update_joint_pos(arm_servo_rad: np.ndarray[5], gripper_servo_value: float)`
- `update_cube_pos(cube_world_xyz: np.ndarray[3])`
- `update_last_action(action_6: np.ndarray[6])`  ← clip 후 값 (policy.get_action 반환값)
- `build() -> np.ndarray[32]`

joint_pos_rel: current 10 - default 10 (default 는 inspection 결과 활용)
joint_vel: finite diff (prev joint_pos, dt = inspection 의 step_dt)
cube_vel: 일단 zeros(3) 또는 finite diff — 사용자와 의논
gripper 매핑: arm 5 + gripper 1 → sim 10. open/close threshold 적용 (close_command 값은 inspection 으로 확인)

### Dummy 검증 (`__main__`)
- realistic default values 로 update_*
- build() → 32 dim 출력
- policy_inference 연계: obs → get_action → action 출력 (saturated 일 가능성)

## Out of scope (절대 X)
- Phase 3: action_executor.py, main_loop.py
- 실물 servo / D456 코드
- rsl_rl import (nn.Sequential 만)
- policy_inference.py 수정 (이미 검증, np.clip 도 user 추가분 그대로)
- inspection 결과 없이 obs_builder.py 작성
- gripper 매핑 정책 사용자 확인 없이 결정
- conda env 변경 (jabis-sim 고정)

## Codex review
- Step 1 결과 보고 직전 1회
- Step 2 코드 작성 후 1회
