# Real Robot Deployment Plan — SO-ARM101 + PincOpen + Jetson Orin Nano

**상태**: Phase 1 + Phase 2 완료 후 실 배포. 사전 계획 + 인터페이스 정의.

**전체 흐름**:
```
Phase 2 vec env 학습 완료 (success ≥ 70%)
  ↓
Sim2Real gap 분석 (obs 분포 비교)
  ↓
Hand-eye calibration (one-time AprilTag)
  ↓
Jetson Orin Nano 정책 추론 + perception 배포
  ↓
Real cube PnP 시도 (관찰 + 수동 보정)
  ↓
필요 시 few-shot real-data fine-tune
```

---

## 1. Hardware 인터페이스

### 1.1 SO-ARM101 + PincOpen Gripper

- **모터**: Feetech STS3215 × 6 (arm 5 + gripper 1)
- **통신**: USB → Feetech proto (baud 1_000_000)
- **Encoder 매핑** (CLAUDE.md Phase 1 결정사항 2 참고):
  - Encoder 1175 → spread 79mm → sim gripper +0.150 (OPEN)
  - Encoder 2887 → spread 4mm → sim gripper -1.800 (CLOSE)
  - Encoder 2031 → spread 44mm → sim gripper -0.750 (MID)
  - 거의 선형 — 보간으로 sim → real 변환

### 1.2 SO-ARM driver layer 인터페이스 (구현 필요)

```python
# src/khj_rl/hardware/so_arm101.py (Phase 3에서 추가)
class SOArm101Driver:
    """Real SO-ARM101 + PincOpen driver. 정책의 sim action을 그대로 받아
    real motor encoder로 변환 + 송신. obs는 real encoder → sim joint pos."""

    def __init__(self, port="/dev/ttyUSB0", baud=1_000_000):
        ...

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """현재 (joint_pos[5], joint_vel[5])을 sim 단위(rad, rad/s)로 리턴."""

    def get_gripper_state(self) -> float:
        """현재 gripper opening [0..1] (0=close, 1=open)."""

    def set_action(self, arm_delta: np.ndarray, gripper_target: float) -> None:
        """정책 action 적용. arm_delta는 [-1,+1] 정규화 5-D,
        gripper_target은 sim 라디안 (e.g. cfg.robot.gripper_joint_target_low/high)."""

    def emergency_stop(self) -> None:
        """모든 모터 zero-torque + safety latch."""
```

### 1.3 안전장치 (Phase 3 deploy 전 필수)

- **E-stop 물리 스위치**: Feetech 컨트롤러 power line cut
- **Joint soft limit**: driver layer에서 명시적 clamp (soft limit < hard limit margin 10°)
- **Gripper force limit**: STS3215의 토크 limit ~30% (cube 파손 방지)
- **Action delta cap**: 한 step max delta 검증. CLAUDE.md Phase 1 결정사항 2의 `scale=0.05` 그대로
- **Watchdog**: 정책 출력 200ms 이상 없으면 E-stop

---

## 2. Perception 인터페이스

### 2.1 카메라 셋업

- **Top view (eye-to-hand)**: Intel RealSense D456 거치대 위 설치
- **Wrist view (eye-in-hand)**: D405 또는 D456 wrist mount

### 2.2 Cube 검출 — Markerless HSV + depth

```python
# src/khj_rl/perception/cube_detector.py
class HSVCubeDetector:
    def __init__(self, hsv_lower=(45, 80, 80), hsv_upper=(75, 255, 255)):
        # lime cube 기준. cyan은 (85, 80, 80) - (110, 255, 255)
        ...

    def detect(self, rgb, depth, K) -> np.ndarray | None:
        """HSV mask + depth deprojection → cube xyz (camera frame).
        Return None on detection failure (visibility=False)."""
```

### 2.3 좌표계 변환

```
cube (camera frame)
  → cube (world frame) via hand-eye T_cam_world
  → cube (robot base frame) via T_base_world = T_robot_base
```

Robot base frame이 유일한 universal frame (CLAUDE.md 결정사항 2). hand-eye만 캘리브레이션 필요, world frame 별도 X.

---

## 3. Hand-eye Calibration

### 3.1 절차 (one-time)

1. AprilTag (4cm) EE에 부착
2. Robot을 N개 (보통 20-30) 자세로 이동 (자동 grid)
3. 각 자세에서 (T_base_ee, T_cam_tag) 동시 측정
4. AX = XB 풀이로 T_cam_base 추출
5. 결과를 `assets/calib/{top_cam,wrist_cam}_T_base.npy`에 저장
6. **AprilTag 떼고** 작업 시작 (running 중에는 marker 없음)

### 3.2 Calibration 정확도 검증

- 알려진 위치(예: 테이블 모서리)의 cube를 sim과 real에서 비교
- xyz 오차 < 5mm 목표
- 오차 클수록 정책이 학습한 분포에서 벗어남 (obs alignment 가드와 동일 문제)

---

## 4. Sim2Real Obs 분포 비교

### 4.1 검증 도구 (구현 필요)

```python
# scripts/check_sim2real_obs.py
def main():
    # 1. Sim에서 N=100 ep 수집, obs 분포 dump
    sim_obs = collect_sim_obs(n=100)

    # 2. Real robot 동일 행동 (oracle 또는 BC) N=100 ep 수집
    real_obs = collect_real_obs(n=100)

    # 3. 각 obs 차원별 KS test, mean/std 비교
    for i, key in enumerate(env_cfg.obs_keys):
        sim_d = sim_obs[..., i]
        real_d = real_obs[..., i]
        ks_stat, p = stats.ks_2samp(sim_d.ravel(), real_d.ravel())
        print(f"{key}: ks={ks_stat:.3f} p={p:.4f}")
```

### 4.2 가드 — 정책 추론 시 분포 어긋남 catch

기존 `assert_obs_distribution` 재사용. real obs가 sim 분포 (BC mean/std)에서 6σ 안인지 매 step 확인. 어긋나면:
- 경고 로그
- 안 어긋날 때까지 정책 inference 보류 (E-stop과 별도)
- 분포 어긋남 패턴 → 어떤 obs 차원이 sim2real gap 큰지 진단

---

## 5. Jetson Orin Nano 셋업

### 5.1 환경

- **JetPack 6** + Python 3.10
- **PyTorch for Jetson** (NVIDIA 공식 wheel, JetPack 버전 매칭)
- **librealsense + pyrealsense2** (source build)
- **opencv-python** (또는 NVIDIA hardware-accelerated)
- **Pinocchio** 없어도 OK — FK는 정책 내부에 안 들어감 (obs.ee_pose_base는 학습된 정책의 입력일 뿐, 추론에는 robot 직접 측정)

→ `pip install -e ".[dev,jetson]"` (jetson extras에 torch / pyrealsense2 제외 — 위 별도 설치)

### 5.2 정책 추론 latency 목표

- Policy 20Hz = 50ms budget
- Cube detection + obs build: 20-30ms
- Policy inference (BC + actor_mean): < 5ms (작은 MLP)
- Motor send: 5-10ms
- 총 35-50ms — 여유 약함, 최적화 필요할 수 있음

### 5.3 정책 weight 이식

```bash
# Host에서:
scp runs/stage<final>/ppo.pt runs/stage<final>/norm.json jetson:~/khj_rl_deploy/

# Jetson에서:
python -c "
import torch
ckpt = torch.load('ppo.pt', map_location='cpu')
# 동일 ActorCritic 클래스로 load. obs_dim/action_dim은 cfg 동일이라 매치.
"
```

---

## 6. Few-shot Real-data Fine-tune (선택)

Sim 정책이 real에서 < 70% 성공률이면:

1. Real robot으로 success한 episode 10-50개 수집 (사람이 oracle 역할)
2. 그 demo로 BC fine-tune (frozen obs normalizer 그대로, 1-2 epoch만)
3. 평가, 반복

목적: 미세 sim2real gap (마찰, 모터 deadband) 학습. 전체 재학습 X.

---

## 7. 안전 체크리스트 (Phase 3 시작 전)

| 항목 | 상태 |
|---|---|
| E-stop 물리 스위치 작동 확인 | □ |
| Joint soft limit driver layer 구현 + 테스트 | □ |
| Gripper torque limit 설정 | □ |
| Hand-eye calibration 오차 < 5mm | □ |
| Sim2Real obs 분포 KS test 통과 | □ |
| Jetson inference latency < 50ms | □ |
| Watchdog 200ms 이상 idle 시 E-stop 작동 | □ |
| 첫 실 테스트는 cube 대신 soft foam | □ |
| 안전 거리 (사람 - robot) 50cm 이상 | □ |

---

## 8. 단계별 deploy 일정

| Step | 작업 | 시간 |
|---|---|---|
| 1 | Hand-eye calibration | 2-4h (one-time) |
| 2 | SO-ARM driver layer 구현 + 단위 테스트 | 8-12h |
| 3 | HSV cube detector + 좌표계 통합 | 4-8h |
| 4 | Sim2Real obs 분포 비교 도구 | 4-6h |
| 5 | Jetson 환경 셋업 + inference latency 측정 | 4-8h |
| 6 | Sim 정책 → Real 추론 첫 시도 (cube 없이) | 2-4h |
| 7 | Real cube 첫 PnP 시도 (관찰 + 안전 거리) | 2-4h |
| 8 | Few-shot fine-tune (필요 시) | 8-12h |
| **총** | | **~30-50h** (몇 일~주) |

---

## 9. Phase별 차이

- **Phase 1**: sim에서 single-env로 학습 + 안전장치 검증 (현재)
- **Phase 2**: vec env로 학습 throughput 100배 ↑, robust 정책
- **Phase 3 (Real deploy)**: sim 정책을 real에서 검증, 필요 시 few-shot fine-tune
- **Phase 4** (선택): real 데이터 누적 → real-only DAgger / online RL

---

## 10. 미해결 / 후속 결정

- **Real cube 시작 위치 결정**: sim curriculum stage 0 (5cm)부터 시작 vs sim stage 3 (20cm) 일반화된 정책 직접 사용
- **Gripper force 임계 측정**: 실 cube 잡을 때 최소 토크 vs cube 파손 토크 측정 필요
- **Hand-eye 자동화 spawn**: AprilTag 부착 자세 자동 grid 명령 스크립트
- **Real failure 분석 toolkit**: PnP 실패 시 video + obs trace + sim 재현 워크플로 정의
