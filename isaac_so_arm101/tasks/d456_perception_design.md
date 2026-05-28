# D456 Perception Design (Sim2Real Deploy)

작성: 2026-05-11 오후 · 5/11 저녁 개정 (cube 단일 → 임의 물체 3종) · 대상: SO-ARM101 한 팔 pick & lift, 5/14(수) deploy 결정용
입력 spec: `tasks/oracle_interface_spec.md` (P0 6 / P1 5 / P2 4)
관련 코드: `/home/j-k14d101/jabis_sim/sim2real/oracle/oracle_policy.py` (참고용, 수정 금지)

**5/10 결정 반영** — 시연 물체 3종: **phone** (확정, S26 3D 프린팅 + 거치대 완성), **cube** (sim 검증 동일), **trash** (cube 형태 잠정, 팀원 의논 중). Oracle 아키텍처 (c) 채택: Oracle은 grip pose(position + orientation)만 입력받아 pick & lift; 물체별 grip pose 결정은 perception/상위 로직 책임.

> ⚠ "사용자 확인 필요" 표시는 SO-ARM101 실물 사양·팀 표준이 코드만으로 확정 불가한 항목.

---

## 1. Calibration 전략

spec P0 #1, #2, #3을 본 문서로 이관. 의존 순서: **§1.2 영점 → §1.3 EE link → §1.1 Extrinsic** (사전 조건이 깨지면 후속 측정 의미 없음).

**5/10 결정 반영** — §1 (extrinsic / 영점 / EE link) 모두 **물체 형태 무관**. 시연 물체가 cube 단일 → 임의 3종(phone/cube/trash)으로 바뀌어도 §1.1~§1.3 절차·합격 기준 변경 없음.

### 1.1 Extrinsic (D456 camera frame ↔ robot base frame)

**무엇을** — D456가 캡처한 3D point를 robot base frame으로 변환하는 4×4 행렬 `T_base_cam`.

**방법 후보 비교**

| 방법 | 정확도 | 소요 시간 | 비고 |
|---|---|---|---|
| ArUco 보드 | mm급 (board 4 코너 평균) | 30분 | 보드 1개, 알려진 base 위치에 고정 |
| Hand-eye (eye-on-base) | mm~cm급 | 1~2시간 | EE에 마커 부착, 여러 자세 수집 |
| 수동 측정 | cm급 | 5분 | 자/줄자, 정확도 낮음 |

**추천**: **ArUco 보드** — 시간/정확도 균형. EE link(§1.3) 미확정 상태에서도 가능 (hand-eye는 §1.3 의존). hand-eye는 시간 부족.

**측정 절차**
1. ArUco 보드를 base 좌표 (0, 0, 0)에서 알려진 위치/회전 (`T_base_board`)로 고정. 보드 4 코너의 base frame 좌표를 자/캘리퍼로 측정 (mm 단위). → 사용자 측정.
2. D456 RGB로 보드 캡처. `cv2.aruco.detectMarkers` + `estimatePoseSingleMarkers`로 `T_cam_board` 산출.
3. `T_base_cam = T_base_board · T_cam_board⁻¹`.
4. 결과 행렬을 `~/jabis_sim/sim2real/perception/extrinsic.yaml`에 저장.

**합격 기준** (수치)
- 보드 4 코너 reproject 오차: **각 ≤ 5 mm**, 평균 ≤ 3 mm
- 검증: 알려진 base 좌표 5개 점 (테이블 위 grid)을 `T_base_cam`으로 재투영 → **실측-예측 오차 ≤ 10 mm** 모두 충족
- (참고: 잡기 여유는 물체별로 다름 — cube/trash는 ±10 mm, phone은 거치대 정밀도에 의존. §4에서 물체별 분리 측정. extrinsic 자체 정확도(≤10 mm)는 §1.3 EE link(≤5 mm)와 합쳐 §4 합격 기준의 base가 됨.)

### 1.2 서보 영점 ↔ sim `default_arm` 일치

**무엇을** — 5 arm 서보의 영점(zero position)이 sim의 `default_arm` (oracle_policy.py:120, `robot.data.default_joint_pos[:, arm_ids_t]`)과 일치한다는 것을 보장. 안 맞으면 oracle action 변환 `joint_target = arm_raw * scale + default_arm` 이 모두 offset.

**방법** (선택지 없음, 영점 정렬은 필수)
1. sim의 `default_arm` 값 5개를 dump (rad). → 즉시 가능.
2. 실물 SO-ARM101을 sim과 동일 home 자세로 수동 정렬 (사용자 jig 필요 또는 시각적 확인).
3. Feetech 서보 `present_position` 읽기 → step 단위. step → rad 변환 (서보 분해능 필요, **사용자 확인 필요**: 보통 STS3215는 4096 step / 360°).
4. 실물 영점이 sim과 ±X rad 차이 시: (a) 서보 영점 재설정 (Feetech `Lock=0` + `Torque_Enable=0` 후 `goal_position` 으로 재정의), 또는 (b) oracle 측 `default_arm`을 실물 값으로 override (`OraclePolicy.setup` 후 `oracle.default_arm = realbot_default`).

**합격 기준** (수치)
- joint별 정렬 오차: **각 ≤ 0.01 rad (≈0.6°)**
- 검증: 영점 정렬 후 sim에서 dump한 `default_arm`을 실물에 명령 → 실물 `present_position` 읽어 차이 측정
- 실패 시: 어느 joint가 어긋났는지 출력 + (b) 옵션으로 fallback

### 1.3 EE link `gripper_link` 정의 일치

**무엇을** — sim URDF에서 `gripper_link`로 지정된 link (oracle_policy.py:107)가 실물 URDF에서도 같은 이름·같은 origin·같은 orientation으로 존재. FK/Jacobian/IK target frame이 모두 이 link 기준이므로 위치가 다르면 IK 결과가 어긋남.

**방법**
1. **코드 검토**: sim URDF (`src/isaac_so_arm101/robots/trs_so101/`)와 실물 URDF (사용자가 사용할 URDF, **사용자 확인 필요**) 의 `gripper_link` 정의 비교 — `<origin xyz=... rpy=...>` 일치 여부.
2. **물리 측정**: 실물 robot을 home 자세에 두고 `gripper_link` 위치(중심점)를 base 기준으로 자/캘리퍼 측정. 동일 자세에서 sim FK 결과 (oracle_policy.py:284-289 의 `ee_pose_w` → `ee_pos_b`)와 비교.

**합격 기준** (수치)
- URDF origin 차이: xyz **각 ≤ 1 mm**, rpy **각 ≤ 0.5°**
- 물리 측정 vs sim FK: position 오차 **≤ 5 mm**, orientation 오차 **≤ 2°**
- 실패 시: 실물 URDF를 sim에 맞추거나 oracle의 `ee_body_name` 변경 (이 경우 jacobian index도 재확인 필요)

---

## 2. Perception Pipeline

**5/10 결정 반영** — cube 단일 → 다중 클래스. HSV segmentation **폐기** (다중 클래스 단색 가정 무너짐). YOLO multi-class only. Detection 결과에 grip pose lookup 단계 추가.

```
D456 RGBD ─► YOLO multi-class detection ─► 3D deprojection ─► object_class 분류 ─►
   (RGB)         (bbox + class)              (camera frame)
                                                                ▼
                              ◄── grip pose lookup table (물체별 grip 전략) ──
                                                                │
                              base frame 변환 (T_base_cam) ─► grip_pose_b
                                                                  (pos+quat)
```

| 단계 | 입력 | 출력 | 라이브러리 / 모델 | 주기 | 실패 모드 |
|---|---|---|---|---|---|
| (a) RGBD 캡처 | — | RGB `(720, 1280, 3)` uint8, depth `(720, 1280)` uint16 (mm) | `pyrealsense2`, depth align to RGB | 30 Hz (D456 datasheet) | USB drop, exposure 부적합 |
| (b) Multi-class detection | RGB | bbox `(x1,y1,x2,y2)` + class ∈ {phone, cube, trash} + score | YOLOv8n custom-train (3-class) | 30 Hz | detection 없음 / 다중 detection / class 혼동 / occlusion |
| (c) 3D deprojection | bbox 중심 (u,v), bbox 내 depth median | `(X,Y,Z)` camera frame (m) | `rs.rs2_deproject_pixel_to_point` (D456 intrinsic) | 30 Hz | depth=0 (홀), depth noise (±5 mm @ 1 m) |
| (d) Class → grip strategy | `class` | grip 전략 (offset, orientation 규칙) | python dict (lookup table, §2.1) | 30 Hz | unknown class (default = side-grip) |
| (e) Grip pose 계산 | `(X,Y,Z)_cam` + grip 전략 | `grip_pose_cam` (pos + quat) | numpy / scipy.spatial.transform | 30 Hz | bbox orientation 계산 실패 (phone) |
| (f) Base 변환 | `grip_pose_cam` | `grip_pose_b` (pos + quat, base frame) | numpy (`T_base_cam` from §1.1) | 30 Hz | extrinsic 오차 그대로 전파 |
| (g) Schema pack | `grip_pose_b` + class + confidence | `{object_class, object_pos_b, grip_pose_b, confidence, timestamp}` | numpy + monotonic clock | 30 Hz | — |

**참고**
- (c) depth 잡음 완화: bbox 중심 1픽셀 대신 **bbox 내 depth median** 사용 (잡음 1/√N 감소).
- 좌표계: D456는 카메라 +Z가 광축 forward, +X 우, +Y 아래. 변환 시 부호 주의 (extrinsic.yaml에 명시).
- (b) YOLO 3-class 학습: 데이터 수집·라벨링 필요. 다른 팀원 IL 트랙 모델은 buds/keys/pen 클래스 → **재활용 불가**, 별도 학습 필요. → §5 #1로 이동.

### 2.1 Grip pose lookup table (5/10 결정 반영)

| `object_class` | grip 전략 | grip position offset (object 중심 기준) | grip orientation (base frame, quat) | 근거 |
|---|---|---|---|---|
| `cube` | **side-grip** | `(0, 0, 0)` (중심) | yaw 0° (gripper x-axis = approach) — sim 검증 동일 | sim 검증 완료 (Path C 시연) |
| `trash` | **side-grip** (cube 통일 가정) | cube와 동일 | cube와 동일 | 형태 통일 가정 — §5 #2 미해결 시 재정의 |
| `phone` | **side-grip** (거치대에 세워야 하므로 top-down 부적합) | 폰 옆면 중심 — bbox 단축(short side) 중심 | yaw = bbox 장축(long side) angle (RGB에서 추출) | 거치대 시나리오, top-down은 polo 자세로 안됨 |

**구현 위치 (예정)**: `~/jabis_sim/sim2real/perception/grip_lookup.py`

**미해결** (§5로):
- phone bbox 장축 angle → grip yaw 변환 정밀도 요구 (거치대 허용 오차) — §5 #3
- cube/trash 형태 통일 여부 → trash 행 재정의 가능성 — §5 #2
- place-with-orientation (phone을 거치대에 세우기) — 단순 lift 후 추가 단계 필요 — §5 #4

---

## 3. Interface (spec Oracle 입력과 매칭)

**5/10 결정 반영** — Oracle 아키텍처 (c) 채택. perception은 단순 위치(`cube_pos_b`)가 아닌 **`grip_pose_b` (position + orientation)** 를 공급. spec Oracle 입력의 `target_object_pose` 자리에 grip_pose_b가 들어감.

| spec Oracle 입력 | perception 출력 항목 | 형식 | 주기 | 책임 |
|---|---|---|---|---|
| target_object_pose (= grip_pose_b) | `grip_pose_b` `(pos:(3,), quat:(4,))` float32, base frame | numpy struct + class metadata | 30 Hz stream | **perception** (lookup 결과) |
| `goal_pos_b` `(N, 3)` (선택, callback) | (해당 없음) | — | — | **perception 아님** — 1회 측정/하드코딩 (spec 미해결 #9). phone의 경우 거치대 위치가 이에 해당 — §5 #4 참조 |
| `ee_pos_b`, `ee_quat_b` `(N, 3/4)`, base | (해당 없음) | — | — | **로봇 컨트롤러** (FK from joint encoder) |
| `joint_pos_arm` `(N, 5)` rad | (해당 없음) | — | — | **로봇 컨트롤러** (Feetech `present_position` → rad) |
| `jacobian` `(N, 6, 5)`, base frame | (해당 없음) | — | — | **로봇 컨트롤러** (URDF FK 라이브러리, e.g. pinocchio) |

**결과**: perception은 **`grip_pose_b` 1개 항목만** 책임 (lookup table 결과로 계산). object_class 등은 metadata로 동봉. 나머지 4개 spec 입력은 로봇 컨트롤러 layer 책임. **미매칭 항목 없음**.

**Schema 합의 필요** (spec 미해결 #11 transport, 본 문서 §5 #5):
```python
# perception → oracle (proposed, ROS2 topic 또는 shared memory)
{
    "object_class": str,        # "phone" | "cube" | "trash"
    "object_pos_b": np.ndarray(shape=(3,), dtype=np.float32),    # m, base frame, object 중심
    "grip_pose_b": {
        "position":    np.ndarray(shape=(3,), dtype=np.float32), # m, base frame, grip point
        "orientation": np.ndarray(shape=(4,), dtype=np.float32), # quat (qx,qy,qz,qw), base frame
    },
    "confidence": float,        # YOLO score, 0~1
    "timestamp": float,         # monotonic time, sec
    "valid": bool,              # detection success
    "depth_quality": float,     # optional, 0~1, bbox depth std 기반
}
```

**필드별 책임/계산**:
- `object_class`: YOLO bbox class label (§2 (b)).
- `object_pos_b`: bbox 중심의 deproject + base 변환 (§2 (c)+(f)). 동일 물체에서 grip_pose_b.position과 다를 수 있음 (예: phone은 옆면 중심으로 grip).
- `grip_pose_b.position`: lookup table의 offset 적용 후 base 변환 (§2 (d)+(e)+(f)).
- `grip_pose_b.orientation`: lookup table의 orientation 규칙 (cube=고정 yaw0, phone=bbox 장축 angle) → base frame quat.
- `confidence`: YOLO detection score 그대로.
- `valid`: detection 1개 이상 + depth 유효 + grip 계산 성공 시 True.

---

## 4. 검증 절차 (실물 통합 전 perception 단독)

**5/10 결정 반영** — 물체별로 분리 측정. 물체 위치 오차 + grip pose 정확도 둘 다 검증.

**목적**: §1 calibration + §2 pipeline + §2.1 lookup이 합쳐
- (i) **object_pos_b 위치 오차**가 oracle이 도달 가능한 한계 (`reach_dist=0.02 m`) 안에 들어옴
- (ii) **grip_pose_b** (특히 phone orientation)가 실제 grip 가능한 정확도

**셋업**
- 실물 SO-ARM101 base 고정.
- D456 mount 위치 확정 (extrinsic 측정 후).
- 시연 물체 3종 (phone 1대 / cube 1개 / trash 1개) — phone·cube 사양은 부록 A3a/A3b.

### 4.1 위치 정확도 (object_pos_b) — 물체별 분리

| 물체 | 위치 N | 측정/물체 | 합격 기준 (mean / p95 / max / 성공률) |
|---|---|---|---|
| `cube` | 10 (x∈[0.1, 0.4]×4, y∈[-0.1, 0.1]×3 grid 중 12 → 10 sampling) | 100 frame | mean ≤ 5 mm / p95 ≤ 10 mm / max ≤ 15 mm / 성공률 ≥ 95% |
| `phone` | **N = 6** (직립 거치대 입구 근처, x∈[0.15, 0.30]×3, y∈[-0.05, 0.05]×2) | 100 frame | cube와 동일 |
| `trash` | (cube와 형태 통일 시 별도 측정 **불필요**, cube 결과 재사용) | — | (cube 결과로 갈음) — §5 #2 결정에 따라 변경 |

**절차** (각 물체 동일)
1. 테이블 위 알려진 base frame 좌표 N 위치에 물체 배치. 실측은 자/캘리퍼 (mm 단위).
2. 각 위치에서 perception pipeline 실행 → `object_pos_b_pred` 100 frame 평균.
3. 오차 `e_i = ||object_pos_b_pred_i − object_pos_b_true_i||` 계산. 분포 (mean, std, max, p95) 출력.

### 4.2 Grip pose 정확도 (orientation) — 물체별

| 물체 | 검증 방법 | 합격 기준 |
|---|---|---|
| `cube` | 고정 yaw 0°, orientation 의존 없음 → 검증 생략 (lookup 상수) | (생략) |
| `phone` | 회전 5종 자세 (0°, 30°, 60°, 90°, 135°) × 5회 = 25 trial. perception이 추정한 grip yaw vs 실측 (수동 각도기) | **각도 오차 mean ≤ 5°, max ≤ 10°** (거치대 허용 오차 가정 — §5 #3) |
| `trash` | (cube 통일 가정 시 cube 동일) | (cube 결과로 갈음) |

### 4.3 Grip 실제 성공률 (sim 또는 실물)

각 물체에 대해 lookup table이 산출한 grip_pose_b로 oracle을 구동했을 때 실제 grip 성공 여부.
- 시도 횟수: 물체별 **N = 20** (시간 허용 범위)
- 합격 기준: **성공률 ≥ 80%** (cube), **≥ 70%** (phone, 거치대 시나리오 난이도 반영)
- 실패 분류: detection fail / grip pose 오차 / oracle IK 실패 — §5 #4 (place-with-orientation)와 분리

**스크립트 위치 / 실행 명령** (구현 예정)
- 작성: `~/jabis_sim/sim2real/perception/eval_static_objects.py`
- 실행:
  ```bash
  python eval_static_objects.py --extrinsic extrinsic.yaml \
      --object {phone|cube|trash} \
      --positions positions_<obj>.csv --frames_per_pos 100 \
      --output static_eval_<obj>_2026-05-13.csv
  ```

**Pass/Fail 처리**
- §4.1 Pass → object_pos_b 신뢰. Fail (mean > 5 mm) → extrinsic 재측정 또는 §2(c) depth median window 확대 (3×3 → 7×7).
- §4.2 Pass → grip orientation 신뢰. Fail (phone) → bbox 장축 추출 알고리즘 교체 (PCA → minAreaRect 등).
- §4.3 Fail → 어느 단계 실패인지 분류 후 lookup table 또는 oracle 측 튜닝.

---

## 5. 미해결 이슈 / 결정 필요 사항 (perception 고유)

**5/10 결정 반영** — 기존 #2 "Cube color/shape/size" 단일 항목 제거 (부록 A3a/A3b로 분리). 다중 클래스/grip pose 확장 관련 신규 4개 추가.

spec에 없는 perception-specific 이슈만. deadline은 5/14 deploy 결정 시점 기준.

1. **YOLO 다중 클래스 모델 학습/획득** *(신규, 5/10 결정 반영)* — 3-class (phone/cube/trash) 학습 필요. 다른 팀원 IL 트랙 모델은 buds/keys/pen 클래스라 **재활용 불가**. 학습 데이터 수집(D456으로 각 물체 N장 캡처) + 라벨링 + 학습(YOLOv8n, 30분~수시간) 또는 zero-shot (e.g. Grounding-DINO/SAM) 사용 결정.
   - 영향: §2(b) 구현 가능 시점. 학습 fail 시 시연 trash·cube 분리 불가.
   - deadline: **5/12(월)** — perception 코드 작성 시점. 사용자 확인 필요 (학습 vs zero-shot, 데이터 수집 방법).
2. **Cube ↔ trash 형태 통일 여부** *(신규, 5/10 결정 반영)* — trash는 현재 cube 형태 잠정. 팀원 의논 진행 중. 통일 안 되면 §2.1 lookup `trash` 행 재정의 (다른 grip 전략) + §4 trash 위치 별도 측정 필요.
   - 영향: §2.1 lookup 1행, §4.1 trash 측정 N, §4.2 trash orientation 검증 추가 가능성.
   - deadline: **5/12(월)** — lookup table 확정 전. 팀원 의논 결과 대기.
3. **Phone grip pose 정밀도 요구 (거치대 허용 오차)** *(신규, 5/10 결정 반영)* — phone을 거치대에 세울 때 grip orientation 허용 오차 (현재 §4.2에 ±5° / ±10° 가정). 거치대가 빡빡하면 더 엄격 필요.
   - 영향: §4.2 합격 기준 수치, §2 (e) bbox 장축 추출 알고리즘 선택.
   - deadline: **5/13(화)** — 검증(§4) 전. 김현준 직접 측정 가능 (부록 A3b).
4. **Place-with-orientation Oracle 확장 여부** *(신규, 5/10 결정 반영, 본 perception 문서 범위 약간 초과 — flag 용도)* — phone을 거치대에 세우는 시나리오는 단순 pick & lift가 **아님**. orientation을 유지하며 특정 위치에 place 필요. 현 oracle 5-state machine (`APPROACH→DESCEND→CLOSE→LIFT→MOVE_TO_GOAL`) 중 `MOVE_TO_GOAL`이 position-only이므로 확장 또는 별도 oracle variant 필요.
   - 영향: oracle 아키텍처 결정. perception은 `goal_pose_b` 까지 공급해야 할지 / oracle이 알아서 처리할지 interface 변경 가능성.
   - 결정 옵션: (a) Oracle state machine에 `PLACE_WITH_ORIENT` state 추가, (b) phone 전용 oracle variant 작성, (c) phone은 시연 lift만 하고 place 생략.
   - deadline: **5/12(월)** — oracle 작업 시점.
   - 사용자 확인 필요. (perception 문서 범위 초과지만 perception schema가 영향 받으므로 flag.)
5. **D456 mount 위치 / 자세** — eye-on-base (테이블 위 별도 stand) vs eye-on-hand (EE에 부착) vs 천장 mount.
   - 영향: extrinsic 측정 방법 (eye-on-hand이면 ArUco 무용, hand-eye 강제), 가시성 (eye-on-hand은 occlusion 적지만 거리 변동 큼), 캘리브레이션 빈도 (mount 흔들리면 재측정).
   - deadline: **5/12(월) 오전** — extrinsic 작업 전제. 사용자 확인 필요.
6. **Lighting 조건 / 변동** — 시연 환경 조명 (실내 형광? 자연광? 가변?), reflection·shadow 정도.
   - 영향: YOLO detection robustness 요구 (학습 데이터 다양성), depth 품질 (D456 IR projector는 형광등 영향 적지만 직사광 약함). phone 표면 반사 주의.
   - deadline: **5/13(화)** — 검증(§4) 전. 사용자 확인 필요.
7. **Detection 실패 시 동작 정책 (다중 클래스 반영)** — `valid=False` 1프레임 vs 연속 K프레임. 다중 클래스이므로 **어느 class fail인지** 추가 정보 필요 (예: phone class만 탐지 안 됨 vs 전체 detection 0).
   - 영향: perception이 stale pose hold 책임지는지 vs oracle이 책임지는지 — interface 의미 결정.
   - 결정 옵션: (a) perception이 K=3 프레임 내 hold + valid 유지, (b) perception은 즉시 valid=False + 어느 class fail인지 metadata, oracle 측 K-streak 판단.
   - 추천: **(b)** — 책임 분리 명확. perception은 "지금 무엇이 보이냐"만 담당.
   - deadline: **5/12(월)** — interface schema 확정 시점.
8. **Transport / IPC 채널** — ROS2 topic / shared memory / ZMQ / 파일 (spec 미해결 #11과 동일).
   - 영향: latency (shm: <1ms / ROS2: ~5ms / ZMQ: ~2ms), 팀 표준 호환.
   - deadline: **5/12(월)** — perception/oracle 양쪽 코드 작성 전제. 사용자 확인 필요.

---

## 부록 — 가정/사용자 확인 필요 항목 모음

본 문서는 다음을 가정. 다르면 해당 섹션 재작성 필요.

| # | 가정 | 사용자 확인 |
|---|---|---|
| A1 | D456 datasheet: depth 0.4–10 m, RGB 1280×720 @30Hz | 공식 spec, OK |
| A2 | Feetech STS3215 분해능 4096 step / 360° | 사용자 확인 필요 (다른 모델일 수 있음) |
| A3a | **Cube / 쓰레기 사양** *(5/10 결정 반영)* — size (가정 50 mm), color (가정 단색), 형태 (가정 정사각형), cube↔쓰레기 형태 통일 여부 | 사용자 확인 필요 (§5 #2 — 팀원 의논 중) |
| A3b | **Phone 사양** *(5/10 결정 반영)* — S26 3D 프린팅 모델 정확한 dimension (mm), 거치대 입구 폭/높이/깊이, 거치대 base 좌표 위치 + orientation | **김현준 직접 측정 가능** (deadline 5/12) — §5 #3와 연결 |
| A4 | Robot base가 world frame과 회전 없이 align (spec P2 #13) | 사용자 확인 필요 |
| A5 | sim URDF의 `gripper_link`가 실물 URDF에 동일 이름으로 존재 | 사용자 확인 필요 (§1.3) |
| A6 | 작업 반경: **모든 시연 물체** (phone/cube/trash)가 base 기준 x∈[0.1, 0.4], y∈[-0.15, 0.15], z=table 안에 위치. phone 거치대도 동일 범위 안 | 사용자 확인 필요 (oracle workspace 한계) |
