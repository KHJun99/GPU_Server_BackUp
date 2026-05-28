# perception/ — D456 ChArUco 검출

D456 RGBD 카메라 ↔ 로봇 base frame extrinsic 캘리브레이션의 첫 단계.
`charuco_detector.py` 가 인쇄된 ChArUco 보드를 검출해 카메라 frame
기준 6-DoF 보드 pose 를 산출. 5/12 deploy 시 `T_base_camera`
계산의 입력으로 사용.

## 보드 파라미터 (현재 가정값)

| 항목 | 값 | 출처 |
|---|---|---|
| Dictionary | `cv2.aruco.DICT_4X4_50` | 사이트 default 가정 |
| Checker rows × cols | 5 × 7 | 인쇄된 보드 |
| Checker size | 0.040 m | 인쇄된 보드 |
| Marker size | 0.030 m | ChArUco 규약 (checker × 0.7~0.8) |
| 최소 검출 코너 | 6 | 안전 마진 (PnP 최소 4) |

> **TODO (5/12 전)**: 인쇄 PDF 메타정보로 dictionary, marker size 검증.
> 차이가 있으면 `charuco_detector.py` 상단 상수 4개를 실측값으로 교체.
> 인쇄 보드와 코드의 보드 파라미터가 일치하지 않으면 검출 0개.

검증 방법:
```python
from charuco_detector import get_board
img = get_board().generateImage((800, 640), marginSize=20, borderBits=1)
import cv2; cv2.imwrite("/tmp/expected_board.png", img)
# /tmp/expected_board.png 와 인쇄된 보드 비교 — 마커 패턴이 같아야 함.
```

## D456 Intrinsics 추출

```bash
python get_d456_intrinsics.py             # → ./d456_intrinsics.json 생성
python get_d456_intrinsics.py --output /tmp/intr.json
```

D456 미연결 / pyrealsense2 미설치 시 명확한 에러 + exit 1.
저장 JSON 구조와 charuco_detector 와의 연결은 스크립트 docstring 참조.

**주의**:
- 저장된 K/dist 는 **640×480 frame 에만 유효** — 다른 해상도로 촬영
  하려면 해당 해상도용 intrinsics 를 다시 추출.
- D456 color stream 의 distortion model 이 종종
  `inverse_brown_conrady` 로 노출됨. OpenCV `solvePnP` 는 forward
  Brown-Conrady 가정. 5/12 실측 시 `intr["color"]["model"]` 확인 +
  reprojection RMS 검증 필수 (Codex review 5/11).
- `depth_to_color_extrinsic.rotation` 은 row-major 3×3 (librealsense
  내부의 column-major 를 row-major 로 변환해 저장).

## 사용 예 (charuco_detector + intrinsics 결합)

```python
import json, numpy as np
from charuco_detector import detect_charuco

# get_d456_intrinsics.py 로 미리 저장한 JSON 로드
intr = json.load(open("d456_intrinsics.json"))
K    = np.array(intr["color"]["K"])
dist = np.array(intr["color"]["dist"])

bgr = ...  # D456 RGB 프레임 (BGR uint8)
pose = detect_charuco(bgr, K, dist)
if pose is None:
    print("검출 실패 — 보드가 안 보이거나 코너 < 6개")
else:
    T_camera_board = pose.to_4x4_matrix()  # 4×4
    print(f"corners={pose.n_corners_detected}, RMS={pose.reprojection_rms_px:.2f}px")
```

## 좌표계 contract

`BoardPose.to_4x4_matrix()` 는 **`T_camera_board`** 를 반환.
정확한 의미:

> 보드 frame 의 3D 점 `P_b = (X, Y, Z, 1)` 를 카메라 frame 으로 매핑하는
> 변환. `P_c = T_camera_board @ P_b`.

OpenCV `solvePnP` 의 정의 (`rvec`, `tvec` 가 board → camera) 와 동일.

**주의**: `T_world_camera` (world ↔ camera 변환) 는 본 파일에서 계산하지
않음. 보드를 책상 가운데(world 원점) 에 정확히 두는 placement 측정이
완료된 5/12 이후에 별도 함수로 추가 (`board_to_world_transform` 자리는
의도적으로 stub 도 두지 않음 — Codex review 5/11 지적: identity stub 은
silent 하게 잘못된 좌표를 흘려보낼 위험).

contract 가 확정되면 다음 형태로 작성 예정:
```
T_world_camera = T_world_board @ T_board_camera
              = T_world_board @ inv(T_camera_board)
```
`T_world_board` 는 보드 placement 측정 결과 (3-2-1 회전 + 평행이동).
"world 원점 = 책상 가운데" 를 만족하면 `T_world_board ≈ I` 이지만
실측 후 확정.

## 다음 단계 (5/12 deploy)

1. **D456 intrinsics 실측** — `pyrealsense2` 로 `K`, `dist` 가져와 yaml 저장.
2. **보드 placement 측정** — 보드 4 코너 base 좌표 자/캘리퍼 측정 (mm 단위).
   `T_world_board` 산출.
3. **`T_world_camera` 계산 함수 작성** — 본 README §"좌표계 contract" 참조.
4. **합격 기준 검증** (`d456_perception_design.md` §1.1):
   - 보드 4 코너 reproject 오차: 각 ≤ 5 mm, 평균 ≤ 3 mm
   - base grid 5 점 재투영 오차 ≤ 10 mm

## 알려진 이슈 / 주의

- **Dictionary 불일치 시 검출 0개**: 인쇄 PDF 가 DICT_4X4_50 이 아닌
  DICT_4X4_100 / 250 / 1000 일 가능성. 검출률이 0% 이면 가장 먼저 의심.
- **Marker size 불일치**: pose 는 산출되지만 거리 추정이 비례적으로
  틀어짐 (실제가 32 mm 인데 30 mm 가정 → tvec 가 ~6.7% 짧게 나옴).
- **OpenCV 4.7+ 의존**: 4.6 이전의 `CharucoBoard_create` 는 폐기됨.
  현재 환경 4.11.0 확인.
- **ArUco namespace**: 표준 `opencv-python` 또는 `opencv-contrib-python`
  어느 쪽이든 4.5+ 부터 `cv2.aruco` 노출. import 시 명시 체크는 없으니
  ImportError 가 나면 패키지 종류 점검.
- **이미지 채널 순서 (BGR 가정)**: 함수 인자명은 `rgb_image` 이지만
  내부적으로 BGR (OpenCV 기본) 또는 grayscale 로 처리. `pyrealsense2`
  stream 은 `rgb8` / `bgr8` 둘 다 설정 가능하니 호출 측에서 일관되게
  **BGR 로 맞춰서** 넣을 것. ChArUco 검출은 binary 라 swap 이 검출률에는
  거의 영향 없음, 다만 디버그 이미지 저장이나 색 normalization 등
  color-sensitive downstream 에서 surprise 가능.

## 테스트

```
cd ~/jabis_sim/sim2real && python3 -m pytest perception/ -v
```

11 테스트 — 합성 보드 정상 검출, 검은 이미지 → None, 부분 가시성,
임계값 미만 → None, grayscale 입력, K/dist/image shape 오류 raise,
`to_4x4_matrix()` SE(3) 형식 검증 등.
