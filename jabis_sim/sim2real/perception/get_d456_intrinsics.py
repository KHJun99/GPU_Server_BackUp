#!/usr/bin/env python3
"""D456 RGB(+depth) intrinsics 추출 → JSON 저장.

5/12 calibration 작업의 첫 단계. ``charuco_detector.detect_charuco`` 가
요구하는 ``camera_matrix`` (3×3 K) 와 ``dist_coeffs`` 를 생성한다.

실행 예
-------
    cd ~/jabis_sim/sim2real/perception
    python get_d456_intrinsics.py
    # → ./d456_intrinsics.json 생성

    python get_d456_intrinsics.py --output /tmp/intr.json
    # 다른 경로에 저장

종료 코드
--------
- 0 : 정상 저장
- 1 : pyrealsense2 미설치 또는 D456 미연결 (USB / 포트 문제 등)

저장 JSON 구조
--------------
::

    {
      "device": {"device_name": ..., "serial": ..., "firmware": ...},
      "fps": 30,
      "color": {
        "width": 640, "height": 480,
        "fx": ..., "fy": ..., "ppx": ..., "ppy": ...,
        "model": "...",
        "K":    [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]],
        "dist": [k1, k2, p1, p2, k3]   # OpenCV Brown-Conrady, 길이 5
      },
      "depth": { ...같은 형식 (보너스) },
      "depth_to_color_extrinsic": {
        "rotation":    [[r11,r12,r13], [r21,r22,r23], [r31,r32,r33]],
        "translation": [tx, ty, tz]    # m, depth → color 좌표계
      }
    }

charuco_detector 와 호환
------------------------
``detect_charuco(rgb, K, dist)`` 에 그대로 전달 가능. **단 두 가지 주의**:

1. ``intr["color"]["model"]`` 가 ``"distortion.brown_conrady"`` 일 때만
   OpenCV ``solvePnP`` 와 의미가 일치한다. D45x/D456 은 종종
   ``"distortion.inverse_brown_conrady"`` 로 노출되며, 이 경우 OpenCV
   forward distortion 모델과 의미가 달라 calibration 정확도 위험.
   5/12 실측 시 model 필드를 확인하고, inverse 라면 reprojection RMS 와
   pose 안정성을 검증하거나 별도 변환 로직 추가 필요.
2. K/dist 는 **저장된 해상도(640×480)에서 받은 frame 에만 유효**.
   다른 해상도로 촬영하면 그 해상도용 intrinsics 를 다시 추출해야 함.

::

    import json, numpy as np
    from charuco_detector import detect_charuco
    intr = json.load(open("d456_intrinsics.json"))
    assert intr["color"]["model"] in ("distortion.brown_conrady",
                                      "distortion.modified_brown_conrady"), \\
        f"unexpected distortion model: {intr['color']['model']}"
    K    = np.array(intr["color"]["K"])
    dist = np.array(intr["color"]["dist"])
    pose = detect_charuco(bgr_frame_640x480, K, dist)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 하드코딩 상수 — D456 RGB stream 표준 설정
WIDTH: int = 640
HEIGHT: int = 480
FPS: int = 30


def _intrinsics_to_dict(intr) -> Dict[str, Any]:
    """``rs.intrinsics`` → JSON-serializable dict."""
    K = [
        [float(intr.fx), 0.0, float(intr.ppx)],
        [0.0, float(intr.fy), float(intr.ppy)],
        [0.0, 0.0, 1.0],
    ]
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "K": K,
        # OpenCV 기대 순서 (k1, k2, p1, p2, k3 — Brown-Conrady).
        # RealSense 는 distortion 모델이 ``inverse_brown_conrady`` 인 경우가 많지만
        # coeffs 길이/순서는 OpenCV 와 동일하게 5개로 노출됨.
        "dist": [float(c) for c in intr.coeffs],
    }


def _extrinsics_to_dict(extr) -> Dict[str, Any]:
    """``rs.extrinsics`` → JSON-serializable dict (R 3×3 row-major, t 3-벡터, m).

    Important: ``rs2_extrinsics.rotation`` 은 librealsense docs 상 길이-9
    **column-major** 배열. JSON 출력은 row-major 3x3 으로 변환해 저장한다
    (Codex review 5/11 — 그대로 [0,1,2]/[3,4,5]/[6,7,8] 행으로 나누면
    실제 회전이 전치된 행렬이 됨).
    """
    r = extr.rotation  # column-major: col0=r[0:3], col1=r[3:6], col2=r[6:9]
    R = [
        [float(r[0]), float(r[3]), float(r[6])],
        [float(r[1]), float(r[4]), float(r[7])],
        [float(r[2]), float(r[5]), float(r[8])],
    ]
    t = [
        float(extr.translation[0]),
        float(extr.translation[1]),
        float(extr.translation[2]),
    ]
    return {"rotation": R, "translation": t}


def _device_info(device, rs) -> Dict[str, Optional[str]]:
    """Device name / serial / firmware (없는 필드는 None)."""

    def _get(field):
        if device.supports(field):
            try:
                return str(device.get_info(field))
            except RuntimeError:
                return None
        return None

    return {
        "device_name": _get(rs.camera_info.name),
        "serial": _get(rs.camera_info.serial_number),
        "firmware": _get(rs.camera_info.firmware_version),
    }


def main(output_path: Path) -> int:
    # --- pyrealsense2 lazy import ---------------------------------------
    try:
        import pyrealsense2 as rs  # noqa: F401  (사용)
    except ImportError:
        print(
            "ERROR: pyrealsense2 가 설치되어 있지 않습니다.\n"
            "  설치: `pip install pyrealsense2` (또는 conda env 에 추가).",
            file=sys.stderr,
        )
        return 1

    # --- pipeline 구성 + start (depth fallback) --------------------------
    # Spec 상 depth 는 "보너스" — 활성화 실패해도 color 만으로 진행. depth 가
    # config 에 들어 있으면 pipeline.start 단계에서 통째로 실패하므로, color
    # 만으로 재시도하는 fallback 흐름을 명시적으로 둔다 (Codex review 5/11).
    pipeline = rs.pipeline()

    def _config(with_depth: bool):
        c = rs.config()
        c.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        if with_depth:
            c.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        return c

    profile = None
    start_error_full = None
    try:
        profile = pipeline.start(_config(with_depth=True))
    except RuntimeError as e:
        start_error_full = e

    if profile is None:
        # depth 실패 가능성 — color 만으로 재시도
        try:
            profile = pipeline.start(_config(with_depth=False))
            print(
                "WARNING: depth stream 활성화 실패, color intrinsics 만 추출.\n"
                f"  원인 후보: {start_error_full}",
                file=sys.stderr,
            )
        except RuntimeError as e_color:
            print(
                "ERROR: D456 이 연결되지 않았습니다 (color stream 도 실패).\n"
                "  점검: USB 3.x 포트 / `realsense-viewer` 인식 여부 / 다른 프로세스 점유.\n"
                f"  full-config error : {start_error_full}\n"
                f"  color-only error  : {e_color}",
                file=sys.stderr,
            )
            return 1

    # --- intrinsics 추출 + JSON 저장 -------------------------------------
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()

        depth_intr = None
        depth_to_color = None
        try:
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intr = depth_stream.get_intrinsics()
            depth_to_color = depth_stream.get_extrinsics_to(color_stream)
        except RuntimeError:
            # depth 가 없거나 extrinsic 추출 실패 — color 만 저장하고 계속.
            pass

        out: Dict[str, Any] = {
            "device": _device_info(profile.get_device(), rs),
            "fps": FPS,
            "color": _intrinsics_to_dict(color_intr),
        }
        if depth_intr is not None:
            out["depth"] = _intrinsics_to_dict(depth_intr)
        if depth_to_color is not None:
            out["depth_to_color_extrinsic"] = _extrinsics_to_dict(depth_to_color)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(out, f, indent=2)

        # --- 콘솔 요약 ---------------------------------------------------
        print(f"OK: intrinsics 저장됨 → {output_path}")
        print(
            f"  color K = "
            f"[[{color_intr.fx:.2f}, 0, {color_intr.ppx:.2f}], "
            f"[0, {color_intr.fy:.2f}, {color_intr.ppy:.2f}], "
            f"[0, 0, 1]]"
        )
        print(
            f"  color dist (len {len(color_intr.coeffs)}) = "
            f"{[round(float(c), 6) for c in color_intr.coeffs]}"
        )
        if depth_intr is not None:
            print(f"  depth fx,fy = ({depth_intr.fx:.2f}, {depth_intr.fy:.2f})")
        return 0
    finally:
        pipeline.stop()


def _parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "d456_intrinsics.json"
    parser = argparse.ArgumentParser(
        description="D456 RGB(+depth) intrinsics → JSON 저장",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"출력 JSON 경로 (default: {default_out})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args.output))
