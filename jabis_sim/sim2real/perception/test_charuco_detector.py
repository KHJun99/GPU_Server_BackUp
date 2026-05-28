"""Unit tests for charuco_detector.

Hardware-free: synthetic ChArUco images are rendered via
``cv2.aruco.CharucoBoard.generateImage()`` (OpenCV 4.7+ API).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

# Make sibling module importable when running ``pytest`` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from charuco_detector import (  # noqa: E402
    CHECKER_COLS,
    CHECKER_ROWS,
    MIN_CORNERS_FOR_POSE,
    BoardPose,
    detect_charuco,
    get_board,
)

# ChArUco interior corners = (rows-1) * (cols-1).
INTERIOR_CORNERS = (CHECKER_ROWS - 1) * (CHECKER_COLS - 1)


def _make_synthetic_board_bgr(image_size=(800, 640), margin: int = 20) -> np.ndarray:
    """Render the configured ChArUco board to a BGR uint8 image."""
    board = get_board()
    # OpenCV 4.7+ signature: generateImage(outSize, marginSize, borderBits)
    gray = board.generateImage(image_size, marginSize=margin, borderBits=1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _synthetic_K(image_size=(800, 640), fov_deg: float = 60.0) -> np.ndarray:
    w, h = image_size
    f = w / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _zero_dist() -> np.ndarray:
    return np.zeros(5, dtype=np.float64)


# --------------------------- Detection cases --------------------------- #


def test_synthetic_charuco_detection():
    """Full board image → detect most interior corners with low reprojection error."""
    img = _make_synthetic_board_bgr()
    K = _synthetic_K()
    pose = detect_charuco(img, K, _zero_dist())
    assert pose is not None
    # On a clean synthetic image we expect to recover almost all interior corners.
    assert pose.n_corners_detected >= INTERIOR_CORNERS - 4, (
        f"expected ≥ {INTERIOR_CORNERS - 4} corners, got {pose.n_corners_detected}"
    )
    # Tolerance is in pixels (Codex review): synthetic image with planar projection
    # should reproject far below 2 px RMS.
    assert pose.reprojection_rms_px <= 2.0, (
        f"RMS = {pose.reprojection_rms_px:.3f} px exceeds 2.0 px"
    )


def test_detection_failure_graceful_black_image():
    """검은 이미지 (보드 없음) → None 반환, 예외 raise 금지."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    K = _synthetic_K(image_size=(640, 480))
    pose = detect_charuco(img, K, _zero_dist())
    assert pose is None


def test_partial_detection():
    """오른쪽 절반 가림 → 코너 수 줄지만 pose는 산출 (또는 임계값 미만이면 None)."""
    img = _make_synthetic_board_bgr()
    img_masked = img.copy()
    img_masked[:, img_masked.shape[1] // 2:, :] = 0
    K = _synthetic_K()
    pose = detect_charuco(img_masked, K, _zero_dist())
    if pose is not None:
        # 가렸으니 전체 코너보다는 적게 검출돼야 함
        assert pose.n_corners_detected < INTERIOR_CORNERS
        assert pose.n_corners_detected >= MIN_CORNERS_FOR_POSE


def test_below_threshold_returns_none():
    """30 px strip만 남기면 코너가 임계값 미만이라 None."""
    img = _make_synthetic_board_bgr()
    masked = np.zeros_like(img)
    masked[:, :30, :] = img[:, :30, :]
    K = _synthetic_K()
    pose = detect_charuco(masked, K, _zero_dist())
    assert pose is None


def test_grayscale_input_supported():
    """Grayscale (H,W) 입력도 받아야 함."""
    img = _make_synthetic_board_bgr()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    K = _synthetic_K()
    pose = detect_charuco(gray, K, _zero_dist())
    assert pose is not None
    assert pose.n_corners_detected >= MIN_CORNERS_FOR_POSE


# --------------------------- Input validation --------------------------- #


def test_invalid_K_shape_raises():
    img = _make_synthetic_board_bgr()
    with pytest.raises(ValueError, match="3x3"):
        detect_charuco(img, np.eye(2), _zero_dist())


def test_invalid_dist_shape_raises():
    img = _make_synthetic_board_bgr()
    K = _synthetic_K()
    bad_dist = np.zeros(3)  # 3 not in {4,5,8,12,14}
    with pytest.raises(ValueError, match="dist"):
        detect_charuco(img, K, bad_dist)


def test_invalid_image_shape_raises():
    K = _synthetic_K()
    bad_img = np.zeros((10, 10, 4), dtype=np.uint8)  # 4-channel
    with pytest.raises(ValueError):
        detect_charuco(bad_img, K, _zero_dist())


def test_none_image_raises():
    K = _synthetic_K()
    with pytest.raises(ValueError, match="None"):
        detect_charuco(None, K, _zero_dist())


# ---------------------------- BoardPose API ---------------------------- #


def test_to_4x4_matrix_format():
    """to_4x4_matrix() returns a valid SE(3) homogeneous matrix."""
    img = _make_synthetic_board_bgr()
    K = _synthetic_K()
    pose = detect_charuco(img, K, _zero_dist())
    assert pose is not None

    M = pose.to_4x4_matrix()
    assert M.shape == (4, 4)
    assert np.allclose(M[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)

    R = M[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "rotation block must be orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "rotation block must have det=1"


def test_board_pose_dataclass_fields():
    """BoardPose carries rvec, tvec, n_corners_detected, reprojection_rms_px."""
    img = _make_synthetic_board_bgr()
    K = _synthetic_K()
    pose = detect_charuco(img, K, _zero_dist())
    assert isinstance(pose, BoardPose)
    assert pose.rvec.shape == (3,)
    assert pose.tvec.shape == (3,)
    assert isinstance(pose.n_corners_detected, int)
    assert isinstance(pose.reprojection_rms_px, float)
    assert pose.reprojection_rms_px >= 0.0
