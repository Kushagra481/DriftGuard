"""RANSAC known-map pose estimation. This is NOT monocular SLAM."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from backend.simulation.geometry import K, project


@dataclass
class VisualPose:
    position: np.ndarray
    rotation: Rotation
    residuals: np.ndarray
    inliers: np.ndarray
    rmse: float
    ratio: float


def estimate_pose(landmarks, ids, uv):
    if len(ids) < 6:
        return None
    points = np.ascontiguousarray(landmarks[ids], dtype=np.float64)
    pixels = np.ascontiguousarray(uv, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, pixels, K, None, iterationsCount=120,
        reprojectionError=4., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None
    ii = inliers.ravel()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(points[ii], pixels[ii], K, None, rvec, tvec)
    except cv2.error:
        return None
    rotation = Rotation.from_rotvec(rvec.ravel()).inv()
    position = -rotation.apply(tvec.ravel())
    if not np.all(np.isfinite(position)) or np.linalg.norm(position) > 1e4:
        return None
    predicted, depth = project(points, position, rotation)
    residuals = np.linalg.norm(pixels-predicted, axis=1)
    good = (residuals <= 4) & (depth > .1)
    if good.sum() < 6:
        return None
    return VisualPose(position, rotation, residuals, good,
                      float(np.sqrt(np.mean(residuals[good]**2))), float(good.mean()))
