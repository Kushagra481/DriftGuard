"""OpenCV convention: x right, y down, z forward; world gravity +y."""
import numpy as np
from scipy.spatial.transform import Rotation

K = np.array([[460., 0, 320.], [0, 460., 240.], [0, 0, 1.]])
GRAVITY = np.array([0., 9.81, 0.])


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def project(landmarks, position, rotation, intrinsics=K):
    """Project world points through R_wc (camera-to-world); return uv/depth."""
    pc = (np.asarray(landmarks) - position) @ rotation.as_matrix()
    z = pc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = pc[:, :2] / safe_z[:, None]
    uv = uv * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2]
    return uv, z


def pose_dict(position, rotation):
    return {"position": np.asarray(position).tolist(),
            "quaternion": rotation.as_quat().tolist()}


def camera_points(landmarks, position, rotation):
    return (landmarks - position) @ rotation.as_matrix()
