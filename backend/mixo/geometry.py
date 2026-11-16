"""Body-frame incremental translation and rotation-aware SO(3) fusion."""
import numpy as np
from scipy.spatial.transform import Rotation


def camera_to_body(position, rotation, extrinsic_rotation, extrinsic_translation):
    body_rotation = rotation * extrinsic_rotation.inv()
    return position-body_rotation.apply(extrinsic_translation), body_rotation


def relative_increment(previous_position, previous_rotation, position, rotation):
    return np.r_[previous_rotation.inv().apply(position-previous_position),
                 (previous_rotation.inv()*rotation).as_rotvec()]


def fuse_increment(increments, weights):
    """Translation barycentre and intrinsic rotation mean; never quaternion averaging."""
    increments = np.asarray(increments)
    weights = np.asarray(weights)
    if weights.sum() <= 0:
        return np.zeros(6)
    weights = weights/weights.sum()
    translation = weights@increments[:, :3]
    rotations = Rotation.from_rotvec(increments[:, 3:])
    mean = rotations[int(np.argmax(weights))]
    for _ in range(12):
        step = weights@(mean.inv()*rotations).as_rotvec()
        mean = mean*Rotation.from_rotvec(step)
        if np.linalg.norm(step) < 1e-11:
            break
    return np.r_[translation, mean.as_rotvec()]


def integrate_increment(position, rotation, increment):
    return position+rotation.apply(increment[:3]), rotation*Rotation.from_rotvec(increment[3:])
