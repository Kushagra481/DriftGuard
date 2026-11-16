"""Sensor health from observations/status, not hidden simulation truth."""
import numpy as np


def visual_health(obs, pose, geometry):
    if pose is None:
        return 0.
    count = min(len(obs["ids"])/65, 1.)
    coverage = geometry["coverage"]
    conditioning = np.clip(1/np.log10(max(geometry["conditionNumber"], 10)), .05, 1)
    parallax = .7+.3*min(geometry["parallax"]/.015, 1.)
    return float(np.clip(count * (.5+.5*coverage) * pose.ratio
                         * np.exp(-pose.rmse/4) * (1-.5*obs["blur"])
                         * (1-.3*obs["dropout"]) * parallax
                         * (.8+.2*conditioning), 0, 1))


def inertial_health(gyro, accel, valid, saturated, filter_state):
    missing = 1-float(np.mean(valid)) if len(valid) else 1.
    sat = float(np.mean(saturated[valid])) if np.any(valid) else 0.
    gyro, accel = gyro[valid], accel[valid]
    # First-difference high-frequency noise estimate is observable; real motion
    # is not subtracted using privileged ground truth.
    if len(gyro) > 1:
        gn = float(np.sqrt(np.mean(np.diff(gyro, axis=0)**2)/2))
        an = float(np.sqrt(np.mean(np.diff(accel, axis=0)**2)/2))
    else:
        gn, an = .002, .025
    bias = np.linalg.norm(filter_state.bg)/.08 + np.linalg.norm(filter_state.ba)/.5
    residual_penalty = min(filter_state.last_mahalanobis/100., 2.)
    quality = (1-missing)*(1-sat)*np.exp(-2*gn-.6*an-.2*bias-.05*residual_penalty)
    return float(np.clip(quality, .005, 1.))
