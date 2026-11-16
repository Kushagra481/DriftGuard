"""Unaligned, synchronized metric-world evaluation; ground truth stays here."""
import numpy as np
from scipy.spatial.transform import Rotation


def compute_metrics(method_id, name, positions, rotations, truth, available, latency, recovery, keyframes):
    positions = np.asarray(positions)
    difference = positions-truth.positions
    errors = np.linalg.norm(difference, axis=1)
    rotations = Rotation.from_quat(np.asarray(rotations))
    angular_error = np.rad2deg((truth.rotations.inv()*rotations).magnitude())
    # SE(3) relative translation error with a one-second comparison interval.
    lag = min(30, len(positions)-1)
    est_delta = rotations[:-lag].inv().apply(positions[lag:]-positions[:-lag])
    gt_delta = truth.rotations[:-lag].inv().apply(truth.positions[lag:]-truth.positions[:-lag])
    rpe = np.linalg.norm(est_delta-gt_delta, axis=1)
    length = np.linalg.norm(np.diff(truth.positions, axis=0), axis=1).sum()
    return {
        "id": method_id, "name": name, "ate": float(np.sqrt(np.mean(errors**2))),
        "rpe": float(np.sqrt(np.mean(rpe**2))), "translationError": float(errors[-1]),
        "rotationError": float(np.sqrt(np.mean(angular_error**2))),
        "drift": float(100*errors[-1]/max(length, 1e-9)) if length > 1e-6 else 0.,
        "availability": float(100*np.mean(available)),
        "meanLatency": float(np.mean(latency)), "medianLatency": float(np.median(latency)),
        "p95Latency": float(np.percentile(latency, 95)),
        "recoveryDuration": float(np.mean(recovery)) if recovery else None,
        "keyframes": int(keyframes),
    }, errors
