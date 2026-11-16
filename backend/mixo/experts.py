"""Measurement-only expert extraction shared by offline training and inference.

No GroundTruth import, simulator trajectory, corruption labels or config values
are available here. Missing frames cannot become one-frame multi-frame deltas.
"""
from dataclasses import dataclass
import hashlib
import time
import cv2
import numpy as np
from backend.estimation import ESKF, estimate_pose
from backend.estimation.health import inertial_health
from .gate import EXPERT_IDS, EXPERT_NAMES
from .geometry import camera_to_body, relative_increment


@dataclass
class ExpertSequence:
    features: np.ndarray
    increments: np.ndarray
    masks: np.ndarray
    diagnostics: list
    views: list
    qualities: np.ndarray
    variances: np.ndarray
    latencies: np.ndarray
    initial_position: np.ndarray
    initial_rotation: object


def extract_experts(measurements, rig, include_views=True):
    """Four independent PnP odometry experts plus independent IMU propagation."""
    m = measurements
    cv2.setNumThreads(1)
    # Stable across training and API call ordering (OpenCV has global RNG).
    cv2.setRNGSeed(int(rig.fingerprint[:7], 16))
    obs0 = m.observations[0]
    initial = estimate_pose(m.landmarks, obs0["ids"], obs0["uv"])
    if initial is None:
        raise ValueError("MIXO requires the shared first-frame PnP initialization.")
    imu = ESKF(initial.position, initial.rotation, adaptive=True)
    previous_camera = [None]*4
    previous_imu_increment = np.zeros(6)
    held_g, held_a = m.gyro[0].copy(), m.accel[0].copy()
    features, increments, masks, diagnostics, views = [], [], [], [], []
    qualities, variances, latencies = [], [], []
    for frame_index, t in enumerate(m.camera_times):
        frame_start = time.perf_counter()
        left = float(m.camera_times[frame_index-1]) if frame_index else 0.
        dt = float(t-left)
        a = max(0, int(left*200))
        b = min(len(m.imu_times), int(t*200)+1)
        iq = inertial_health(m.gyro[a:b], m.accel[a:b], m.imu_valid[a:b], m.saturated[a:b], imu)
        imu_p, imu_r = imu.p.copy(), imu.r
        old_var = np.r_[np.trace(imu.P[:3, :3]), np.trace(imu.P[6:9, 6:9])]
        tic = time.perf_counter()
        while left < t-1e-10:
            k = min(int(np.floor(left*200+1e-7)), len(m.imu_times)-1)
            right = min(float(t), (k+1)/200)
            if m.imu_valid[k]:
                held_g, held_a = m.gyro[k], m.accel[k]
            imu.propagate(held_g, held_a, right-left, iq)
            left = right
        imu_latency = (time.perf_counter()-tic)*1000
        imu_delta = relative_increment(imu_p, imu_r, imu.p, imu.r)
        new_var = np.r_[np.trace(imu.P[:3, :3]), np.trace(imu.P[6:9, 6:9])]
        frame_variance = np.zeros((5, 2))
        frame_variance[4] = np.maximum(new_var-old_var, [1e-8, 1e-9])
        frame_deltas, frame_mask = np.zeros((5, 6)), np.zeros(5, dtype=bool)
        frame_deltas[4], frame_mask[4] = imu_delta, True
        frame_diags, frame_views = [], []
        for c, camera in enumerate(rig.cameras):
            obs = camera.observations[frame_index]
            tic = time.perf_counter()
            signature = hashlib.sha256(obs["ids"].tobytes()+obs["uv"].tobytes()).hexdigest()
            cv2.setRNGSeed(int(signature[:7], 16))
            visual = estimate_pose(rig.landmarks, obs["ids"], obs["uv"])
            available, reason, quality = False, "No robust camera pose", 0.
            current = None
            if visual is not None:
                current = camera_to_body(visual.position, visual.rotation, camera.rotation, camera.translation)
                quality = float(min(len(obs["ids"])/65, 1)*visual.ratio*np.exp(-visual.rmse/4)
                                *(1-.5*obs["blur"])*(1-.3*obs["dropout"]))
                if previous_camera[c] is None:
                    reason = "Initialization: need consecutive valid frames"
                else:
                    delta = relative_increment(*previous_camera[c], *current)
                    # Measurement-based physical safety check, not a GT error gate.
                    plausible = (np.linalg.norm(delta[:3]) <= 10*dt
                                 and np.linalg.norm(delta[3:]) <= 10*dt)
                    available = bool(len(obs["ids"]) >= 8 and visual.ratio >= .3 and plausible)
                    reason = "Consecutive robust poses" if available else "Rejected: insufficient support or >10 m/s / 10 rad/s increment"
                    if available:
                        frame_deltas[c] = delta
                pixel_sigma = max(visual.rmse, .5)
                frame_variance[c] = [6*(pixel_sigma*7/460)**2, 6*(pixel_sigma/460)**2]
            previous_camera[c] = current
            frame_mask[c] = available
            frame_diags.append({
                "id": camera.id, "name": camera.name, "available": available,
                "featureCount": len(obs["ids"]), "inlierRatio": visual.ratio if visual else 0.,
                "reprojectionError": visual.rmse if visual else 0., "quality": quality,
                "increment": frame_deltas[c].tolist(), "weight": 0., "reason": reason,
                "latency": (time.perf_counter()-tic)*1000,
            })
            if include_views:
                # Preview only is evenly subsampled. Every correspondence is
                # still used by PnP, metrics and gate features.
                chosen = np.linspace(0, len(obs["ids"])-1, min(80, len(obs["ids"])), dtype=int)
                frame_views.append({
                    "id": camera.id, "name": camera.name, "width": 640, "height": 480,
                    "featureCount": len(obs["ids"]), "sampled": len(chosen) < len(obs["ids"]),
                    "features": [{"id": int(obs["ids"][j]), "u": float(obs["uv"][j, 0]),
                                  "v": float(obs["uv"][j, 1]),
                                  "residual": float(visual.residuals[j]) if visual else 0.,
                                  "outlier": bool(not visual.inliers[j]) if visual else False}
                                 for j in chosen],
                })
        fresh = bool(np.any(m.imu_valid[a:b]))
        frame_diags.append({
            "id": "imu", "name": EXPERT_NAMES[4], "available": True,
            "featureCount": 0, "inlierRatio": 0., "reprojectionError": 0., "quality": iq,
            "increment": imu_delta.tolist(), "weight": 0.,
            "reason": ("Fresh inertial packets" if fresh else "No fresh packet: last-sample holdover")
                      + ("; low sensor health" if iq < .3 else ""),
            "latency": imu_latency,
        })
        consensus = np.median(frame_deltas[frame_mask], axis=0)
        stats = []
        for c, diag in enumerate(frame_diags):
            disagreement = np.linalg.norm((frame_deltas[c]-consensus)/np.array([.03]*3+[.02]*3))
            stats.extend([diag["featureCount"]/200, diag["inlierRatio"],
                          np.log1p(diag["reprojectionError"])/4, diag["quality"], float(frame_mask[c]),
                          np.linalg.norm(frame_deltas[c, :3])/.03, np.linalg.norm(frame_deltas[c, 3:])/.02,
                          min(disagreement, 20.)])
        stats.extend((previous_imu_increment/np.array([.03]*3+[.02]*3)).tolist()+[iq, dt])
        features.append(stats)
        increments.append(frame_deltas)
        masks.append(frame_mask)
        diagnostics.append(frame_diags)
        views.append(frame_views)
        qualities.append(iq)
        variances.append(frame_variance)
        latencies.append((time.perf_counter()-frame_start)*1000)
        previous_imu_increment = imu_delta
    return ExpertSequence(np.asarray(features), np.asarray(increments), np.asarray(masks),
                          diagnostics, views, np.asarray(qualities), np.asarray(variances),
                          np.asarray(latencies), initial.position.copy(), initial.rotation)
