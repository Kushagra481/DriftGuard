"""Execute five estimators against one frozen measurement sequence.

No estimator receives GroundTruth. Truth enters only after a pose has been
produced, for frame error displays and final evaluation metrics.
"""
import time
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from backend.simulation import generate
from backend.simulation.geometry import pose_dict, project
from backend.estimation import ESKF, estimate_pose
from backend.estimation.health import visual_health, inertial_health
from backend.estimation.recovery import Recovery
from backend.goi import diagnose, KeyframePolicy
from backend.goi.influence import empty_geometry
from .metrics import compute_metrics

METHODS = {
    "imu": "IMU-only", "vision": "Vision-only", "fixed": "Fixed-weight VIO",
    "adaptive": "Adaptive VIO · no GOI", "driftguard": "DriftGuard XR",
}
ABLATIONS = {
    "diagnostics_only": "Adaptive + GOI diagnostics only",
    "geometry_fixed": "Adaptive + geometry covariance · fixed interval",
    "keyframes_basic": "Adaptive + observability keyframes · basic covariance",
}
ALL_METHODS = {**METHODS, **ABLATIONS}


def run_estimation(measurements):
    """Pure estimation boundary. Takes measurements, never ground truth."""
    m = measurements
    first = m.observations[0]
    initial = estimate_pose(m.landmarks, first["ids"], first["uv"])
    if initial is None:
        raise ValueError("Unable to initialize from the first calibrated camera frame")
    filters = {key: ESKF(initial.position, initial.rotation, adaptive=key not in ("imu", "fixed"))
               for key in ("imu", "fixed", "adaptive", "driftguard", *ABLATIONS)}
    vision_p, vision_r = initial.position.copy(), initial.rotation
    policy = KeyframePolicy()
    recoveries = {key: Recovery() for key in filters}
    trajectories = {key: {"p": [], "q": [], "available": [], "latency": []} for key in ALL_METHODS}
    frames, details = [], []
    previous = None
    imu_index = 0
    held_gyro, held_accel = m.gyro[0].copy(), m.accel[0].copy()
    fixed_count = 0
    for frame_index, camera_time in enumerate(m.camera_times):
        obs = m.observations[frame_index]
        next_index = min(int(np.searchsorted(m.imu_times, camera_time, side="right")), len(m.imu_times))
        start_index = max(imu_index-1, 0)
        iq = inertial_health(m.gyro[start_index:next_index], m.accel[start_index:next_index],
                             m.imu_valid[start_index:next_index], m.saturated[start_index:next_index],
                             filters["driftguard"])
        timings = {key: 0. for key in ALL_METHODS}
        # Integrate each camera-to-camera interval using the same IMU packets.
        left_time = m.camera_times[frame_index-1] if frame_index else 0.
        while left_time < camera_time-1e-10:
            idx = min(int(np.floor(left_time*200+1e-7)), len(m.imu_times)-1)
            right_time = min(camera_time, (idx+1)/200)
            if m.imu_valid[idx]:
                held_gyro, held_accel = m.gyro[idx], m.accel[idx]
            dt = float(right_time-left_time)
            for key, filt in filters.items():
                tic = time.perf_counter()
                filt.propagate(held_gyro, held_accel, dt, iq)
                timings[key] += (time.perf_counter()-tic)*1000
            left_time = right_time
        imu_index = next_index
        tic = time.perf_counter()
        visual = estimate_pose(m.landmarks, obs["ids"], obs["uv"])
        vision_latency = (time.perf_counter()-tic)*1000
        for key in ("vision", "fixed", "adaptive", "driftguard", *ABLATIONS):
            timings[key] += vision_latency
        if visual is not None:
            vision_p, vision_r = visual.position, visual.rotation
        tic = time.perf_counter()
        # Diagnose at the measured visual pose where available, predicted state
        # otherwise. No GT projections are used in quality/influence scoring.
        gpos = visual.position if visual else filters["driftguard"].p
        grot = visual.rotation if visual else filters["driftguard"].r
        geometry = diagnose(m.landmarks, obs["ids"], obs["uv"], gpos, grot, previous)
        qv = visual_health(obs, visual, geometry)
        # Adaptive no-GOI ablation excludes curvature, coverage and keyframe utility.
        qbasic = (float(min(len(obs["ids"])/65, 1)*visual.ratio
                  * np.exp(-visual.rmse/4)*(1-.5*obs["blur"])*(1-.3*obs["dropout"]))
                  if visual is not None else 0.)
        chosen, gain, reason = policy.choose(float(camera_time), geometry, qv)
        geometry_latency = (time.perf_counter()-tic)*1000
        for key in ("driftguard", *ABLATIONS):
            timings[key] += geometry_latency
        fixed_keyframe = frame_index % 9 == 0 and visual is not None
        if fixed_keyframe:
            fixed_count += 1
        accepted = {key: None for key in filters}
        if visual is not None:
            if fixed_keyframe:
                tic = time.perf_counter()
                accepted["fixed"] = filters["fixed"].update(
                    visual.position, visual.rotation, np.diag([.035**2]*3+[.012**2]*3), gate=False)
                timings["fixed"] += (time.perf_counter()-tic)*1000
            for key, enabled, quality in (
                    ("adaptive", fixed_keyframe, qbasic), ("driftguard", chosen, qv),
                    ("diagnostics_only", fixed_keyframe, qbasic),
                    ("geometry_fixed", fixed_keyframe, qv), ("keyframes_basic", chosen, qbasic)):
                if enabled and quality > .035:
                    tic = time.perf_counter()
                    # Heteroscedastic covariance uses residual scale and sensor
                    # health, not a ground-truth error or tuned result target.
                    scale = np.clip(1/max(quality, .03), 1, 35)
                    residual_scale = max(1., visual.rmse/.9)
                    covariance = np.diag([(.016*residual_scale)**2]*3
                                         + [(.006*residual_scale)**2]*3)*scale
                    accepted[key] = filters[key].update(visual.position, visual.rotation, covariance)
                    timings[key] += (time.perf_counter()-tic)*1000
            previous = (obs["ids"], obs["uv"], visual.rotation)
        sigma, theta_sigma = filters["driftguard"].uncertainty()
        states = {}
        for key, filt in filters.items():
            health = qbasic if key in ("adaptive", "diagnostics_only", "keyframes_basic") else qv
            if key == "imu":
                health = 0.
            states[key] = recoveries[key].step(float(camera_time), health, iq, filt.uncertainty()[0], accepted[key])
        poses = {key: pose_dict(filt.p, filt.r) for key, filt in filters.items()}
        poses["vision"] = pose_dict(vision_p, vision_r)
        for key in ALL_METHODS:
            tr = trajectories[key]
            tr["p"].append(poses[key]["position"])
            tr["q"].append(poses[key]["quaternion"])
            usable = visual is not None if key == "vision" else filters[key].uncertainty()[0] <= .75
            # Availability is declared capability, NOT GT-threshold accuracy.
            tr["available"].append(bool(usable))
            tr["latency"].append(float(timings[key]))
        state, cause = states["driftguard"]
        frame = {
            "index": frame_index, "time": float(camera_time),
            "poses": {key: poses[key] for key in METHODS},
            "state": state, "cause": cause, "visualQuality": qv, "imuQuality": iq,
            "inlierRatio": visual.ratio if visual else 0.,
            "reprojectionError": visual.rmse if visual else 0.,
            "featureCount": len(obs["ids"]), "positionUncertainty": sigma,
            "orientationUncertainty": theta_sigma,
            "eigenvalues": geometry["eigenvalues"], "conditionNumber": geometry["conditionNumber"],
            "translationObservability": geometry["translationObservability"],
            "rotationObservability": geometry["rotationObservability"], "keyframe": chosen,
            "fixedKeyframe": bool(fixed_keyframe), "keyframeReason": reason,
            "informationGain": gain, "goiMean": geometry["goiMean"],
            "latency": timings["driftguard"], "degraded": obs["degraded"],
        }
        missing_pred, _ = project(m.landmarks[obs["missing_ids"]], gpos, grot)
        features = geometry["features"] + [{
            "id": int(fid), "u": float(uv[0]), "v": float(uv[1]),
            "predictedU": float(pred[0]), "predictedV": float(pred[1]),
            "landmark": m.landmarks[fid].tolist(), "residual": 0., "weight": 0.,
            "goi": 0., "outlier": False, "missing": True,
        } for fid, uv, pred in zip(obs["missing_ids"], obs["missing_uv"], missing_pred)]
        details.append({
            "features": features, "weakestDirection": geometry["weakestDirection"],
            "observableRank": geometry["observableRank"], "coverage": geometry["coverage"],
            "parallax": geometry["parallax"], "blur": obs["blur"],
        })
        frames.append(frame)
    recovery_stats = {key: obj.durations for key, obj in recoveries.items()}
    recovery_stats["vision"] = visual_recovery_durations(trajectories["vision"]["available"], m.camera_times)
    counts = {"imu": 0, "vision": sum(trajectories["vision"]["available"]),
              "fixed": fixed_count, "adaptive": filters["adaptive"].accepted_updates,
              "driftguard": filters["driftguard"].accepted_updates}
    counts.update({key: filters[key].accepted_updates for key in ABLATIONS})
    return frames, details, trajectories, recovery_stats, counts


def visual_recovery_durations(available, times):
    durations, lost = [], None
    for ok, t in zip(available, times):
        if not ok and lost is None:
            lost = t
        if ok and lost is not None:
            durations.append(float(t-lost))
            lost = None
    return durations


def influence_distributions(frames, details):
    """Shared logarithmic bins over real per-feature influence samples."""
    phases = {"before": [], "during": [], "after": []}
    first = next((i for i, f in enumerate(frames) if f["degraded"]), len(frames))
    last = max((i for i, f in enumerate(frames) if f["degraded"]), default=-1)
    for i, detail in enumerate(details):
        phase = "before" if i < first else "during" if i <= last else "after"
        phases[phase].extend(f["goi"] for f in detail["features"]
                             if not f["missing"] and f["goi"] > 0)
    values = [v for samples in phases.values() for v in samples]
    if not values:
        return []
    low = np.floor(np.log10(max(min(values), 1e-14)))
    high = max(low+1, np.ceil(np.log10(max(values))))
    edges = np.logspace(low, high, 11)
    histograms = {key: np.histogram(samples, edges)[0] for key, samples in phases.items()}
    return [{"label": f"{edges[i]:.1e}–{edges[i+1]:.1e}",
             "lower": float(edges[i]), "upper": float(edges[i+1]),
             **{phase: int(counts[i]) for phase, counts in histograms.items()}}
            for i in range(len(edges)-1)]


def run_experiment(config):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(config.seed)
    started = time.perf_counter()
    measurements, truth = generate(config)
    frames, details, trajectories, recovery, counts = run_estimation(measurements)
    metrics, ablation_metrics = [], []
    for key, name in ALL_METHODS.items():
        tr = trajectories[key]
        metric, errors = compute_metrics(key, name, tr["p"], tr["q"], truth,
                                         tr["available"], tr["latency"], recovery[key], counts[key])
        if key in METHODS:
            metrics.append(metric)
            for i, frame in enumerate(frames):
                frame.setdefault("errors", {})[key] = float(errors[i])
        else:
            ablation_metrics.append(metric)
    for i, frame in enumerate(frames):
        frame["truth"] = pose_dict(truth.positions[i], truth.rotations[i])
    resolved = config.model_dump()
    resolved["parameters"] = config.resolved()
    result = {
        "id": str(uuid.uuid4()), "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": resolved, "frames": frames, "metrics": metrics,
        "landmarks": measurements.landmarks.tolist(),
        "camera": {"width": 640, "height": 480, "fx": 460., "fy": 460., "cx": 320., "cy": 240.},
        "metadata": {
            "source": "Synthetic calibrated known-landmark pose benchmark; NOT general-purpose SLAM.",
            "cameraHz": 30, "imuHz": 200, "imuPackets": len(measurements.imu_times),
            "cameraFrames": len(frames), "measurementFingerprint": measurements.fingerprint,
            "methodFingerprints": {key: measurements.fingerprint for key in METHODS},
            "ablationFingerprints": {key: measurements.fingerprint for key in ABLATIONS},
            "ablationMetrics": ablation_metrics,
            "goiDistributions": influence_distributions(frames, details),
            "degradationWindow": [config.duration*.3, config.duration*.7],
            "elapsedSeconds": time.perf_counter()-started,
            "cameraToImuExtrinsics": {"rotationXyzw": [0, 0, 0, 1], "translationMeters": [0, 0, 0]},
            "initialization": "All methods share a first-frame PnP pose and zero-velocity prior. No ground-truth initialization.",
            "goi": "GOI-inspired observable-subspace deletion influence; not a claim to reproduce a specific published GOI.",
            "metric": "Camera-centred SE(3), translation normalized by L=1 m, rotation in radians.",
            "availability": "Vision: a valid RANSAC pose; filters: position sigma <=0.75 m. Availability is not a ground-truth accuracy guarantee.",
            "recovery": "Completed sensor-health degradation-to-normal episodes; null means no completed recovery, not zero.",
            "latency": "Measured wall-clock ms per camera interval, including shared PnP for visual methods and GOI for DriftGuard; excludes simulation, serialization and network.",
            "ablations": [
                "Fixed-weight VIO: fixed covariance, fixed 9-frame updates, no innovation gate.",
                "Adaptive VIO: adaptive sensor covariance and consistency gate, fixed 9-frame updates, no geometry diagnostics.",
                "Full DriftGuard: geometry-aware covariance, influence diagnostics, observability-aware keyframe policy.",
                "Diagnostics-only vs adaptive: identical estimates, with measured diagnostic computation overhead.",
                "Geometry covariance vs adaptive: same update frames, isolates geometry-conditioned visual covariance.",
                "Observability keyframes vs adaptive: same basic covariance, isolates update selection.",
            ],
            "limitations": [
                "Simulated correspondences with a known metric map; no real image frontend, mapping or loop closure.",
                "Camera and IMU are collocated with identity extrinsics; no calibration or time-offset estimation.",
                "Pure rotation is still pose-observable with a known 3D map; it is not monocular triangulation degeneracy.",
                "Diagonal adaptive visual covariance is heuristic; PnP errors are not fully Gaussian.",
                "Fixed and adaptive-no-GOI use 9-frame update policies; full method uses an information-based variable update budget.",
                "GOI diagnostics alone do not improve pose; compare the covariance and keyframe isolation ablations.",
                "Zero reprojection RMSE on frames without a valid visual pose means unavailable; inspect featureCount and inlierRatio.",
                "Drift percent is undefined for zero-length pure-rotation paths and represented as 0; inspect absolute error instead.",
                "Reduced-parallax scaling changes the whole continuous trajectory, not a discontinuous corruption-window jump.",
            ],
        },
    }
    return result, details
