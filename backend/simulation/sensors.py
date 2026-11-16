"""Reproducible calibrated camera/IMU simulator with no estimator coupling."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import project, GRAVITY
from backend.models import DEFAULTS


@dataclass
class Measurements:
    landmarks: np.ndarray
    camera_times: np.ndarray
    imu_times: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    imu_valid: np.ndarray
    saturated: np.ndarray
    observations: list
    fingerprint: str


@dataclass
class GroundTruth:
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation


def trajectory(t, motion="loop", scale=1.):
    """Analytic smooth path with p(0)=0 and v(0)=0, not estimator input."""
    w = .55
    a = scale * (0.015 if motion == "low_parallax" else 1.)
    if motion == "pure_rotation":
        a = 0.
    positions = np.column_stack([1.5*a*(1-np.cos(w*t)),
                                 .35*a*(1-np.cos(2*w*t)),
                                 .65*a*(1-np.cos(.75*w*t))])
    accel = np.column_stack([1.5*a*w*w*np.cos(w*t),
                            .35*a*4*w*w*np.cos(2*w*t),
                            .65*a*(.75*w)**2*np.cos(.75*w*t)])
    euler = np.column_stack([.10*np.sin(.7*t), .22*np.sin(.5*t), .09*np.sin(.6*t)])
    rotations = Rotation.from_euler("xyz", euler)
    return positions, rotations, accel


def generate(config):
    rng = np.random.default_rng(config.seed)
    params = config.resolved()
    it = np.arange(0, config.duration + 1e-8, 1/200)
    ct = np.arange(0, config.duration + 1e-8, 1/30)
    ip, ir, acceleration = trajectory(it, config.motion, params["parallax_scale"])
    cp, cr, _ = trajectory(ct, config.motion, params["parallax_scale"])
    # Landmarks on four indoor surfaces, deliberately not all coplanar.
    landmarks = rng.uniform([-6., -3.7, 4.5], [7., 3.7, 13.], (260, 3))
    landmarks[:90, 2] = 10.
    landmarks[90:130, 1] = 3.
    landmarks[130:170, 0] = -5.
    angular = np.zeros((len(it), 3))
    angular[:-1] = (ir[:-1].inv() * ir[1:]).as_rotvec() * 200
    angular[-1] = angular[-2]
    specific = ir.inv().apply(acceleration - GRAVITY)
    degraded_imu = (it >= config.duration*.3) & (it <= config.duration*.7)
    gyro = np.empty_like(angular)
    accel = np.empty_like(specific)
    valid = np.ones(len(it), dtype=bool)
    saturated = np.zeros(len(it), dtype=bool)
    bg = rng.normal(size=3) * DEFAULTS["gyro_bias"]
    ba = rng.normal(size=3) * DEFAULTS["accel_bias"]
    bias_direction_g = np.array([.6, -.4, .7])
    bias_direction_a = np.array([.8, .3, -.5])
    for i, active in enumerate(degraded_imu):
        p = params if active else DEFAULTS
        bg += rng.normal(0, p["bias_walk"] * np.sqrt(.005), 3)
        ba += rng.normal(0, p["bias_walk"] * np.sqrt(.005), 3)
        g_bias = bg + (p["gyro_bias"]-DEFAULTS["gyro_bias"])*bias_direction_g
        a_bias = ba + (p["accel_bias"]-DEFAULTS["accel_bias"])*bias_direction_a
        gyro[i] = angular[i]*(1+p["scale_error"]) + g_bias + rng.normal(0, p["gyro_noise"], 3)
        accel[i] = specific[i]*(1+p["scale_error"]) + a_bias + rng.normal(0, p["accel_noise"], 3)
        saturated[i] = np.any(np.abs(gyro[i]) > p["saturation"]) or np.any(np.abs(accel[i]) > p["saturation"])
        gyro[i] = np.clip(gyro[i], -p["saturation"], p["saturation"])
        accel[i] = np.clip(accel[i], -p["saturation"], p["saturation"])
        valid[i] = rng.random() >= p["imu_dropout"]
    observations = []
    previous_uv, previous_ids = None, None
    feature_reference = 1
    for i, t in enumerate(ct):
        active = config.duration*.3 <= t <= config.duration*.7
        p = params if active else DEFAULTS
        uv, depth = project(landmarks, cp[i], cr[i])
        visible = (depth > .3) & (uv[:, 0] > 8) & (uv[:, 0] < 632) & (uv[:, 1] > 8) & (uv[:, 1] < 472)
        ids = np.flatnonzero(visible)
        clean = uv[ids]
        # Feature concentration is a sensor selection effect, not a GOI weight.
        if p["concentration"] > 0 and len(ids):
            radius = np.linalg.norm((clean-[320, 240])/[320, 240], axis=1)
            keep = radius < 1.35*(1-.83*p["concentration"])
            ids, clean = ids[keep], clean[keep]
        keep = rng.random(len(ids)) > np.clip(p["feature_dropout"]+.55*(1-p["illumination"])+.20*p["blur"], 0, .98)
        if rng.random() < p["frame_dropout"]:
            keep[:] = False
        missing_ids, missing_uv = ids[~keep], clean[~keep]
        ids, clean = ids[keep], clean[keep]
        # A noisy inertial exposure-status indicator, not privileged true motion.
        angular_speed = np.linalg.norm(gyro[min(int(round(t*200)), len(gyro)-1)])
        blur = float(np.clip(p["blur"]*(.4 + 3*angular_speed), 0, 1))
        sigma = p["pixel_noise"] * (1+2*blur+1.5*(1-p["illumination"]))
        measured = clean + rng.normal(0, sigma, clean.shape)
        injected = rng.random(len(ids)) < p["outlier_ratio"]
        measured[injected] += rng.normal(0, 65, (int(injected.sum()), 2))
        # Measured optical displacement (rotation included); true parallax
        # is computed later from estimated bearings after rotation removal.
        optical_flow = 0.
        if previous_ids is not None:
            _, aidx, bidx = np.intersect1d(ids, previous_ids, return_indices=True)
            if len(aidx):
                optical_flow = float(np.median(np.linalg.norm(measured[aidx]-previous_uv[bidx], axis=1)))
        feature_reference = max(feature_reference, len(ids))
        observations.append({
            "ids": ids, "uv": measured, "missing_ids": missing_ids, "missing_uv": missing_uv,
            "blur": blur, "degraded": bool(active), "optical_flow": optical_flow,
            # Available sensor status metadata; not true pose or injected labels.
            "illumination": p["illumination"], "dropout": float(1-len(ids)/feature_reference),
        })
        previous_ids, previous_uv = ids, measured
    # Freeze actual arrays: all estimators must receive exactly this sequence.
    digest = hashlib.sha256()
    for x in (landmarks, ct, it, gyro, accel, valid, saturated):
        digest.update(x.tobytes())
        x.flags.writeable = False
    for obs in observations:
        digest.update(obs["uv"].tobytes())
        digest.update(obs["ids"].tobytes())
        obs["uv"].flags.writeable = False
        obs["ids"].flags.writeable = False
    return Measurements(landmarks, ct, it, accel, gyro, valid, saturated,
                        observations, digest.hexdigest()), GroundTruth(ct, cp, cr)
