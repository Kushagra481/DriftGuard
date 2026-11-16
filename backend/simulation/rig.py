"""Four real pinhole camera streams with known, nonidentity rig extrinsics.

Only this simulator may use analytic motion. RigMeasurements deliberately has
no ground-truth poses. Front observations are the exact original sensor stream.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from backend.models import DEFAULTS
from .geometry import project
from .sensors import trajectory

CAMERAS = [
    ("front", "Front camera", 0., [0., 0., 0.]),
    ("left", "Left camera", -90., [-.08, 0., 0.]),
    ("right", "Right camera", 90., [.08, 0., 0.]),
    ("rear", "Rear camera", 180., [0., 0., -.08]),
]
WINDOWS = {"front": [.3, .7], "left": [.45, .85], "right": [.15, .55], "rear": [.35, .75]}


@dataclass
class RigCamera:
    id: str
    name: str
    rotation: Rotation  # camera -> body
    translation: np.ndarray  # camera origin in body
    observations: list

    def descriptor(self):
        return {"id": self.id, "name": self.name, "quaternion": self.rotation.as_quat().tolist(),
                "translation": self.translation.tolist()}


@dataclass
class RigMeasurements:
    landmarks: np.ndarray
    cameras: list
    fingerprint: str
    windows: dict


def generate_rig(config, primary, rng_namespace=0):
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 94721, rng_namespace]))
    # Additional room surfaces supply genuine left/right/rear fields of view.
    # The original front map and correspondences remain byte-for-byte intact.
    room = rng.uniform([-8., -3.8, -11.], [8., 3.8, 11.], (660, 3))
    room[:220, 0] = -7.
    room[220:440, 0] = 7.
    room[440:, 2] = -10.
    landmarks = np.vstack((primary.landmarks, room))
    times = primary.camera_times
    positions, rotations, _ = trajectory(times, config.motion, config.resolved()["parallax_scale"])
    cameras, windows = [], {}
    digest = hashlib.sha256(primary.fingerprint.encode())
    digest.update(landmarks.tobytes())
    for camera_index, (camera_id, name, yaw, translation) in enumerate(CAMERAS):
        # Per-camera RNG streams keep changes to one sensor from changing the
        # noise/dropout realization of another sensor in paired experiments.
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 94721, rng_namespace, camera_index+1]))
        extrinsic = Rotation.from_euler("y", yaw, degrees=True)
        offset = np.asarray(translation)
        bounds = WINDOWS[camera_id] if config.rigDegradation == "staggered" else [.3, .7]
        windows[camera_id] = [float(config.duration*b) for b in bounds]
        params = {**config.resolved(), **config.cameraParameters.get(camera_id, {})}
        if camera_id == "front":
            observations = primary.observations
        else:
            observations = []
            reference_count = 1
            for i, t in enumerate(times):
                active = windows[camera_id][0] <= t <= windows[camera_id][1]
                p = params if active else DEFAULTS
                position = positions[i] + rotations[i].apply(offset)
                rotation = rotations[i]*extrinsic
                uv, depth = project(landmarks, position, rotation)
                visible = (depth > .3) & (uv[:, 0] > 8) & (uv[:, 0] < 632) & (uv[:, 1] > 8) & (uv[:, 1] < 472)
                ids = np.flatnonzero(visible)
                pixels = uv[ids]
                if p["concentration"] > 0:
                    keep = np.linalg.norm((pixels-[320, 240])/[320, 240], axis=1) < 1.35*(1-.83*p["concentration"])
                    ids, pixels = ids[keep], pixels[keep]
                keep = rng.random(len(ids)) > np.clip(p["feature_dropout"]+.55*(1-p["illumination"])+.2*p["blur"], 0, .98)
                if rng.random() < p["frame_dropout"]:
                    keep[:] = False
                ids, pixels = ids[keep], pixels[keep]
                packet = min(int(t*200), len(primary.gyro)-1)
                while packet > 0 and not primary.imu_valid[packet]:
                    packet -= 1
                blur = float(np.clip(p["blur"]*(.4+3*np.linalg.norm(primary.gyro[packet])), 0, 1))
                sigma = p["pixel_noise"]*(1+2*blur+1.5*(1-p["illumination"]))
                pixels = pixels + rng.normal(0, sigma, pixels.shape)
                outliers = rng.random(len(ids)) < p["outlier_ratio"]
                pixels[outliers] += rng.normal(0, 65, (int(outliers.sum()), 2))
                reference_count = max(reference_count, len(ids))
                ids.flags.writeable = False
                pixels.flags.writeable = False
                observations.append({"ids": ids, "uv": pixels, "blur": blur,
                                     "dropout": 1-len(ids)/reference_count, "degraded": bool(active)})
        digest.update(extrinsic.as_quat().tobytes())
        digest.update(offset.tobytes())
        for obs in observations:
            digest.update(obs["ids"].tobytes())
            digest.update(obs["uv"].tobytes())
            digest.update(np.asarray([obs["blur"], obs["dropout"]], dtype=np.float64).tobytes())
        cameras.append(RigCamera(camera_id, name, extrinsic, offset, observations))
    landmarks.flags.writeable = False
    return RigMeasurements(landmarks, cameras, digest.hexdigest(), windows)
