"""Validated experiment configuration and independent sensor controls."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Default sensor model outside the 30–70% corruption window is intentionally
# non-perfect. Noise units are per-sample standard deviations, not densities.
DEFAULTS = {
    "blur": 0.0, "illumination": 1.0, "feature_dropout": 0.0,
    "frame_dropout": 0.0, "pixel_noise": 0.6, "outlier_ratio": 0.0,
    "concentration": 0.0, "parallax_scale": 1.0,
    "gyro_noise": 0.002, "accel_noise": 0.025,
    "gyro_bias": 0.001, "accel_bias": 0.015, "bias_walk": 0.0001,
    "saturation": 100.0, "imu_dropout": 0.0, "scale_error": 0.0,
}
SCENARIOS = [
    ("clean", "Clean environment", "Nominal camera and IMU noise.", {}),
    ("low_light", "Low illumination", "Reduced feature detection and increased pixel uncertainty.", {"illumination": .15, "pixel_noise": 2.0}),
    ("motion_blur", "Heavy motion blur", "Angular-motion-dependent feature noise and loss.", {"blur": .8, "pixel_noise": 2.0}),
    ("occlusion", "Partial camera occlusion", "Most features disappear; remaining features become concentrated.", {"feature_dropout": .7, "concentration": .55}),
    ("dynamic", "Dynamic objects", "Incorrect correspondences contaminate the static map.", {"outlier_ratio": .4}),
    ("imu_drift", "IMU drift", "Increased bias and scale error in the inertial measurements.", {"gyro_bias": .025, "accel_bias": .18, "bias_walk": .005, "scale_error": .015}),
    ("vibration", "IMU vibration/noise", "High-frequency inertial noise.", {"gyro_noise": .05, "accel_noise": .7}),
    ("camera_dropout", "Camera frame dropout", "Intermittent camera loss requires inertial holdover.", {"frame_dropout": .85}),
    ("combined", "Combined sensor degradation", "Visual loss and inertial drift in the same sequence.", {"illumination": .35, "feature_dropout": .35, "outlier_ratio": .2, "pixel_noise": 2.5, "accel_bias": .1, "gyro_bias": .012, "frame_dropout": .12}),
    ("extreme", "Extreme tracking conditions", "Severe simultaneous degradation; accuracy is not guaranteed.", {"illumination": .1, "feature_dropout": .75, "frame_dropout": .5, "outlier_ratio": .5, "gyro_noise": .04, "accel_noise": .5, "accel_bias": .3, "imu_dropout": .15, "concentration": .65}),
]
PRESETS = [{"id": key, "name": name, "description": desc,
            "parameters": {**DEFAULTS, **params}}
           for key, name, desc, params in SCENARIOS]
PRESET_MAP = {x["id"]: x for x in PRESETS}


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    preset: str = "combined"
    seed: int = Field(default=42, ge=0, le=2147483647)
    duration: float = Field(default=8, ge=3, le=30)
    motion: Literal["loop", "pure_rotation", "low_parallax"] = "loop"
    parameters: dict[str, float] = Field(default_factory=dict)
    rigDegradation: Literal["staggered", "common"] = "staggered"
    cameraParameters: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("preset")
    @classmethod
    def valid_preset(cls, value):
        if value not in PRESET_MAP:
            raise ValueError("Unknown preset")
        return value

    @field_validator("parameters")
    @classmethod
    def valid_parameters(cls, values):
        import math
        ratios = {"blur", "illumination", "feature_dropout", "frame_dropout",
                  "outlier_ratio", "concentration", "parallax_scale", "imu_dropout", "scale_error"}
        for k, v in values.items():
            if k not in DEFAULTS:
                raise ValueError(f"Unknown sensor parameter: {k}")
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be finite and nonnegative")
            if k in ratios and v > 1:
                raise ValueError(f"{k} must be between 0 and 1")
            if k == "saturation" and v < .1:
                raise ValueError("saturation must be at least 0.1")
            if k not in ratios and v > 100:
                raise ValueError(f"{k} exceeds research prototype bound of 100")
        return values

    def resolved(self):
        return {**PRESET_MAP[self.preset]["parameters"], **self.parameters}

    @field_validator("cameraParameters")
    @classmethod
    def valid_camera_parameters(cls, values):
        visual = {"blur", "illumination", "feature_dropout", "frame_dropout",
                  "pixel_noise", "outlier_ratio", "concentration"}
        for camera, parameters in values.items():
            if camera not in {"front", "left", "right", "rear"}:
                raise ValueError(f"Unknown camera: {camera}")
            if set(parameters) - visual:
                raise ValueError("Camera overrides must be visual measurement parameters; motion and IMU controls are global.")
            cls.valid_parameters(parameters)
        return values
