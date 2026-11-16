"""A small trained NumPy neural gating network; no LLM/cloud dependencies."""
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).with_name("trained_gate.json")
EXPERT_IDS = ("front", "left", "right", "rear", "imu")
EXPERT_NAMES = ("Front camera", "Left camera", "Right camera", "Rear camera", "IMU propagation")
STAT_NAMES = ("features", "inliers", "log_rmse", "quality", "available",
              "translation_magnitude", "rotation_magnitude", "consensus_disagreement")
FEATURE_NAMES = [f"{expert}.{stat}" for expert in EXPERT_IDS for stat in STAT_NAMES] + [
    "previous_imu_tx", "previous_imu_ty", "previous_imu_tz",
    "previous_imu_rx", "previous_imu_ry", "previous_imu_rz", "imu_quality", "dt"]


def masked_softmax(logits, mask):
    logits, mask = np.asarray(logits), np.asarray(mask, dtype=bool)
    scores = np.where(mask, logits, -1e30)
    shifted = scores-np.max(scores, axis=-1, keepdims=True)
    values = np.exp(shifted)*mask
    return values/np.maximum(values.sum(axis=-1, keepdims=True), 1e-30)


class Gate:
    def __init__(self, parameters, mean, scale, report):
        self.parameters = {k: np.asarray(v, dtype=float) for k, v in parameters.items()}
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.report = report

    def predict(self, features, mask):
        x = np.clip((np.asarray(features)-self.mean)/self.scale, -8, 8)
        p = self.parameters
        h1 = np.tanh(x@p["w1"]+p["b1"])
        h2 = np.tanh(h1@p["w2"]+p["b2"])
        return masked_softmax(h2@p["w3"]+p["b3"], mask)

    def save(self, path=MODEL_PATH):
        parameters = {k: v.tolist() for k, v in self.parameters.items()}
        payload = {"parameters": parameters, "mean": self.mean.tolist(),
                   "scale": self.scale.tolist(), "report": self.report}
        signature = json.dumps({"parameters": parameters, "mean": payload["mean"], "scale": payload["scale"]},
                               sort_keys=True, separators=(",", ":")).encode()
        payload["report"]["modelHash"] = hashlib.sha256(signature).hexdigest()
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temp.replace(path)


def load_gate(path=MODEL_PATH):
    if not path.exists():
        raise RuntimeError("The MIXO-inspired gate has not been trained. Run python -m backend.mixo.train.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = json.dumps({k: payload[k] for k in ("parameters", "mean", "scale")},
                           sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(signature).hexdigest() != payload["report"].get("modelHash"):
        raise RuntimeError("Trained gate checksum mismatch")
    report = {**payload["report"], "command": "python -m backend.mixo.train"}
    return Gate(payload["parameters"], payload["mean"], payload["scale"], report)


def model_report():
    if not MODEL_PATH.exists():
        return {"status": "untrained", "source": "No trained gate artifact; no learned results are available."}
    return load_gate().report
