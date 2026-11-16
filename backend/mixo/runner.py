"""Learned and same-rig baseline inference, strictly measurement-only."""
import time
import numpy as np
from backend.estimation.recovery import Recovery
from backend.simulation.geometry import pose_dict
from .experts import extract_experts
from .gate import EXPERT_IDS, load_gate
from .geometry import fuse_increment, integrate_increment

RIG_METHODS = {"mixo": "MIXO-inspired 6-DoF",
               "mixo_equal": "Same rig · uniform weights", "mixo_match": "Same rig · most matches"}


def baseline_weights(sequence, index, method):
    mask = sequence.masks[index]
    if method == "mixo_equal":
        return mask/np.maximum(mask.sum(), 1)
    scores = np.array([x["featureCount"] for x in sequence.diagnostics[index]], dtype=float)
    scores[~mask] = -1.
    # MATCH chooses the highest-supported valid camera, IMU if all are absent.
    weights = np.zeros(5)
    weights[int(np.argmax(scores))] = 1.
    return weights


def run_mixo(measurements, rig, gate=None):
    gate = load_gate() if gate is None else gate
    seq = extract_experts(measurements, rig)
    trajectories, diagnostic_frames, recoveries, counts = {}, [], {}, {}
    for method in RIG_METHODS:
        p, r = seq.initial_position.copy(), seq.initial_rotation
        var_p, var_r = .03**2*3, .015**2*3
        recovery = Recovery()
        tr = {"p": [], "q": [], "available": [], "latency": []}
        count = 0
        for i, t in enumerate(measurements.camera_times):
            tic = time.perf_counter()
            weights = (gate.predict(seq.features[i], seq.masks[i]) if method == "mixo"
                       else baseline_weights(seq, i, method))
            delta = fuse_increment(seq.increments[i], weights)
            p, r = integrate_increment(p, r, delta)
            spread = seq.increments[i]-delta
            # Conservative diagnostic propagation, NOT a calibrated Kalman
            # posterior. Shared map/temporal correlations remain unmodelled.
            vp = float(weights@np.sum(spread[:, :3]**2, axis=1))
            vr = float(weights@np.sum(spread[:, 3:]**2, axis=1))
            var_p += float((weights**2)@seq.variances[i, :, 0])+vp
            var_r += float((weights**2)@seq.variances[i, :, 1])+vr
            camera_valid = bool(np.any(seq.masks[i, :4]))
            if not camera_valid:
                # Unmodelled bias/held packets accumulate during visual loss.
                dt = float(t-measurements.camera_times[i-1]) if i else 0.
                var_p += (.03+1-seq.qualities[i])**2*dt**2
                var_r += (.003+.1*(1-seq.qualities[i]))**2*dt
            sigma, sigma_r = float(np.sqrt(var_p)), float(np.rad2deg(np.sqrt(var_r)))
            quality = max((d["quality"] for d in seq.diagnostics[i][:4] if d["available"]), default=0.)
            visual_used = camera_valid and float(weights[:4].sum()) > .02
            state, cause = recovery.step(float(t), quality, float(seq.qualities[i]), sigma,
                                         True if visual_used else None)
            if (not camera_valid and seq.qualities[i] < .15) or sigma_r > 15:
                state, cause = "TRACKING_UNCERTAIN", "No reliable visual increment and degraded inertial holdover, or large angular uncertainty."
            count += int(visual_used)
            pose = pose_dict(p, r)
            latency = float(seq.latencies[i]+(time.perf_counter()-tic)*1000)
            tr["p"].append(pose["position"])
            tr["q"].append(pose["quaternion"])
            tr["available"].append(bool(sigma <= .75 and sigma_r <= 15 and not (not camera_valid and seq.qualities[i] < .15)))
            tr["latency"].append(latency)
            if method == "mixo":
                experts = [{**diag, "weight": float(weights[c])} for c, diag in enumerate(seq.diagnostics[i])]
                diagnostic_frames.append({
                    "index": i, "time": float(t), "weights": dict(zip(EXPERT_IDS, weights.tolist())),
                    "experts": experts, "entropy": float(-np.sum(weights*np.log(np.maximum(weights, 1e-30)))),
                    "state": state, "cause": cause, "positionUncertainty": sigma,
                    "orientationUncertainty": sigma_r, "disagreement": float(np.sqrt(vp+vr)),
                    "activeExperts": int(seq.masks[i].sum()), "latency": latency,
                    "fusedIncrement": delta.tolist(), "pose": pose,
                })
        trajectories[method], recoveries[method], counts[method] = tr, recovery.durations, count
    return diagnostic_frames, seq.views, trajectories, recoveries, counts
