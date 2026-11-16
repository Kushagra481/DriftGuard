"""Reproducible offline supervised training: python -m backend.mixo.train.

Ground truth is imported ONLY here for offline labels, never in inference.
Training minimizes scaled body-frame translation/rotation-vector increment MSE.
Inference uses the intrinsic SO(3) mean, not this small-increment approximation.
"""
import copy
import json
import time
import cv2
import numpy as np
from backend.models import ExperimentInput, PRESETS
from backend.simulation.sensors import generate
from backend.simulation.rig import generate_rig
from .experts import extract_experts
from .gate import Gate, FEATURE_NAMES, masked_softmax, MODEL_PATH
from .geometry import relative_increment
from .runner import baseline_weights

TARGET_SCALE = np.array([.03]*3+[.02]*3)
SPLITS = {"training": (11001, 32, 1101), "validation": (22001, 8, 2202),
          "test": (33001, 8, 3303)}


def dataset(split):
    start, count, namespace = SPLITS[split]
    features, estimates, masks, labels, match = [], [], [], [], []
    fingerprints = []
    for index in range(count):
        seed = start+index
        rng = np.random.default_rng(seed)
        preset = PRESETS[index % len(PRESETS)]["id"]
        parameters = {}
        if index % 8 == 7:
            parameters = {"frame_dropout": 1., "imu_dropout": .25, "gyro_noise": .025}
        overrides = {camera: {"pixel_noise": float(rng.uniform(.5, 5)),
                              "frame_dropout": float(rng.choice([0., .15, .5, 1.])),
                              "outlier_ratio": float(rng.uniform(0, .5))}
                     for camera in ("left", "right", "rear")}
        config = ExperimentInput(seed=seed, duration=3.+index % 3, preset=preset,
                                 motion=("loop", "loop", "pure_rotation", "low_parallax")[index % 4],
                                 parameters=parameters, cameraParameters=overrides,
                                 rigDegradation="common" if index % 2 else "staggered")
        m, truth = generate(config, rng_namespace=namespace)
        rig = generate_rig(config, m, rng_namespace=namespace)
        seq = extract_experts(m, rig, include_views=False)
        target = np.zeros((len(truth.times), 6))
        for j in range(1, len(truth.times)):
            target[j] = relative_increment(truth.positions[j-1], truth.rotations[j-1],
                                           truth.positions[j], truth.rotations[j])
        features.append(seq.features)
        estimates.append(seq.increments/TARGET_SCALE)
        masks.append(seq.masks)
        labels.append(target/TARGET_SCALE)
        match.extend(baseline_weights(seq, j, "mixo_match") for j in range(len(seq.masks)))
        fingerprints.append(rig.fingerprint)
        print(f"{split} sequence {index+1}/{count}: {len(target)} frames", flush=True)
    return (np.concatenate(features), np.concatenate(estimates), np.concatenate(masks),
            np.concatenate(labels), np.asarray(match), fingerprints)


def loss(gate, data, weights=None):
    x, e, mask, y = data[:4]
    weights = gate.predict(x, mask) if weights is None else weights
    return float(np.mean((np.einsum("be,bed->bd", weights, e)-y)**2))


def train(epochs=140):
    started = time.perf_counter()
    cv2.setNumThreads(1)
    training, validation, test = (dataset(split) for split in SPLITS)
    x, estimates, mask, y = training[:4]
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
    x = np.clip((x-mean)/scale, -8, 8)
    rng = np.random.default_rng(77104)
    p = {"w1": rng.normal(0, np.sqrt(1/len(FEATURE_NAMES)), (len(FEATURE_NAMES), 16)),
         "b1": np.zeros(16), "w2": rng.normal(0, .25, (16, 16)), "b2": np.zeros(16),
         "w3": rng.normal(0, .15, (16, 5)), "b3": np.zeros(5)}
    first_moment, second_moment = ({k: np.zeros_like(v) for k, v in p.items()} for _ in range(2))
    gate = Gate(p, mean, scale, {})
    p = gate.parameters
    best, best_loss, best_epoch, step = copy.deepcopy(p), float("inf"), 0, 0
    for epoch in range(1, epochs+1):
        indices = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            rows = indices[start:start+128]
            xb, eb, mb, yb = x[rows], estimates[rows], mask[rows], y[rows]
            a1 = np.tanh(xb@p["w1"]+p["b1"])
            d1 = (rng.random(a1.shape) >= .1)/.9
            h1 = a1*d1
            a2 = np.tanh(h1@p["w2"]+p["b2"])
            d2 = (rng.random(a2.shape) >= .1)/.9
            h2 = a2*d2
            w = masked_softmax(h2@p["w3"]+p["b3"], mb)
            error = np.einsum("be,bed->bd", w, eb)-yb
            dw = np.einsum("bd,bed->be", error, eb)*2/(len(rows)*6)
            dz = w*(dw-np.sum(w*dw, axis=1, keepdims=True))
            dh2 = (dz@p["w3"].T)*d2*(1-a2*a2)
            dh1 = (dh2@p["w2"].T)*d1*(1-a1*a1)
            gradients = {"w3": h2.T@dz, "b3": dz.sum(axis=0),
                         "w2": h1.T@dh2, "b2": dh2.sum(axis=0),
                         "w1": xb.T@dh1, "b1": dh1.sum(axis=0)}
            step += 1
            for key, gradient in gradients.items():
                gradient = np.clip(gradient+1e-5*p[key], -5, 5)
                first_moment[key] = .9*first_moment[key]+.1*gradient
                second_moment[key] = .999*second_moment[key]+.001*gradient**2
                p[key] -= .003*(first_moment[key]/(1-.9**step))/(np.sqrt(second_moment[key]/(1-.999**step))+1e-8)
        if epoch % 5 == 0 or epoch == 1:
            current_loss = loss(gate, validation)
            print(f"epoch {epoch}: validation scaled increment MSE={current_loss:.8g}", flush=True)
            if current_loss < best_loss:
                best, best_loss, best_epoch = copy.deepcopy(p), current_loss, epoch
            elif epoch-best_epoch >= 35:
                break
    gate.parameters = best
    uniform = test[2]/test[2].sum(axis=1, keepdims=True)
    gate.report = {
        "status": "trained", "architecture": [len(FEATURE_NAMES), 16, 16, 5],
        "activation": "tanh", "dropout": .1, "optimizer": "Adam", "learningRate": .003,
        "epochs": epoch, "selectedEpoch": best_epoch,
        "trainingSamples": len(training[0]), "validationSamples": len(validation[0]),
        "testSamples": len(test[0]), "trainingLoss": loss(gate, training),
        "validationLoss": loss(gate, validation), "testLoss": loss(gate, test),
        "uniformTestLoss": loss(gate, test, uniform), "matchTestLoss": loss(gate, test, test[4]),
        "featureNames": FEATURE_NAMES,
        **{f"{split}Seeds": list(range(start, start+count)) for split, (start, count, _) in SPLITS.items()},
        "namespaces": {"runtime": 0, **{split: s[2] for split, s in SPLITS.items()}},
        "datasetFingerprints": {split: data[5] for split, data in zip(SPLITS, (training, validation, test))},
        "targetScale": TARGET_SCALE.tolist(),
        "lossDefinition": "Mean squared six-dimensional body-frame increment error after scaling translation by 0.03m and rotation-vector by 0.02rad. Not trajectory ATE.",
        "source": "Offline synthetic known-landmark PnP experts and independent inertial propagation. Ground truth labels are used only offline.",
        "citation": "Morra et al., MIXO, IEEE Transactions on Consumer Electronics, 2023. DOI:10.1109/TCE.2023.3238655",
        "warnings": [
            "MIXO-inspired adaptation, not a paper reproduction or pretrained nuScenes model.",
            "Synthetic splits share a limited analytic motion family and room geometry; unseen real-world generalization is unproven.",
            "Validation selects the checkpoint; test is evaluated only after selection.",
            "Small-increment tangent MSE training; inference uses an intrinsic SO(3) rotation mean.",
            "Held-out incremental loss is not a claim of better trajectory accuracy or calibrated uncertainty.",
        ],
        "trainingSeconds": time.perf_counter()-started,
    }
    gate.save()
    print(json.dumps({"model": str(MODEL_PATH), **{k: gate.report[k] for k in
                     ("trainingSamples", "validationSamples", "testSamples", "testLoss", "uniformTestLoss", "matchTestLoss", "modelHash")}}, indent=2))
    return gate


if __name__ == "__main__":
    train()
