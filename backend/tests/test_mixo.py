"""MIXO-inspired adaptation invariants, not paper performance claims."""
import inspect
import json
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from backend.models import ExperimentInput
from backend.simulation.sensors import generate
from backend.simulation.rig import generate_rig
from backend.mixo.experts import extract_experts
from backend.mixo.gate import load_gate, masked_softmax, FEATURE_NAMES
from backend.mixo.geometry import camera_to_body, relative_increment, fuse_increment, integrate_increment
from backend.mixo.runner import run_mixo, baseline_weights


def test_camera_extrinsics_roundtrip():
    p, r = np.array([1., -2., 3.]), Rotation.from_euler("xyz", [.2, -.3, .4])
    extrinsic = Rotation.from_euler("y", 90, degrees=True)
    offset = np.array([.08, .02, -.03])
    body_p, body_r = camera_to_body(p+r.apply(offset), r*extrinsic, extrinsic, offset)
    np.testing.assert_allclose(body_p, p, atol=1e-12)
    assert (body_r.inv()*r).magnitude() < 1e-12


def test_local_increment_reconstructs_full_6dof_pose():
    p, r = np.array([1., 2., 3.]), Rotation.from_euler("xyz", [.1, .2, -.4])
    target_p, target_r = np.array([1.3, 1.9, 3.2]), r*Rotation.from_rotvec([.04, -.03, .02])
    delta = relative_increment(p, r, target_p, target_r)
    new_p, new_r = integrate_increment(p, r, delta)
    np.testing.assert_allclose(new_p, target_p)
    assert (new_r.inv()*target_r).magnitude() < 1e-12


def test_rotation_fusion_respects_wraparound_and_order():
    inc = np.zeros((2, 6))
    inc[:, 3:] = Rotation.from_euler("z", [[179.], [-179.]], degrees=True).as_rotvec()
    fused = fuse_increment(inc, [.5, .5])
    assert abs(np.linalg.norm(fused[3:])-np.pi) < 1e-8
    assert np.linalg.norm(fuse_increment(inc[::-1], [.5, .5])[3:]) == pytest.approx(np.pi)
    np.testing.assert_array_equal(fuse_increment(inc, [0, 0]), np.zeros(6))


def test_masked_softmax_is_finite_and_cannot_trust_unavailable_experts():
    logits = np.array([[1e9, 2., 3.], [7., 8., 9.]])
    weights = masked_softmax(logits, [[False, True, True], [False, False, False]])
    assert weights[0, 0] == 0
    assert weights[0].sum() == pytest.approx(1.)
    np.testing.assert_array_equal(weights[1], np.zeros(3))
    assert np.isfinite(weights).all()


def test_camera_controls_validate_independently():
    with pytest.raises(ValidationError):
        ExperimentInput(cameraParameters={"left": {"gyro_noise": .1}})
    with pytest.raises(ValidationError):
        ExperimentInput(cameraParameters={"bad-camera": {"blur": .1}})
    with pytest.raises(ValidationError):
        ExperimentInput(cameraParameters={"rear": {"frame_dropout": 1.01}})
    config = ExperimentInput(cameraParameters={"front": {"illumination": .1}, "rear": {"blur": .8}})
    assert config.cameraParameters["rear"]["blur"] == .8


def test_independent_camera_streams_and_shared_front():
    config = ExperimentInput(duration=3, seed=76)
    m, _ = generate(config)
    original = generate_rig(config, m)
    changed = generate_rig(config.model_copy(update={"cameraParameters": {"left": {"frame_dropout": 1.}}}), m)
    assert original.cameras[0].observations is m.observations
    assert original.fingerprint != changed.fingerprint
    for c in (0, 2, 3):
        for before, after in zip(original.cameras[c].observations, changed.cameras[c].observations):
            np.testing.assert_array_equal(before["ids"], after["ids"])
            np.testing.assert_array_equal(before["uv"], after["uv"])
    assert not original.landmarks.flags.writeable
    assert not original.cameras[1].observations[0]["uv"].flags.writeable


def test_offline_stream_namespace_cannot_overlap_runtime_even_with_same_seed():
    config = ExperimentInput(seed=11001, duration=3)
    runtime, _ = generate(config)
    offline, _ = generate(config, rng_namespace=1101)
    assert runtime.fingerprint != offline.fingerprint
    again, _ = generate(config)
    assert runtime.fingerprint == again.fingerprint


def test_experts_and_fusion_never_accept_truth():
    assert list(inspect.signature(extract_experts).parameters) == ["measurements", "rig", "include_views"]
    assert list(inspect.signature(run_mixo).parameters) == ["measurements", "rig", "gate"]
    source = inspect.getsource(extract_experts)
    assert '["degraded"]' not in source
    assert "truth." not in source


def test_trained_gate_and_masking_are_real(tmp_path):
    gate = load_gate()
    assert gate.report["status"] == "trained"
    assert gate.parameters["w1"].shape == (len(FEATURE_NAMES), 16)
    assert gate.report["trainingSamples"] > 1000
    assert gate.report["validationSamples"] > 0 and gate.report["testSamples"] > 0
    assert len(set(gate.report["namespaces"].values())) == 4
    outputs = gate.predict(np.array([gate.mean, gate.mean+gate.scale]), np.ones((2, 5), dtype=bool))
    assert np.max(abs(outputs[0]-outputs[1])) > 1e-5
    masked = gate.predict(gate.mean, np.array([False, False, False, False, True]))
    np.testing.assert_array_equal(masked, [0, 0, 0, 0, 1])
    path = tmp_path/"gate.json"
    gate.save(path)
    payload = json.loads(path.read_text())
    payload["parameters"]["b3"][0] += .1
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="checksum"):
        load_gate(path)


def test_dropout_reappearance_never_fuses_multi_frame_motion_as_one_frame():
    config = ExperimentInput(duration=3, rigDegradation="common", preset="clean",
                             parameters={"frame_dropout": 1.})
    m, _ = generate(config)
    rig = generate_rig(config, m)
    seq = extract_experts(m, rig, include_views=False)
    assert seq.features.shape[1] == len(FEATURE_NAMES)
    missing = np.flatnonzero([len(obs["ids"]) == 0 for obs in m.observations])
    assert not seq.masks[missing, :4].any()
    assert not seq.masks[missing[-1]+1, :4].any()
    assert seq.masks[missing[-1]+2, :4].all()
    for i in missing:
        np.testing.assert_array_equal(baseline_weights(seq, i, "mixo_match"), [0, 0, 0, 0, 1])
    learned = load_gate().predict(seq.features[missing], seq.masks[missing])
    np.testing.assert_array_equal(learned[:, :4], np.zeros((len(missing), 4)))
    np.testing.assert_allclose(learned.sum(axis=1), 1)
    frames, _, paths, _, _ = run_mixo(m, rig)
    assert frames[missing[-1]]["positionUncertainty"] > frames[missing[0]]["positionUncertainty"]
    for path in paths.values():
        assert np.isfinite(path["p"]).all()
        np.testing.assert_allclose(np.linalg.norm(path["q"], axis=1), 1., atol=1e-10)


def test_simultaneous_camera_and_imu_failure_is_not_declared_healthy():
    config = ExperimentInput(duration=3, rigDegradation="common", preset="clean",
                             parameters={"frame_dropout": 1., "imu_dropout": 1.})
    m, _ = generate(config)
    rig = generate_rig(config, m)
    frames, _, paths, _, _ = run_mixo(m, rig)
    middle = [i for i, obs in enumerate(m.observations) if len(obs["ids"]) == 0][2:-2]
    assert middle
    assert all(frames[i]["state"] == "TRACKING_UNCERTAIN" for i in middle)
    assert all(not paths["mixo"]["available"][i] for i in middle)
    assert all(frames[i]["weights"]["imu"] == 1. for i in middle)
