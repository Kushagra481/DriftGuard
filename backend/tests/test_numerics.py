"""Fast, deterministic numerical unit/regression tests; no browser required."""
import inspect
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from backend.models import ExperimentInput
from backend.simulation import generate
from backend.simulation.geometry import project, K, GRAVITY
from backend.simulation.sensors import GroundTruth
from backend.estimation import ESKF, estimate_pose
from backend.estimation.recovery import Recovery
from backend.goi import diagnose, KeyframePolicy
from backend.benchmarking.experiment import run_experiment, run_estimation
from backend.benchmarking.metrics import compute_metrics


@pytest.fixture(scope="module")
def measurements():
    return generate(ExperimentInput(preset="clean", duration=3))


def test_deterministic_simulation(measurements):
    m, gt = measurements
    other, other_gt = generate(ExperimentInput(preset="clean", duration=3))
    assert m.fingerprint == other.fingerprint
    np.testing.assert_array_equal(m.accel, other.accel)
    np.testing.assert_array_equal(gt.positions, other_gt.positions)
    np.testing.assert_array_equal(m.observations[10]["uv"], other.observations[10]["uv"])


def test_seed_changes_measurements(measurements):
    other, _ = generate(ExperimentInput(preset="clean", seed=13, duration=3))
    assert other.fingerprint != measurements[0].fingerprint


def test_pinhole_projection():
    uv, depth = project(np.array([[0., 0, 2], [1., 1., 2]]), np.zeros(3), Rotation.identity())
    np.testing.assert_allclose(uv, [[320, 240], [550, 470]])
    np.testing.assert_allclose(depth, [2, 2])


def test_imu_stationary_and_constant_acceleration():
    f = ESKF()
    for _ in range(200):
        f.propagate(np.zeros(3), -GRAVITY+np.array([1., 0, 0]), .005)
    np.testing.assert_allclose(f.p, [.5, 0, 0], atol=1e-9)
    np.testing.assert_allclose(f.v, [1, 0, 0], atol=1e-9)


def test_quaternion_normalization():
    f = ESKF()
    for _ in range(400):
        f.propagate(np.array([.03, -.1, .2]), -GRAVITY, .005)
    assert abs(np.linalg.norm(f.r.as_quat())-1) < 1e-12


def test_covariance_symmetric_positive_semidefinite():
    f = ESKF(adaptive=True)
    for i in range(300):
        f.propagate(np.array([.01, .02, -.005]), -GRAVITY, .005, .4)
        if i % 20 == 0:
            f.update(np.zeros(3), Rotation.identity(), np.eye(6)*.01)
        np.testing.assert_allclose(f.P, f.P.T, atol=1e-12)
        assert np.linalg.eigvalsh(f.P).min() >= -1e-10


def test_covariance_grows_during_visual_dropout():
    f = ESKF(adaptive=True)
    initial = f.uncertainty()[0]
    for _ in range(800):
        f.propagate(np.zeros(3), -GRAVITY, .005)
    assert f.uncertainty()[0] > initial*5


def test_recovery_measurement_reduces_uncertainty():
    f = ESKF(adaptive=True)
    for _ in range(400):
        f.propagate(np.array([.001, 0, 0]), -GRAVITY, .005)
    before = f.uncertainty()[0]
    assert f.update(np.zeros(3), Rotation.identity(), np.eye(6)*.0004)
    assert f.uncertainty()[0] < before


def test_reject_inconsistent_pose():
    f = ESKF()
    before = f.p.copy()
    assert not f.update(np.array([20., 0, 0]), Rotation.identity(), np.eye(6)*.001)
    np.testing.assert_array_equal(before, f.p)
    assert f.last_mahalanobis > 22.458


def test_pnp_rejects_visual_outliers(measurements):
    m, _ = measurements
    ids = m.observations[0]["ids"]
    uv, _ = project(m.landmarks[ids], np.zeros(3), Rotation.identity())
    corrupted = uv.copy()
    corrupted[::3] += np.array([80., -60.])
    pose = estimate_pose(m.landmarks, ids, corrupted)
    assert pose is not None
    assert np.linalg.norm(pose.position) < .01
    assert .5 < pose.ratio < .8


def test_goi_observable_and_weak_cases(measurements):
    m, _ = measurements
    ids = m.observations[0]["ids"]
    uv, _ = project(m.landmarks[ids], np.zeros(3), Rotation.identity())
    strong = diagnose(m.landmarks, ids, uv, np.zeros(3), Rotation.identity())
    assert strong["observableRank"] == 6
    weak_points = np.tile([0., 0., 6.], (20, 1))
    weak_uv, _ = project(weak_points, np.zeros(3), Rotation.identity())
    weak = diagnose(weak_points, np.arange(20), weak_uv, np.zeros(3), Rotation.identity())
    assert weak["observableRank"] < strong["observableRank"]
    assert weak["conditionNumber"] > strong["conditionNumber"]
    assert all(np.isfinite(f["goi"]) for f in strong["features"])


def test_goi_is_invariant_to_world_recentering(measurements):
    m, _ = measurements
    ids, uv = m.observations[0]["ids"], m.observations[0]["uv"]
    original = diagnose(m.landmarks, ids, uv, np.zeros(3), Rotation.identity())
    offset = np.array([500., -300., 200.])
    translated = diagnose(m.landmarks+offset, ids, uv, offset, Rotation.identity())
    np.testing.assert_allclose(original["eigenvalues"], translated["eigenvalues"], rtol=1e-8)
    np.testing.assert_allclose([x["goi"] for x in original["features"]],
                               [x["goi"] for x in translated["features"]], rtol=1e-7)


def test_metrics_against_known_ground_truth():
    t = np.arange(61)/30
    p = np.column_stack((t, t*0, t*0))
    r = Rotation.identity(len(t))
    gt = GroundTruth(t, p, r)
    metrics, errors = compute_metrics("test", "test", p+np.array([1., 0, 0]),
                                     r.as_quat(), gt, [True]*61, [2.]*61, [], 0)
    assert metrics["ate"] == pytest.approx(1)
    assert metrics["rpe"] == pytest.approx(0, abs=1e-12)
    assert metrics["rotationError"] == pytest.approx(0)
    assert metrics["meanLatency"] == 2
    assert metrics["recoveryDuration"] is None


def test_recovery_state_machine():
    r = Recovery()
    assert r.step(0, .8, .9, .03, True)[0] == "NORMAL"
    assert r.step(1, 0, .9, .1, False)[0] == "INERTIAL_HOLDOVER"
    assert r.step(2, 0, .2, .2, False)[0] == "MULTI_SENSOR_DEGRADED"
    assert r.step(3, 0, .2, .8, False)[0] == "TRACKING_UNCERTAIN"
    assert r.step(4, .8, .9, .1, True)[0] == "VISUAL_RECOVERY"
    assert r.step(4.3, .8, .9, .04, True)[0] == "NORMAL"
    assert r.durations == pytest.approx([3.3])


def test_no_ground_truth_argument_and_frozen_identical_inputs(measurements):
    m, _ = measurements
    assert list(inspect.signature(run_estimation).parameters) == ["measurements"]
    assert not m.accel.flags.writeable
    assert not m.gyro.flags.writeable
    with pytest.raises(ValueError):
        m.observations[0]["uv"][0] = 0


def test_experiment_identical_stream_and_finite_outputs():
    r, d = run_experiment(ExperimentInput(preset="camera_dropout", duration=3,
                                          parameters={"frame_dropout": 1.}))
    assert len(r["frames"]) == len(d) == 91
    assert len(set(r["metadata"]["methodFingerprints"].values())) == 1
    middle = [f for f in r["frames"] if f["degraded"]]
    assert all(f["featureCount"] == 0 for f in middle)
    assert any(f["state"] == "INERTIAL_HOLDOVER" for f in middle)
    assert r["frames"][-1]["state"] == "NORMAL"
    assert r["frames"][-1]["positionUncertainty"] < middle[-1]["positionUncertainty"]
    for metric in r["metrics"]:
        assert np.isfinite(metric["ate"])
        assert np.isfinite(metric["rpe"])
    adaptive = next(x for x in r["metrics"] if x["id"] == "adaptive")
    diag_only = next(x for x in r["metadata"]["ablationMetrics"] if x["id"] == "diagnostics_only")
    assert diag_only["ate"] == adaptive["ate"]
    assert r["metadata"]["goiDistributions"]
    assert sum(x["during"] for x in r["metadata"]["goiDistributions"]) == 0


def test_independent_parameter_validation():
    with pytest.raises(ValueError):
        ExperimentInput(parameters={"frame_dropout": 2})
    with pytest.raises(ValueError):
        ExperimentInput(parameters={"not_a_sensor": .2})
    with pytest.raises(ValueError):
        ExperimentInput(parameters={"pixel_noise": float("nan")})
