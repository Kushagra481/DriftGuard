# DriftGuard XR

**A sensor-degradation-aware 6-DoF trajectory-estimation research prototype.**

DriftGuard XR generates calibrated synthetic camera/IMU observations, runs actual
numerical pose estimators on frozen corrupted sensor sequences, and presents their
trajectories, uncertainty, geometric influence and evaluation metrics in an
interactive research workbench.

This is a **controlled, known-landmark pose-estimation benchmark**, not
general-purpose monocular SLAM. It does not process real headset footage and it
does not claim state-of-the-art performance. Improvements are not guaranteed:
the experiment reports whatever the estimators actually produce.

## Quick start

Requirements: Python 3.11+, Node.js 22+ and pnpm, uv, PostgreSQL.

```sh
pnpm install
uv sync --frozen
# Set DATABASE_URL through your environment or the workspace Secrets panel.
pnpm --filter @workspace/db run push
```

In Replit, start the existing **API Server** and **DriftGuard XR** workflows.
They supply the service ports and URL routing. The API is a Python FastAPI
application even though the workflow is invoked through a pnpm package script.
The unused Express scaffold is not part of the numerical runtime.

Outside Replit, the Python API can run with:

```sh
PORT=8080 python -m backend.api
# Separate terminal; configure an external reverse proxy:
# /api -> :8080, all other paths -> :5173
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/driftguard run dev
```

The frontend deliberately uses same-origin `/api` routes, not hardcoded hostnames.
The root artifact's base path is `/`.

## Reproduce an experiment

1. Open **Sensor Laboratory**.
2. Select a scenario, seed, duration, motion and individual corruption parameters.
3. Run the experiment. Computation is synchronous and usually takes a few seconds.
4. Scrub the completed sequence in **Live Experiment**, **Trajectory Analysis**
   or **GOI Explorer**. Playback is a replay of computed data, not live sensors.
5. Inspect learned per-camera weights and failures in **MIXO Experts**.
6. Read the comparison and isolated ablations in **Benchmarking**.
7. Download JSON (including per-feature diagnostics and gate provenance) or per-frame trajectory CSV.

The default experiment uses seed `42`, `combined`, an 8-second loop, 30 Hz
camera and 200 Hz IMU. Corruption is applied between 30% and 70% of the sequence,
allowing healthy, degraded and recovered periods to be inspected in one run.
Clean sensors still contain nominal measurement noise.

An API reproduction:

```sh
curl -X POST http://localhost:80/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"preset":"combined","seed":42,"duration":8,"motion":"loop"}'
```

Numerical outputs are reproducible for a fixed seed, configuration and
NumPy/OpenCV version. IDs, timestamps and measured latency are intentionally
not deterministic. OpenCV RNG is seeded, its internal threads are restricted
to one and runs are serialized because its RANSAC RNG is global. Requests
submitted while another experiment runs receive HTTP 409.

The measurement fingerprint hashes the IMU samples, packet-validity mask and
camera correspondences and measured health metadata. The original five methods
and GOI ablations consume the same immutable front-camera/IMU arrays. The
MIXO-inspired method and its uniform/MATCH baselines consume the same immutable
four-camera/IMU rig, including that exact front stream. **Comparing MIXO against
the original methods changes the sensor/map budget.** Its same-rig baselines
isolate gating without that confound.

## Problem and coordinates

We estimate camera/world pose and inertial biases when image features become
scarce, noisy or contaminated, and when inertial observations become biased,
noisy, saturated or unavailable.

World/camera axes follow OpenCV: **x right, y down, z forward**. Gravity is
`[0, 9.81, 0]` m/s². `R_wc` maps camera vectors to world coordinates.
Exported quaternions use **xyzw** order and rotations are integrated on SO(3).
The original front camera and IMU are collocated with identity extrinsics.
The MIXO rig additionally has left/right/rear cameras with known yaw rotations
and 8 cm offsets. Neither branch estimates calibration.

## Sensor model

A smooth analytical path supplies simulator-only position, acceleration and
orientation. Initial velocity is zero. Motion options are a translating loop,
pure rotation, and a low-translation path.

Static points lie on and between simulated indoor surfaces. The calibrated
pinhole camera has `fx=fy=460`, principal point `(320,240)`, image size 640×480:

```text
X_c = R_wcᵀ (X_w − p_w)
u = fx X_c.x / X_c.z + cx
v = fy X_c.y / X_c.z + cy
```

Inertial samples follow:

```text
ω_m = (1+s) ω_true + b_g + n_g
a_m = (1+s) R_wcᵀ (a_world − g) + b_a + n_a
b_(k+1) = b_k + n_walk √dt
```

The generator returns a `Measurements` object separately from `GroundTruth`.
The original estimator boundary accepts only `Measurements`; MIXO inference
accepts only primary and rig measurements. Ground truth is used after estimation
for error reporting and separately as **offline supervised gate-training labels**.
It never enters inference correction, quality scoring, uncertainty or initialization.

All estimators share the first healthy frame's PnP pose and a zero-velocity
prior. The IMU-only baseline is therefore **inertial propagation after a
shared visual initialization**, not uninitialized inertial navigation.

### Independent controls

| Parameter | Meaning / units |
|---|---|
| `blur` | 0–1 angular-motion-dependent feature loss and pixel noise |
| `illumination` | 0–1 brightness proxy, 1 means bright |
| `feature_dropout` | per-feature missing probability |
| `frame_dropout` | whole-camera-frame missing probability |
| `pixel_noise` | Gaussian pixel standard deviation |
| `outlier_ratio` | probability of a large false correspondence displacement |
| `concentration` | 0–1 central spatial restriction of available features |
| `parallax_scale` | 0–1 scale of translation for the whole continuous path |
| `gyro_noise` | rad/s per-sample standard deviation |
| `accel_noise` | m/s² per-sample standard deviation |
| `gyro_bias` | rad/s directional bias magnitude |
| `accel_bias` | m/s² directional bias magnitude |
| `bias_walk` | bias random-walk coefficient, both sensor types |
| `saturation` | shared numerical clip magnitude (gyro rad/s, accel m/s²) |
| `imu_dropout` | packet missing probability |
| `scale_error` | multiplicative inertial scale deviation |

Noise controls are **per-sample standard deviations**, not noise spectral
densities. Blur and low illumination are measurement-level proxies, not
rendered-image photometric simulations. Dynamic contamination changes
correspondences, not a full moving-object image renderer. The shared saturation
and bias-walk controls simplify otherwise separate physical sensor units.
Reduced parallax changes the whole motion path to avoid discontinuous position
jumps at degradation boundaries.

Ten presets cover nominal conditions, illumination, blur, occlusion, dynamic
outliers, inertial drift, vibration, camera dropout, combined and extreme
conditions. Every preset can be independently edited.

## Estimators and equations

### IMU-only

Propagates the shared initial state from measured acceleration and angular
velocity with no subsequent visual corrections. Missing packets use the last
received sample; gaps are reflected in adaptive process uncertainty.

### Vision-only

OpenCV `solvePnPRansac` (EPNP) estimates a pose from known landmark identities.
It uses a 4-pixel consensus threshold, up to 120 hypotheses and confidence
0.999. `solvePnPRefineLM` refines the consensus pose; reprojection residuals
then recompute inliers. At least six correspondences/inliers are required.
On failure the plotted pose holds its last valid value, but the frame is
marked unavailable rather than claiming a fresh estimate.

### Fifteen-state error-state Kalman filter

Nominal state:

```text
x = (p_world, v_world, q_wc, b_g, b_a)
δx = [δp, δv, δθ, δb_g, δb_a] ∈ R¹⁵
```

Right-multiplicative orientation error is used. For each actual IMU interval:

```text
ω = ω_m − b_g
a = R(q ⊗ Exp(ω dt/2)) (a_m − b_a) + g
p⁺ = p + v dt + ½ a dt²
v⁺ = v + a dt
q⁺ = q ⊗ Exp(ω dt)
```

Continuous error Jacobian nonzero blocks:

```text
F_pv = I
F_vθ = −R [a_m − b_a]×       F_vba = −R
F_θθ = −[ω]×                F_θbg = −I
Φ ≈ I + F dt + ½ F² dt²
P⁻ = Φ P Φᵀ + Q dt
```

Visual residual and update:

```text
r = [p_visual − p_predicted, Log(R_predicted⁻¹ R_visual)]
H selects position and right-local rotation error
S = H P⁻ Hᵀ + R_visual
d² = rᵀ S⁻¹ r
K = P⁻ Hᵀ S⁻¹
δx = K r
P⁺ = (I−KH) P⁻ (I−KH)ᵀ + K R_visual Kᵀ
```

The orientation error is injected on SO(3), followed by the right-error
covariance reset. Cholesky solves replace explicit inverses in the filter.
Covariance is symmetrized and non-finite states fail explicitly.

### Fixed-weight VIO

Uses fixed process/visual covariance and a fixed nine-frame visual update
interval. It shares robust PnP with all visual methods but does not apply the
additional predicted-state innovation gate.

### Adaptive VIO without GOI

Uses feature count, consensus ratio, reprojection RMSE, blur and dropout to
adapt visual covariance. IMU quality uses high-frequency first differences,
packet validity, saturation metadata, estimated biases and recent innovation.
It uses the same fixed nine-frame policy and rejects visual innovations above
the chi-square(6) 99.9% threshold, `22.458`.

### Full DriftGuard

Adds coverage, rotation-compensated bearing change and geometric conditioning
to visual health. For visual quality `q_v`, covariance scales inversely with
quality and quadratically with robust reprojection residual scale. Adaptive
process variance grows as measured inertial health decreases. These are
explicit **heuristic covariance-adaptation rules**, not a learned or
theoretically optimal probabilistic noise model.

Unreliable measurements are downweighted or not used. An unsuccessful
innovation check never silently forces a pose correction. Inertial propagation
continues and uncertainty grows. Returning visual observations must pass the
same consistency test.

### MIXO-inspired learned multicamera branch

The approved adaptation follows Morra et al., *MIXO: Mixture Of Experts-Based
Visual Odometry for Multicamera Autonomous Systems*, IEEE TCE (2023),
DOI [10.1109/TCE.2023.3238655](https://doi.org/10.1109/TCE.2023.3238655).
It is **not a reproduction** of their nuScenes/ORB-SLAM2 yaw-focused system.

Four independent known-map PnP experts and independent inertial propagation
produce body-frame motion increments. A trained NumPy MLP with two 16-unit tanh
layers and masked softmax learns context-dependent weights. Translation is
combined in the previous body frame; rotations use an intrinsic SO(3) mean.
Unavailable camera increments are hard-masked to zero weight. Camera recovery
requires two consecutive robust poses, avoiding a multi-frame jump interpreted
as one-frame motion. No camera and poor IMU health explicitly reports uncertainty.

**MIXO Experts** shows actual weights, failure reasons, sampled camera
observations, extrinsics, training provenance and same-rig comparisons. The
original five methods and three GOI ablations remain available. Old saved runs
are not retroactively assigned fictitious gate results.

Training is reproducible offline with no cloud services:

```sh
python -m backend.mixo.train
curl http://localhost:80/api/mixo/model
```

A small trained artifact is included. Training, validation, test and runtime use
disjoint random-stream namespaces even when numerical seeds coincide. Checkpoint
selection uses validation only. Reported held-out losses are **scaled incremental
motion MSE, not trajectory ATE**. Synthetic held-out success is not evidence of
real headset robustness or guaranteed improvement.

Per-camera visual overrides are optional:

```json
{
  "preset": "combined",
  "seed": 42,
  "duration": 8,
  "rigDegradation": "common",
  "cameraParameters": {
    "left": {"frame_dropout": 1},
    "rear": {"illumination": 0.2, "outlier_ratio": 0.35}
  }
}
```

Absent overrides inherit the global preset/parameters. Motion and IMU parameters
remain global. Default `staggered` corruption windows as fractions of duration
are front `[.3,.7]`, left `[.45,.85]`, right `[.15,.55]`, rear `[.35,.75]`.
`common` applies `[.3,.7]` to every camera. Per-camera RNG streams keep another
camera's observed noise unchanged when one override changes. Detailed equations,
paper differences and failure limits are in [the adaptation notes](docs/MIXO_ADAPTATION.md).

## GOI-inspired influence and observability

For feature residual `r_i` and camera-centered pose perturbation, compute the
pinhole projection derivative and the SE(3) pose Jacobian:

```text
J_i = Dπ(X_ci) [−I, [X_ci]×]
H_i = (w_i / σ_pixel²) J_iᵀ J_i
H = Σ H_i = V Λ Vᵀ
```

The SE(3) metric normalizes translation by **L=1 metre** and rotation in
radians. Camera-centered coordinates avoid arbitrary world-origin
dependence. Eigenvectors with eigenvalues greater than
`max(λ_max × 10⁻⁶, 10⁻⁸)` form the observable subspace:

```text
H† = V_obs Λ_obs⁻¹ V_obsᵀ
ℓ_i = trace(H† H_i)
g_i = J_iᵀ w_i r_i / σ_pixel²
GOI_i = ‖H† g_i‖ / max(1 − ℓ_i, 0.001)
```

This is a **GOI-inspired observable-subspace deletion-influence
approximation**, not a verified reproduction of a particular published
Geometric Observability Index. It attributes pose sensitivity; high influence
does not by itself make a feature an outlier.

Huber weights use normalized reprojection residuals with a MAD-based pixel
scale (floor 0.7 px). The UI's outlier classification uses normalized residual
greater than four, independently of raw GOI. Very large residuals receive
small robust weights. RANSAC consensus classification and this robust
diagnostic test need not produce identical outlier labels.

The interface exposes all six eigenvalues, observable rank, condition number,
weakest eigenvector, per-feature residual/weight/influence and temporal
distributions. Translation/rotation observability values are **fractions of
curvature trace under the chosen metric**, not calibrated probabilities.

Pure rotation does not necessarily make pose estimation unobservable with a
known metric 3D map. It does remove monocular triangulation parallax; this
prototype must not be confused with a map-building observability study.

## Keyframes

A local decaying information matrix represents the selected-frame history.
Each candidate's utility measures log-determinant information gain,
improvement along the weakest prior eigenvector, coverage, parallax and
conditioning. The reasoning and numeric values are exposed per frame.

Updates are limited to approximately 10 Hz, with a healthy-pose fallback
after 0.3 seconds. The fixed policy uses every ninth 30 Hz camera frame.
These selected keyframes control visual ESKF updates; they are not merely
decorative timeline markers. Different update counts are reported and must
be considered when comparing accuracy/latency.

## Recovery state machine

States are `NORMAL`, `VISUAL_DEGRADED`, `IMU_DEGRADED`,
`MULTI_SENSOR_DEGRADED`, `INERTIAL_HOLDOVER`, `VISUAL_RECOVERY` and
`TRACKING_UNCERTAIN`. They use measured health, consistency checks and
estimated uncertainty, not the true pose error.

Position uncertainty greater than 0.75 m overrides other states and marks
tracking uncertain. Reliable visual corrections must remain consistent for
0.2 seconds before returning to normal. Unscheduled correction frames do
not reset a valid recovery interval. Causes are shown in the interface.
Recovery duration is measured from the start of a degraded episode to the
validated return to normal. An unfinished/no recovery returns `null`, not zero.
Vision-only has no inertial state machine: its recovery duration measures
completed gaps from a failed PnP frame to the first returning valid PnP pose.

## Evaluation and ablations

Comparisons within each sensor budget share frozen inputs and initialization.
MIXO's four-camera/IMU budget is compared fairly with its uniform/MATCH
baselines, not described as equal-sensor superiority over the original
single-camera methods. Ground truth is evaluated without alignment or scale
fitting because the maps are already metric.

- **ATE RMSE:** root mean squared Euclidean position error over camera frames.
- **RPE RMSE:** relative translation error in local pose coordinates, with a
  one-second separation (shortened only for sequences under that length).
- **Translation error:** final positional error.
- **Rotation error:** RMSE of absolute SO(3) angle error, in degrees.
- **Drift:** final positional error divided by ground-truth path length, percent.
  For zero-translation paths the ratio is undefined and reported as zero;
  use absolute positional error instead.
- **Availability:** vision requires a valid PnP pose; filters require position
  uncertainty no greater than 0.75 m. This is not a claim of actual accuracy.
- **Uncertainty:** square root of the trace of each position/orientation
  covariance block; not a per-axis one-sigma confidence interval.
- **Latency:** measured wall-clock milliseconds per camera interval, including
  shared PnP work for visual methods, propagation/correction for filters and
  GOI work for geometry methods. It excludes sensor generation, serialization,
  storage and network. Mean, median and p95 are computed from measured samples.

Three additional real runs isolate contributions:

1. **Diagnostics only:** same adaptive-no-GOI updates and estimates, with GOI
   runtime overhead. Diagnostics alone must not magically improve a pose.
2. **Geometry covariance, fixed interval:** changes geometry-sensitive
   covariance while keeping update selection fixed.
3. **Observability keyframes, basic covariance:** changes keyframe policy
   while retaining basic adaptive covariance.

These are controlled implementation ablations, not a large multi-seed
statistical study. No performance percentages are prewritten or guaranteed.

## API

All routes are prefixed `/api`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/healthz` | Process health |
| GET | `/presets` | Ten fully specified scenarios |
| GET | `/mixo/model` | Actual trained gate status and split evidence |
| GET | `/experiments` | Thirty most recent saved summaries |
| POST | `/experiments` | Compute and save a new seeded run |
| GET | `/experiments/{id}` | Trajectories, metrics and provenance |
| PATCH | `/experiments/{id}` | Derive a new run; original stays immutable |
| GET | `/experiments/{id}/frames/{index}` | Features and geometric diagnostics |
| GET | `/experiments/{id}/trajectories` | Camera-rate estimated poses |
| GET | `/experiments/{id}/diagnostics` | Temporal sensor/estimator health |
| GET | `/experiments/{id}/goi` | Temporal observability summary |
| GET | `/experiments/{id}/benchmarks` | Six primary methods; inspect sensor-budget differences |
| GET | `/experiments/{id}/export/json` | Full structured JSON |
| GET | `/experiments/{id}/export/csv` | Camera-rate pose/error CSV |

Interactive backend documentation: `/api/docs`. The frontend contract lives
in `lib/api-spec/openapi.yaml`; after changing it run
`pnpm --filter @workspace/api-spec run codegen`.

Structured experiment data persists in PostgreSQL. No images or user-uploaded
files are stored in the database. The prototype has no authentication and is
intended for a controlled research/demo environment, not an unrestricted
multi-user compute service.

## Tests and code structure

```sh
python -m pytest backend/tests -q
pnpm run typecheck
```

The numerical tests cover seed determinism, pinhole projection, inertial
integration, quaternion normalization, covariance symmetry/PSD, observable
and weak geometry, world-origin invariance, visual outliers, innovation
rejection, uncertainty growth, recovery, identical immutable inputs and known
metric values. Tests assert mathematical behavior rather than superiority
of the proposed method.

```text
backend/
  models.py               validated inputs and scenario definitions
  simulation/             continuous motion, projection, camera/IMU corruption
  estimation/             PnP, 15-state ESKF, sensor health, recovery
  goi/                    influence, curvature spectrum, keyframe utility
  mixo/                   independent experts, trained gate and 6-DoF fusion
  benchmarking/           same-input orchestration and evaluation metrics
  repository.py           durable structured experiment records
  api.py                  FastAPI REST interface
  tests/                  deterministic numerical tests
artifacts/driftguard/src/  React research workbench and interactive 3D replay
lib/api-spec/             frontend API contract and code generation
lib/db/src/schema/        database schema for experiment records
```

## Known limitations and future real sensors

- No real image ingestion, feature extraction/association, mapping, loop
  closure or photometric rendering.
- Known static landmark identities and metric coordinates provide controlled
  correspondences unavailable to general-purpose monocular SLAM.
- Identity front-camera/IMU extrinsics and fixed known side/rear extrinsics;
  fixed gravity, no rolling shutter or clock
  offset calibration, no delayed/out-of-sequence measurement handling.
- Bias and saturation models are simplified; the process covariance uses a
  small-step second-order discretization rather than full exact integration.
- Robust residual thresholds and health/covariance rules are heuristic and
  may reject useful data or become inconsistent under extreme corruption.
- The estimator can fail; uncertainty is an estimate, not guaranteed coverage.
- Full DriftGuard and fixed-interval methods do not have equal update budgets.
- No claimed scientific validation against published GOI implementations or
  real XR datasets, and no multi-seed confidence intervals in this version.
- Runs are synchronous, serialized and capped at 30 seconds of simulated time.

Real headset integration needs synchronized calibrated image/IMU ingestion,
robust feature association, map initialization, extrinsic/time calibration and
dataset/hardware validation. None of those are claimed to exist here.
