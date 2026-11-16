# MIXO-inspired DriftGuard adaptation

## Reference and scope

L. Morra, A. Biondo, N. Poerio, F. Lamberti, “MIXO: Mixture Of Experts-Based
Visual Odometry for Multicamera Autonomous Systems,” IEEE Transactions on
Consumer Electronics, 2023. DOI: 10.1109/TCE.2023.3238655.

The user chose an adaptation for DriftGuard, not an exact paper reproduction.
The paper motivates independent camera experts, an inertial expert, context-aware
neural gating and relative-motion fusion. Its demonstrated gate is yaw-focused
automotive motion, trained with nuScenes ground truth; night scenes were
excluded. Its reported improvements are not results for this application.

| Aspect | MIXO paper | This adaptation |
|---|---|---|
| Visual frontend | Independent modified ORB-SLAM2 experts | Independent calibrated known-map PnP experts |
| Sensors | Four selected cameras and inertial odometry | Front, left, right, rear pinhole cameras and inertial propagation |
| Learned target | Incremental yaw | Scaled body-frame translation and rotation-vector increments |
| Gate | Two 16-unit hidden layers, dropout, softmax | Two 16-unit tanh layers, 0.1 dropout during training, masked softmax |
| Training | nuScenes supervised ground truth | Synthetic disjoint training/validation/test namespaces |
| Scale | Inertial translational scale | Known metric landmark maps |
| Inference rotation | Yaw fusion | Intrinsic SO(3) mean |

Extra side/rear map surfaces and three cameras are additional information.
Comparisons against single-camera DriftGuard therefore change the hardware/map
budget, not just the estimator. Three same-rig methods share every observation:
learned gating, uniform weighting over available experts, and MATCH (the valid
camera with most observed matches; inertial fallback).

## Pose and increment convention

`R_wb` maps body to world. `R_bc,t_bc` are calibrated camera-to-body rotation and
camera origin in body coordinates. PnP returns camera-to-world `R_wc,p_wc`.

```text
R_wb = R_wc R_bc^-1
p_wb = p_wc - R_wb t_bc
delta_t_i = R_wb(i,k-1)^T [p_wb(i,k)-p_wb(i,k-1)]
delta_r_i = Log[R_wb(i,k-1)^T R_wb(i,k)]
```

All experts' increments refer to the physical previous body frame. Pose-estimate
noise means these local coordinate systems are only approximately aligned.
Nonidentity camera rotations and offsets are transformed before fusion.

For available-expert mask `a_i` and learned logits `z_i`:

```text
w_i = a_i exp(z_i) / sum_j[a_j exp(z_j)]
delta_t = sum_i w_i delta_t_i
R_delta = argmin_R sum_i w_i ||Log(R^-1 Exp(delta_r_i))||^2
p_k = p_(k-1) + R_(k-1) delta_t
R_k = R_(k-1) R_delta
```

Rotation mean iteration starts at the highest-weight expert and uses
right-multiplicative exponential updates until convergence (at most 12 steps).
This is product-space translation/SO(3) fusion, **not** quaternion component
averaging, a full probabilistic SE(3) posterior or a claim of invariant
Lie-group maximum-likelihood fusion.

## Measured gate inputs

Forty-eight inputs: eight per expert for five experts, plus six previous
inertial-increment components, current inertial health and camera interval.
Per-expert statistics are feature count, robust inlier ratio, log reprojection
RMSE, measurement-derived health, availability mask, incremental translation
magnitude, rotation magnitude and disagreement with available-expert median.
For the IMU, visual-only statistics are zero.

Normalization mean/scale is fitted on training only and persisted with the
learned weights. Standardized inputs are clipped to ±8. The mask is enforced
outside learned logits; a network cannot assign weight to missing camera motion.
No inference input contains true pose, true error, injected-outlier labels or
the simulator's corruption-window flags. Feature residual/outlier previews use
PnP consensus, never injected truth labels.

## Supervised training and evidence

Run `python -m backend.mixo.train`. Only the offline training module obtains
ground-truth arrays. It uses exactly the measurement-only expert extractor
used at inference. The six target dimensions are consecutive body-frame
translation/metres and rotation-vector/radians, scaled by `[.03,.03,.03,.02,.02,.02]`.

The loss is MSE between the weighted expert tangent increments and target
increments. This is a small-rotation approximation to inference's intrinsic
rotation mean; losses must not be labelled trajectory ATE.

Training uses Adam, learning rate .003, batch size 128, at most 140 epochs,
0.1 dropout on both hidden layers, small L2 regularization and gradient
clipping. The checkpoint is chosen by validation loss. Test data is evaluated
only after selection and is never used for parameter updates.

| Split | Numeric seeds | RNG namespace |
|---|---|---|
| Training | 11001–11032 | 1101 |
| Validation | 22001–22008 | 2202 |
| Test | 33001–33008 | 3303 |
| Runtime experiments | User-selected | 0 |

Namespaces enter primary sensor and rig generators, making runtime data
different even if the user picks a listed offline seed. Datasets mix existing
motion families, presets, common/staggered failure windows and per-camera
noise/dropout/outlier overrides. However, all splits share a limited analytic
motion family and synthetic room distribution. This is not a claim of robust
generalization to unseen devices, scenes or real images.

The small model artifact includes actual weights, normalization, split seeds,
dataset fingerprints, selected epoch, observed losses and a parameter checksum.
`GET /api/mixo/model` returns actual provenance. Each saved experiment also
stores the report used for that run, so later retraining cannot silently
rewrite its evidence. Missing/corrupt model files fail explicitly rather than
falling back to fabricated learned weights.

## Degradation, availability and uncertainty

- Each camera independently uses robust known-map PnP.
- A valid increment requires consecutive valid poses, at least eight observed
  points, at least 0.3 inlier ratio and a 10 m/s / 10 rad/s plausibility bound.
  Those safety priors are visible limitations, not learned thresholds.
- After a camera gap, the first valid frame reinitializes its history and has
  zero gate weight; the next consecutive valid frame may contribute.
- Inertial integration uses available packets; missing packets hold the last
  sample and reduce observed inertial health.
- With all cameras unavailable, the gate assigns the inertial expert all
  weight. This does not mean the estimate is reliable. Diagnostic uncertainty
  grows, and simultaneous poor inertial health explicitly marks tracking uncertain.
- Uncertainty combines approximate per-expert noise, mixture disagreement,
  propagation growth and holdover penalties. Shared map/temporal correlations
  are not modelled; it is **not a calibrated Kalman posterior**.
- Declared availability uses position diagnostic ≤0.75 m and angular diagnostic
  ≤15 degrees and excludes simultaneous visual loss/very poor inertial health.
  It is never thresholded using ground-truth error.

The original ESKF/GOI branch retains its own uncertainty, recovery, visual
innovation gates and keyframe policy. It is not replaced by the learned branch.
Neither branch can guarantee recovery or bounded drift after all sensors fail.

## UI, replay and exports

The MIXO Experts page uses each completed experiment's actual frame records:
weights, masks, reasons, camera residuals, pose increments, entropy (natural-log
units), disagreement, state, uncertainty and measured latency. Replay times
are seconds; window metadata is fractions of duration.
Disagreement combines local translation normalized by L=1 m and rotation in
radians as a dimensionless product-space diagnostic. It is not a calibrated
probability of failure. Camera previews evenly sample up to
80 actual correspondences per frame, explicitly labelled; inference uses all.

JSON exports include gate records, training report and per-camera preview
observations. CSV contains six primary pose rows per frame for new runs,
including MIXO-specific state/uncertainty and five gate weights plus entropy.
Legacy runs keep their original data and show that MIXO results are absent.
