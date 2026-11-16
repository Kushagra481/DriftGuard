import type { Frame } from '@workspace/api-client-react';

export const METHOD_IDS = ['imu', 'vision', 'fixed', 'adaptive', 'driftguard'] as const;
export type MethodId = (typeof METHOD_IDS)[number];

export const METHOD_META: Record<MethodId, { label: string; short: string; color: string; desc: string }> = {
  imu: { label: 'IMU dead-reckoning', short: 'IMU', color: '#d4a24a', desc: 'Strapdown integration of gyro + accelerometer, no visual correction.' },
  vision: { label: 'Vision-only PnP', short: 'VIS', color: '#d9737e', desc: 'Per-frame pose from 2D-3D correspondences against known landmarks.' },
  fixed: { label: 'Fixed-weight VIO', short: 'FIX', color: '#7d9ccb', desc: 'ESKF with fixed process/visual covariance and a fixed nine-frame visual update interval.' },
  adaptive: { label: 'Adaptive VIO', short: 'ADP', color: '#a68cff', desc: 'ESKF with health-adapted visual covariance and chi-square innovation gating; fixed nine-frame interval, no GOI.' },
  driftguard: { label: 'DriftGuard', short: 'DGX', color: '#55d3e0', desc: 'Adds geometry-aware covariance heuristics and GOI-inspired observability keyframe selection.' },
};
export const TRUTH_COLOR = '#e6dfcc';

export function methodColor(id: string) {
  return (METHOD_META as Record<string, { color: string }>)[id]?.color ?? '#8a97a6';
}
export function methodShort(id: string) {
  return (METHOD_META as Record<string, { short: string }>)[id]?.short ?? id.toUpperCase();
}

export function fmt(v: number | null | undefined, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '+inf' : '-inf';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

export const PARAM_SPEC: {
  key: string;
  label: string;
  group: 'camera' | 'imu';
  min: number;
  max: number;
  step: number;
  unit: string;
  hint: string;
}[] = [
  { key: 'blur', label: 'Motion blur', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Blur severity applied to detections; inflates keypoint localisation noise.' },
  { key: 'illumination', label: 'Illumination', group: 'camera', min: 0, max: 1, step: 0.01, unit: '0 dark .. 1 bright', hint: 'Scene brightness. Low values suppress detectable features.' },
  { key: 'feature_dropout', label: 'Feature dropout', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Fraction of visible landmarks removed per frame.' },
  { key: 'frame_dropout', label: 'Frame dropout', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Probability an entire 30 Hz camera frame is lost.' },
  { key: 'pixel_noise', label: 'Pixel noise', group: 'camera', min: 0, max: 10, step: 0.1, unit: 'px σ', hint: 'Gaussian noise on observed keypoints.' },
  { key: 'outlier_ratio', label: 'Outlier ratio', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Fraction of correspondences replaced by gross mismatches.' },
  { key: 'concentration', label: 'Feature concentration', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Clusters surviving features into a small image region; reduces geometric spread.' },
  { key: 'parallax_scale', label: 'Parallax scale', group: 'camera', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Scales translational baseline. 0 removes parallax.' },
  { key: 'gyro_noise', label: 'Gyro noise', group: 'imu', min: 0, max: 0.1, step: 0.001, unit: 'rad/s σ', hint: 'White noise on angular rate at 200 Hz.' },
  { key: 'accel_noise', label: 'Accel noise', group: 'imu', min: 0, max: 1, step: 0.01, unit: 'm/s² σ', hint: 'White noise on specific force at 200 Hz.' },
  { key: 'gyro_bias', label: 'Gyro bias', group: 'imu', min: 0, max: 0.1, step: 0.001, unit: 'rad/s', hint: 'Constant angular rate offset.' },
  { key: 'accel_bias', label: 'Accel bias', group: 'imu', min: 0, max: 0.5, step: 0.01, unit: 'm/s²', hint: 'Constant specific-force offset.' },
  { key: 'bias_walk', label: 'Bias random walk', group: 'imu', min: 0, max: 0.1, step: 0.0005, unit: 'per √s', hint: 'Random-walk diffusion of both biases.' },
  { key: 'saturation', label: 'Saturation limit', group: 'imu', min: 0.1, max: 100, step: 0.1, unit: 'rad/s | m/s² max', hint: 'Clip magnitude for IMU readings. Default 100 effectively disables clipping.' },
  { key: 'imu_dropout', label: 'IMU dropout', group: 'imu', min: 0, max: 1, step: 0.01, unit: 'ratio', hint: 'Fraction of IMU samples lost.' },
  { key: 'scale_error', label: 'Scale error', group: 'imu', min: 0, max: 0.5, step: 0.005, unit: 'ratio', hint: 'Multiplicative accelerometer scale factor error.' },
];

/** OpenCV world (x right, y down, z forward) -> three.js (y up). Flips y and z, preserving handedness. */
export function cv2three(p: number[]): [number, number, number] {
  return [p[0] ?? 0, -(p[1] ?? 0), -(p[2] ?? 0)];
}

export function degradedWindow(frames: Frame[]) {
  const idx = frames.map((f, i) => (f.degraded ? i : -1)).filter((i) => i >= 0);
  if (!idx.length) return null;
  return { start: idx[0], end: idx[idx.length - 1], t0: frames[idx[0]].time, t1: frames[idx[idx.length - 1]].time };
}

export function shortId(id: string) {
  return id.slice(0, 8);
}
