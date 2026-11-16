import { useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { Experiment, Feature } from '@workspace/api-client-react';
import { METHOD_IDS, METHOD_META, TRUTH_COLOR, cv2three } from '@/lib/format';

const FOV = 42;

function Landmarks({ pts, opacity = 0.75 }: { pts: number[][]; opacity?: number }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const arr = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      const v = cv2three(p);
      arr[i * 3] = v[0];
      arr[i * 3 + 1] = v[1];
      arr[i * 3 + 2] = v[2];
    });
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    return g;
  }, [pts]);
  return (
    <points geometry={geom}>
      <pointsMaterial size={0.05} color="#6b7a8c" sizeAttenuation transparent opacity={opacity} />
    </points>
  );
}

function quat(q: number[]) {
  return new THREE.Quaternion(q[0] ?? 0, q[1] ?? 0, q[2] ?? 0, q[3] ?? 1);
}

function PoseMarker({ position, quaternion, color, size = 0.12 }: { position: number[]; quaternion: number[]; color: string; size?: number }) {
  const p = cv2three(position);
  // Camera optical axis is +z in the OpenCV camera frame; rotate to world, then map to display frame.
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(quat(quaternion));
  const tip = cv2three([position[0] + fwd.x * size * 2.4, position[1] + fwd.y * size * 2.4, position[2] + fwd.z * size * 2.4]);
  return (
    <group>
      <mesh position={p}>
        <boxGeometry args={[size * 0.55, size * 0.55, size * 0.55]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Line points={[p, tip]} color={color} lineWidth={1.5} />
    </group>
  );
}

/** Frames the camera on given bounds whenever bounds or viewport size change. */
function Framer({ center, radius }: { center: [number, number, number]; radius: number }) {
  const { camera, size, controls } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    const vf = (FOV * Math.PI) / 180;
    const aspect = size.width / Math.max(1, size.height);
    const hf = 2 * Math.atan(Math.tan(vf / 2) * aspect);
    const dist = (radius / Math.sin(Math.min(vf, hf) / 2)) * 1.08;
    const dir = new THREE.Vector3(0.62, 0.55, 0.56).normalize();
    cam.position.set(center[0] + dir.x * dist, center[1] + dir.y * dist, center[2] + dir.z * dist);
    cam.near = Math.max(0.005, dist / 500);
    cam.far = dist * 50;
    cam.lookAt(center[0], center[1], center[2]);
    cam.updateProjectionMatrix();
    const c = controls as unknown as { target?: THREE.Vector3; update?: () => void } | null;
    if (c?.target) {
      c.target.set(center[0], center[1], center[2]);
      c.update?.();
    }
  }, [camera, controls, size.width, size.height, center, radius]);
  return null;
}

function ramp(t: number) {
  const x = Math.max(0, Math.min(1, t));
  const a = x < 0.5 ? new THREE.Color('#465466') : new THREE.Color('#a68cff');
  const b = x < 0.5 ? new THREE.Color('#a68cff') : new THREE.Color('#55d3e0');
  return a.lerp(b, x < 0.5 ? x / 0.5 : (x - 0.5) / 0.5);
}

export interface AnchorResult {
  truth: number[];
  estimate: number[];
  drift: number;
}

/**
 * Evaluation-only virtual anchor. A fixed world point (the known landmark nearest the
 * ground-truth path centroid) is expressed in the true camera frame, then re-placed in the
 * world using the estimated pose: a_est = R_est R_trueᵀ (a − p_true) + p_est.
 * The displacement is what a world-locked hologram would appear to slip by.
 */
export function computeAnchor(experiment: Experiment, frameIndex: number, method = 'driftguard'): AnchorResult | null {
  const f = experiment.frames[frameIndex];
  const est = f?.poses[method];
  if (!f || !est || !experiment.landmarks.length) return null;
  const n = experiment.frames.length;
  const c = experiment.frames.reduce((acc, x) => [acc[0] + x.truth.position[0] / n, acc[1] + x.truth.position[1] / n, acc[2] + x.truth.position[2] / n], [0, 0, 0]);
  let best = experiment.landmarks[0];
  let bd = Infinity;
  for (const l of experiment.landmarks) {
    const d = (l[0] - c[0]) ** 2 + (l[1] - c[1]) ** 2 + (l[2] - c[2]) ** 2;
    if (d < bd) {
      bd = d;
      best = l;
    }
  }
  const a = new THREE.Vector3(best[0], best[1], best[2]);
  const pT = new THREE.Vector3(...(f.truth.position as [number, number, number]));
  const pE = new THREE.Vector3(...(est.position as [number, number, number]));
  const local = a.clone().sub(pT).applyQuaternion(quat(f.truth.quaternion).invert());
  const ae = local.applyQuaternion(quat(est.quaternion)).add(pE);
  return { truth: [a.x, a.y, a.z], estimate: [ae.x, ae.y, ae.z], drift: ae.distanceTo(a) };
}

export function Trajectory3D({
  experiment,
  frameIndex,
  visible,
  showLandmarks = true,
  anchor,
  features,
  selectedFeature,
  onSelectFeature,
  weakDirection,
  focusFrame = false,
}: {
  experiment: Experiment;
  frameIndex: number;
  visible: Record<string, boolean>;
  showLandmarks?: boolean;
  anchor?: AnchorResult | null;
  features?: Feature[];
  selectedFeature?: number | null;
  onSelectFeature?: (id: number | null) => void;
  /** weakest translation direction, already in OpenCV world coordinates (unit) */
  weakDirection?: number[] | null;
  /** frame bounds around current pose + its features instead of full trajectory */
  focusFrame?: boolean;
}) {
  const frames = experiment.frames;
  const f = frames[frameIndex];
  const paths = useMemo(() => {
    const truth = frames.map((x) => cv2three(x.truth.position));
    const m: Record<string, [number, number, number][]> = {};
    for (const id of METHOD_IDS) m[id] = frames.filter((x) => x.poses[id]).map((x) => cv2three(x.poses[id].position));
    return { truth, m };
  }, [frames]);

  const bounds = useMemo(() => {
    const b = new THREE.Box3();
    if (focusFrame && f) {
      b.expandByPoint(new THREE.Vector3(...cv2three(f.truth.position)));
      (features ?? []).forEach((x) => b.expandByPoint(new THREE.Vector3(...cv2three(x.landmark))));
    } else {
      paths.truth.forEach((p) => b.expandByPoint(new THREE.Vector3(...p)));
      // Divergent baselines must remain visible rather than being clipped to
      // ground-truth bounds. Hiding a method recomputes the comparison bounds.
      METHOD_IDS.filter((id) => visible[id]).forEach((id) => {
        paths.m[id].forEach((p) => b.expandByPoint(new THREE.Vector3(...p)));
      });
      if (anchor) b.expandByPoint(new THREE.Vector3(...cv2three(anchor.truth)));
    }
    if (b.isEmpty()) b.expandByPoint(new THREE.Vector3());
    const c = new THREE.Vector3();
    b.getCenter(c);
    const s = new THREE.Vector3();
    b.getSize(s);
    return { c: [c.x, c.y, c.z] as [number, number, number], r: Math.max(0.6, s.length() / 2), minY: b.min.y };
    // focus mode reframes when the selected frame changes; trajectory mode only per run
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paths, visible, focusFrame, focusFrame ? frameIndex : -1, focusFrame ? features : null, anchor ? 1 : 0]);

  const featRange = useMemo(() => {
    const v = (features ?? []).filter((x) => !x.missing && Number.isFinite(x.goi)).map((x) => x.goi);
    return v.length ? { lo: Math.min(...v), hi: Math.max(...v) } : { lo: 0, hi: 1 };
  }, [features]);
  const nrm = (g: number) => (featRange.hi - featRange.lo > 1e-12 ? (g - featRange.lo) / (featRange.hi - featRange.lo) : 0.5);

  const cut = frameIndex + 1;
  // Geometry is evaluated at visual PnP when valid, at the ESKF prediction
  // otherwise. Keep the displayed rays/directions in that same pose frame.
  const diagnosticPose = f ? (f.inlierRatio > 0 ? f.poses.vision : f.poses.driftguard) : undefined;
  const camPos = diagnosticPose ? cv2three(diagnosticPose.position) : null;
  const weakOrigin = diagnosticPose?.position;

  return (
    <Canvas camera={{ fov: FOV }} dpr={[1, 2]} onPointerMissed={() => onSelectFeature?.(null)}>
      <color attach="background" args={['#0b1016']} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.15} />
      <Framer center={bounds.c} radius={bounds.r} />
      <Grid position={[bounds.c[0], bounds.minY - 0.6, bounds.c[2]]} args={[40, 40]} cellSize={0.5} sectionSize={2.5} cellColor="#18212b" sectionColor="#24303d" fadeDistance={bounds.r * 12} infiniteGrid />
      {showLandmarks && experiment.landmarks.length > 0 && <Landmarks pts={experiment.landmarks} opacity={features ? 0.35 : 0.75} />}
      {paths.truth.length > 1 && <Line points={paths.truth} color={TRUTH_COLOR} lineWidth={1} dashed dashSize={0.06} gapSize={0.05} transparent opacity={0.55} />}
      {METHOD_IDS.map((id) => {
        const pts = paths.m[id].slice(0, cut);
        if (!visible[id] || pts.length < 2) return null;
        return <Line key={id} points={pts} color={METHOD_META[id].color} lineWidth={id === 'driftguard' ? 2.2 : 1.4} />;
      })}
      {f && <PoseMarker position={f.truth.position} quaternion={f.truth.quaternion} color={TRUTH_COLOR} size={0.14} />}
      {f && METHOD_IDS.map((id) => (visible[id] && f.poses[id] ? <PoseMarker key={id} position={f.poses[id].position} quaternion={f.poses[id].quaternion} color={METHOD_META[id].color} size={0.1} /> : null))}

      {anchor && (
        <group>
          <mesh position={cv2three(anchor.truth)}>
            <octahedronGeometry args={[0.09]} />
            <meshBasicMaterial color={TRUTH_COLOR} wireframe />
          </mesh>
          <mesh position={cv2three(anchor.estimate)}>
            <octahedronGeometry args={[0.09]} />
            <meshBasicMaterial color={METHOD_META.driftguard.color} />
          </mesh>
          <Line points={[cv2three(anchor.truth), cv2three(anchor.estimate)]} color="#a68cff" lineWidth={1.5} />
        </group>
      )}

      {features && camPos &&
        features.map((x) => {
          const p = cv2three(x.landmark);
          const sel = x.id === selectedFeature;
          const click = (e: { stopPropagation: () => void }) => {
            e.stopPropagation();
            onSelectFeature?.(sel ? null : x.id);
          };
          const col = x.missing ? new THREE.Color('#5b6878') : ramp(nrm(x.goi));
          const s = x.missing ? 0.035 : 0.035 + nrm(x.goi) * 0.05;
          return (
            <group key={x.id}>
              {(sel || (!x.missing && !x.outlier)) && <Line points={[camPos, p]} color={sel ? '#e6dfcc' : col} lineWidth={sel ? 1.5 : 0.5} transparent opacity={sel ? 0.9 : 0.25} />}
              <mesh position={p} onClick={click}>
                <sphereGeometry args={[s, 12, 12]} />
                <meshBasicMaterial color={col} wireframe={x.missing} />
              </mesh>
              {x.outlier && (
                <mesh position={p} onClick={click}>
                  <torusGeometry args={[s + 0.04, 0.01, 6, 24]} />
                  <meshBasicMaterial color="#d9737e" />
                </mesh>
              )}
              {sel && (
                <mesh position={p}>
                  <boxGeometry args={[s * 3.2, s * 3.2, s * 3.2]} />
                  <meshBasicMaterial color="#e6dfcc" wireframe />
                </mesh>
              )}
            </group>
          );
        })}

      {weakDirection && weakOrigin && (() => {
        const L = bounds.r * 0.5;
        const o = weakOrigin;
        const a = cv2three([o[0] - weakDirection[0] * L, o[1] - weakDirection[1] * L, o[2] - weakDirection[2] * L]);
        const b = cv2three([o[0] + weakDirection[0] * L, o[1] + weakDirection[1] * L, o[2] + weakDirection[2] * L]);
        return (
          <group>
            <Line points={[a, b]} color="#d4a24a" lineWidth={2.5} dashed dashSize={0.05} gapSize={0.03} />
          </group>
        );
      })()}
    </Canvas>
  );
}
