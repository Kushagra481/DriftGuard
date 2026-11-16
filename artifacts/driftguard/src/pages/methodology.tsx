import type { ReactNode } from 'react';
import { useExperimentCtx } from '@/lib/experiment-context';
import { METHOD_IDS, METHOD_META } from '@/lib/format';
import { Panel, Swatch } from '@/components/instrument';

function Eq({ children, n }: { children: ReactNode; n: string }) {
  return (
    <div className="my-3 flex items-center gap-4 border-l-2 border-accent/60 bg-secondary/40 px-4 py-2.5">
      <code className="font-mono text-[13px] text-foreground flex-1 overflow-x-auto whitespace-nowrap">{children}</code>
      <span className="font-mono text-[10px] text-muted-foreground">({n})</span>
    </div>
  );
}

function H({ children }: { children: ReactNode }) {
  return <h3 className="font-cond font-semibold text-[15px] tracking-wide mt-6 mb-1 first:mt-0">{children}</h3>;
}

function strList(x: unknown): string[] {
  return Array.isArray(x) ? x.filter((s): s is string => typeof s === 'string') : [];
}

export default function MethodologyPage() {
  const { experiment } = useExperimentCtx();
  const meta = experiment?.metadata ?? {};
  const limitations = strList(meta.limitations);
  const ablations = strList(meta.ablations);
  const fp = meta.measurementFingerprint;
  const fpText = fp === undefined ? null : typeof fp === 'string' ? fp : JSON.stringify(fp, null, 2);

  return (
    <div className="flex-1 overflow-auto">
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-px bg-border min-h-full">
        <article className="bg-background px-5 md:px-10 py-8 max-w-[860px] text-[13.5px] leading-relaxed text-foreground/85">
          <div className="label-xs text-accent">Scope statement</div>
          <h2 className="font-cond text-2xl font-semibold mt-1 text-foreground">A controlled synthetic benchmark with known landmarks</h2>
          <p className="mt-3">
            DriftGuard XR simulates a head-mounted camera (30 Hz) and IMU (200 Hz) moving through a field of 3D landmarks whose positions are known
            exactly. Every method receives the same corrupted measurement streams for a given seed. Ground truth is the simulated trajectory, held
            separately and never visible to the estimators. Degradations are applied between 30% and 70% of the sequence so that each run has a clean
            prefix, a stressed window, and a recovery tail.
          </p>

          <H>Estimators under test</H>
          <ul className="space-y-1.5">
            {METHOD_IDS.map((m) => (
              <li key={m} className="flex items-start gap-3">
                <span className="mt-2"><Swatch color={METHOD_META[m].color} /></span>
                <span><span className="text-foreground">{METHOD_META[m].label}</span> <span className="font-mono text-[10px] text-muted-foreground">{m}</span> — {METHOD_META[m].desc}</span>
              </li>
            ))}
          </ul>

          <H>Reprojection residual and pose Jacobian</H>
          <p>For a known landmark expressed in camera coordinates X_c, the pinhole projection uses the intrinsics returned with each experiment. The pose perturbation is camera-centred, which avoids dependence on the world origin.</p>
          <Eq n="1">π(X_c) = ( fx·X/Z + cx,  fy·Y/Z + cy ),   r_i = (u_i, v_i) − π(X_ci)   [px]</Eq>
          <Eq n="2">J_i = Dπ(X_ci) · [ −I,  [X_ci]× ]</Eq>
          <p>
            The SE(3) metric normalises translation by L = 1 m and rotation in radians, so the six eigenvector components read as (tx, ty, tz) per metre
            and (rx, ry, rz) in radians. These are camera-frame quantities; the GOI Explorer rotates the translation part into the world frame with the
            estimated camera orientation for 3D display.
          </p>

          <H>Curvature, observable subspace and conditioning</H>
          <Eq n="3">H_i = (w_i / σ²) J_iᵀ J_i,   H = Σ_i H_i = V Λ Vᵀ</Eq>
          <Eq n="4">observable: λ_k &gt; max(λ_max·10⁻⁶, 10⁻⁸),   H† = V_obs Λ_obs⁻¹ V_obsᵀ</Eq>
          <p>
            Observable rank counts the retained eigenvalues; the condition number and weakest eigenvector describe the least constrained direction.
            Translation and rotation observability are fractions of the curvature trace under this metric, not calibrated probabilities. Nothing
            here is a determinant criterion.
          </p>

          <H>GOI-inspired Geometric Observability Index diagnostics</H>
          <Eq n="5">ℓ_i = tr(H† H_i),   g_i = J_iᵀ w_i r_i / σ²</Eq>
          <Eq n="6">GOI_i = ‖ H† g_i ‖ / max(1 − ℓ_i, 0.001)</Eq>
          <p>
            This is an observable-subspace deletion-influence approximation: how far the pose would move if feature i were removed, inflated by its
            leverage ℓ_i. It is GOI-inspired, not a verified reproduction of a published Geometric Observability Index. It is{' '}
            <span className="text-foreground">not an outlier score</span>. Outliers are classified separately (normalised residual &gt; 4 with a
            MAD-based pixel scale, floor 0.7 px; Huber weights), so a valid peripheral feature can carry high influence and a strongly downweighted
            outlier can carry little. The interface keeps the two on independent visual channels.
          </p>

          <H>Baselines</H>
          <p>
            The fixed baseline is an ESKF with <span className="text-foreground">fixed process and visual covariance</span> and a fixed nine-frame
            visual update interval. Adaptive VIO adapts visual covariance to measured health and gates innovations at χ²(6) 99.9% = 22.458, still on the
            fixed interval. DriftGuard adds geometry-aware covariance heuristics and observability-driven keyframe selection (≈10 Hz cap, 0.3 s healthy
            fallback). Update budgets therefore differ between methods and must be considered when comparing accuracy and latency.
          </p>

          <H>Uncertainty reporting</H>
          <p>
            Position and orientation uncertainty are the <span className="text-foreground">square root of the trace</span> of the corresponding
            covariance block (m, deg). They are scalar summaries, not calibrated per-axis 1σ confidence intervals. The Analysis view overlays actual
            error so over-confidence is visible rather than hidden.
          </p>

          <H>Metrics</H>
          <Eq n="7">ATE = √( (1/N) Σ_k ‖p̂_k − p_k‖² )   [m], no alignment or scale fit</Eq>
          <Eq n="8">RPE = √( (1/M) Σ_k ‖ R̂_kᵀ(p̂_k+Δ − p̂_k) − R_kᵀ(p_k+Δ − p_k) ‖² ),  Δ = 1 s</Eq>
          <p>RPE is an RMSE of relative translation in local pose coordinates at a one-second lag (shortened only for sequences under 1 s).</p>
          <Eq n="9">drift = ‖p̂_N − p_N‖ / path length × 100   [%]</Eq>
          <p>
            <span className="text-foreground">Availability</span> is not "a pose was produced": Vision counts frames with a valid PnP solution; filters
            count frames whose position uncertainty is ≤ 0.75 m. It is not a claim of actual accuracy.
          </p>
          <p className="mt-2">
            <span className="text-foreground">Recovery duration</span> runs from the onset of a degraded episode to the validated return to the NORMAL
            state, which requires reliable, consistent visual corrections for 0.2 s. It is driven by the health state machine, never by the true
            error. It is null when no episode occurred or the run ended before validated recovery.
            Vision-only instead reports completed gaps between a failed PnP frame and the first returning valid pose.
          </p>
          <p className="mt-2">
            The <span className="text-foreground">virtual anchor</span> on the Live view is an evaluation-only quantity: a fixed true landmark is
            re-placed through the DriftGuard pose error (translation and orientation) to show how far a world-locked overlay would slip. Ground truth
            is never an estimator input.
          </p>

          <H>Honest limitations</H>
          {limitations.length ? (
            <ul className="space-y-1.5" data-testid="list-limitations">
              {limitations.map((l, i) => (
                <li key={i} className="flex gap-3"><span className="font-mono text-[10px] text-accent mt-1">{String(i + 1).padStart(2, '0')}</span><span>{l}</span></li>
              ))}
            </ul>
          ) : (
            <p className="text-muted-foreground">Limitations are reported per run by the backend; load an experiment to see them.</p>
          )}
          <p className="mt-3 text-muted-foreground">
            In addition: landmarks are known and data association is given, so front-end failure modes (tracking loss, mismatched descriptors,
            map drift) are modelled only through the explicit corruption parameters. Findings describe behaviour in this simulator.
          </p>
        </article>

        <aside className="bg-background flex flex-col gap-px">
          <Panel title="Current run provenance" meta={experiment ? experiment.id.slice(0, 8) : 'no run loaded'} className="border-0">
            {experiment ? (
              <dl className="grid grid-cols-[110px_1fr] gap-y-1 px-3 py-2 font-mono text-[11px]">
                <dt className="text-muted-foreground">id</dt><dd className="break-all">{experiment.id}</dd>
                <dt className="text-muted-foreground">created</dt><dd>{new Date(experiment.createdAt).toISOString()}</dd>
                <dt className="text-muted-foreground">preset</dt><dd>{experiment.config.preset}</dd>
                <dt className="text-muted-foreground">seed</dt><dd>{experiment.config.seed}</dd>
                <dt className="text-muted-foreground">duration</dt><dd>{experiment.config.duration} s</dd>
                <dt className="text-muted-foreground">motion</dt><dd>{experiment.config.motion}</dd>
                <dt className="text-muted-foreground">intrinsics</dt>
                <dd>{experiment.camera.width}×{experiment.camera.height} fx {experiment.camera.fx} fy {experiment.camera.fy}</dd>
              </dl>
            ) : (
              <div className="p-3 font-mono text-[11px] text-muted-foreground">Waiting for an experiment.</div>
            )}
          </Panel>
          <Panel title="Measurement fingerprint" meta="identical inputs across methods" className="border-0">
            <pre className="px-3 py-2 font-mono text-[10.5px] text-primary/90 whitespace-pre-wrap break-all" data-testid="text-fingerprint">{fpText ?? 'not reported'}</pre>
          </Panel>
          {ablations.length > 0 && (
            <Panel title="Ablation definitions" className="border-0">
              <ul className="px-3 py-2 space-y-1.5 text-[12px] text-muted-foreground">
                {ablations.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </Panel>
          )}
        </aside>
      </div>
    </div>
  );
}
