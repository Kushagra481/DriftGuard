import { useMemo, useState } from 'react';
import { Link } from 'wouter';
import { SlidersHorizontal } from 'lucide-react';
import { useGetFrame, getGetFrameQueryKey, type Experiment } from '@workspace/api-client-react';
import { ExperimentGate, Panel, Readout, Swatch } from '@/components/instrument';
import { PlaybackBar } from '@/components/playback';
import { Trajectory3D, computeAnchor } from '@/components/trajectory-3d';
import { CameraView, CameraLegend, type ColorMode } from '@/components/camera-view';
import { useExperimentCtx } from '@/lib/experiment-context';
import { METHOD_IDS, METHOD_META, TRUTH_COLOR, fmt, pct, methodColor } from '@/lib/format';
import { Skeleton } from '@/components/ui/skeleton';

const RES_SCALE = 4;

function LiveInner({ experiment }: { experiment: Experiment }) {
  const { frameIndex } = useExperimentCtx();
  const [visible, setVisible] = useState<Record<string, boolean>>(() => Object.fromEntries(METHOD_IDS.map((m) => [m, true])));
  const [landmarks, setLandmarks] = useState(true);
  const [mode, setMode] = useState<ColorMode>('goi');
  const [sel, setSel] = useState<number | null>(null);
  const id = experiment.id;
  const detail = useGetFrame(id, frameIndex, {
    query: { enabled: !!id, queryKey: getGetFrameQueryKey(id, frameIndex), placeholderData: (p) => p, staleTime: Infinity },
  });
  const f = experiment.frames[frameIndex];
  const [showAnchor, setShowAnchor] = useState(true);
  const anchor = useMemo(() => computeAnchor(experiment, frameIndex, 'driftguard'), [experiment, frameIndex]);
  const d = detail.data;
  const cfg = experiment.config;
  const nMissing = d?.features.filter((x) => x.missing).length ?? 0;
  const nOut = d?.features.filter((x) => x.outlier).length ?? 0;

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2 border-b border-border font-mono text-[11px]" data-testid="text-run-config">
        <span className="text-muted-foreground">run</span>
        <span className="text-foreground">{id.slice(0, 8)}</span>
        <span><span className="text-muted-foreground">preset </span>{cfg.preset ?? '—'}</span>
        <span><span className="text-muted-foreground">seed </span>{cfg.seed ?? '—'}</span>
        <span><span className="text-muted-foreground">T </span>{cfg.duration ?? '—'} s</span>
        <span><span className="text-muted-foreground">motion </span>{cfg.motion ?? '—'}</span>
        <span><span className="text-muted-foreground">frames </span>{experiment.frames.length}</span>
        <span><span className="text-muted-foreground">landmarks </span>{experiment.landmarks.length}</span>
        <Link href="/sensors" className="ml-auto inline-flex items-center gap-1.5 text-primary hover:underline" data-testid="link-configure">
          <SlidersHorizontal className="w-3 h-3" /> configure / re-run
        </Link>
      </div>

      <div className="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] xl:grid-rows-[minmax(0,1fr)] gap-px bg-border md:overflow-y-auto xl:overflow-hidden">
        <Panel
          title="Trajectory · world frame"
          meta="OpenCV axes remapped y-up for display"
          className="min-h-[420px] md:min-h-[460px] xl:min-h-0 border-0"
          bodyClass="relative"
          actions={
            <>
            <button
              onClick={() => setShowAnchor(!showAnchor)}
              className={`font-mono text-[10px] px-2 h-5 border ${showAnchor ? 'border-accent/60 text-accent' : 'border-border text-muted-foreground'}`}
              data-testid="toggle-anchor"
            >
              anchor
            </button>
            <button
              onClick={() => setLandmarks(!landmarks)}
              className={`font-mono text-[10px] px-2 h-5 border ${landmarks ? 'border-primary/50 text-primary' : 'border-border text-muted-foreground'}`}
              data-testid="toggle-landmarks"
            >
              landmarks
            </button>
            </>
          }
        >
          <div className="absolute inset-0">
            <Trajectory3D experiment={experiment} frameIndex={frameIndex} visible={visible} showLandmarks={landmarks} anchor={showAnchor ? anchor : null} />
          </div>
          <div className="absolute top-2 left-2 bg-background/85 border border-border p-2 space-y-1">
            <div className="flex items-center gap-2 font-mono text-[10px]">
              <Swatch color={TRUTH_COLOR} dashed /> <span>ground truth</span>
            </div>
            {METHOD_IDS.map((m) => (
              <label key={m} className="flex items-center gap-2 font-mono text-[10px] cursor-pointer select-none" data-testid={`toggle-method-${m}`}>
                <input type="checkbox" className="accent-[#55d3e0] w-3 h-3" checked={visible[m]} onChange={() => setVisible({ ...visible, [m]: !visible[m] })} />
                <Swatch color={METHOD_META[m].color} />
                <span className={visible[m] ? '' : 'text-muted-foreground'}>{METHOD_META[m].label}</span>
                <span className="ml-auto pl-3 tnum text-muted-foreground">{f?.errors[m] !== undefined ? `${fmt(f.errors[m], 3)} m` : '—'}</span>
              </label>
            ))}
          </div>
          {showAnchor && anchor && (
            <div className="absolute top-2 right-2 bg-background/85 border border-accent/40 p-2 font-mono text-[10px] leading-[1.6] max-w-[250px]" data-testid="text-anchor-drift">
              <div className="text-accent tracking-wider">VIRTUAL ANCHOR · EVALUATION ONLY</div>
              <div>drift <span className="text-[#55d3e0] text-[13px] tnum">{fmt(anchor.drift, 4)}</span> m</div>
              <div className="text-muted-foreground">Δ [{anchor.estimate.map((v, i) => fmt(v - anchor.truth[i], 3)).join(', ')}] m</div>
              <div className="text-muted-foreground leading-snug mt-1 font-sans text-[10.5px]">
                Wireframe = fixed true landmark. Solid = where a world-locked hologram lands using the DriftGuard pose error (translation + orientation). Ground truth is used for this comparison only; it is never an estimator input.
              </div>
            </div>
          )}
          <div className="absolute bottom-2 left-2 font-mono text-[9px] text-muted-foreground">drag orbit · scroll zoom · paths drawn to current replay frame</div>
        </Panel>

        <div className="flex flex-col gap-px bg-border min-h-0 xl:overflow-y-auto">
          <Panel
            title="Camera observation"
            meta={d ? `${experiment.camera.width}×${experiment.camera.height} · fx ${fmt(experiment.camera.fx, 1)}` : 'loading frame'}
            bodyClass="flex flex-col"
            className="border-0"
            actions={
              <div className="flex">
                {(['goi', 'residual', 'weight'] as ColorMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    data-testid={`button-mode-${m}`}
                    className={`font-mono text-[10px] px-2 h-5 border -ml-px ${mode === m ? 'border-primary/60 text-primary z-10' : 'border-border text-muted-foreground'}`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            }
          >
            <div className="p-2" style={{ aspectRatio: `${experiment.camera.width} / ${experiment.camera.height}` }}>
              {d ? (
                <CameraView camera={experiment.camera} features={d.features} mode={mode} residualScale={RES_SCALE} selected={sel} onSelect={setSel} />
              ) : detail.error ? (
                <div className="h-full grid place-items-center font-mono text-[11px] text-destructive">frame diagnostics unavailable</div>
              ) : (
                <Skeleton className="w-full h-full rounded-none" />
              )}
            </div>
            <div className="px-3 pb-2">
              <CameraLegend residualScale={RES_SCALE} />
            </div>
          </Panel>

          <Panel title="Frame state" meta={f ? `t = ${fmt(f.time, 3)} s` : ''} className="border-0">
            {f && (
              <div>
                <div className="px-3 py-2 border-b border-card-border flex flex-wrap items-center gap-x-4 gap-y-1">
                  <span className="font-mono text-[11px]" data-testid="text-state">
                    <span className="text-muted-foreground">state </span>
                    <span className={f.degraded ? 'text-accent' : 'text-primary'}>{f.state}</span>
                  </span>
                  <span className="font-mono text-[11px] truncate" data-testid="text-cause">
                    <span className="text-muted-foreground">cause </span>
                    {f.cause || '—'}
                  </span>
                  {f.keyframe && <span className="font-mono text-[10px] px-1.5 border border-primary/50 text-primary">KF · {f.keyframeReason}</span>}
                  {f.fixedKeyframe && <span className="font-mono text-[10px] px-1.5 border border-border text-muted-foreground">fixed-KF</span>}
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-4 divide-x divide-y divide-card-border [&>*]:border-card-border">
                  <Readout label="√tr Σ position" value={fmt(f.positionUncertainty, 4)} unit="m" tone="#a68cff" />
                  <Readout label="√tr Σ orient." value={fmt(f.orientationUncertainty, 3)} unit="deg" tone="#a68cff" />
                  <Readout label="Visual quality" value={pct(f.visualQuality)} />
                  <Readout label="IMU quality" value={pct(f.imuQuality)} />
                  <Readout label="Inlier ratio" value={pct(f.inlierRatio)} />
                  <Readout label="Reproj. error" value={fmt(f.reprojectionError, 3)} unit="px" />
                  <Readout label="Features" value={f.featureCount} sub={d ? `${nOut} outl · ${nMissing} miss` : undefined} />
                  <Readout label="Cond. number" value={fmt(f.conditionNumber, 1)} />
                  <Readout label="Obs. transl." value={fmt(f.translationObservability, 3)} />
                  <Readout label="Obs. rot." value={fmt(f.rotationObservability, 3)} />
                  <Readout label="GOI mean" value={fmt(f.goiMean, 4)} />
                  <Readout label="Latency" value={fmt(f.latency, 2)} unit="ms" />
                </div>
              </div>
            )}
          </Panel>

          <Panel title="Sequence metrics" meta="full replay, per method" className="border-0" bodyClass="overflow-x-auto">
            <table className="w-full font-mono text-[11px] tnum" data-testid="table-metrics">
              <thead>
                <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                  <th className="text-left font-normal px-3 py-1.5">method</th>
                  <th className="text-right font-normal px-2">ATE m</th>
                  <th className="text-right font-normal px-2">RPE m</th>
                  <th className="text-right font-normal px-2">rot °</th>
                  <th className="text-right font-normal px-2">drift %</th>
                  <th className="text-right font-normal px-2">avail %</th>
                  <th className="text-right font-normal px-3">recov s</th>
                </tr>
              </thead>
              <tbody>
                {experiment.metrics.map((m) => (
                  <tr key={m.id} className="border-t border-card-border" data-testid={`row-metric-${m.id}`}>
                    <td className="px-3 py-1.5 flex items-center gap-2">
                      <Swatch color={methodColor(m.id)} /> {m.name}
                    </td>
                    <td className="text-right px-2">{fmt(m.ate, 3)}</td>
                    <td className="text-right px-2">{fmt(m.rpe, 4)}</td>
                    <td className="text-right px-2">{fmt(m.rotationError, 2)}</td>
                    <td className="text-right px-2">{fmt(m.drift, 2)}</td>
                    <td className="text-right px-2">{fmt(m.availability, 1)}</td>
                    <td className="text-right px-3">{m.recoveryDuration === null ? <span className="text-muted-foreground">n/r</span> : fmt(m.recoveryDuration, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="px-3 py-1.5 font-mono text-[9px] text-muted-foreground border-t border-card-border">n/r = null: no degraded episode, or no validated return to NORMAL before sequence end</div>
          </Panel>
        </div>
      </div>
      <PlaybackBar experiment={experiment} />
    </div>
  );
}

export default function LivePage() {
  return <ExperimentGate>{(e) => <LiveInner experiment={e} />}</ExperimentGate>;
}
