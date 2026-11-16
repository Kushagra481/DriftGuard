import { useMemo, useState } from 'react';
import { Bar, BarChart, Legend, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useGetFrame, getGetFrameQueryKey, type Experiment, type Frame } from '@workspace/api-client-react';
import { ExperimentGate, Panel, Readout } from '@/components/instrument';
import { PlaybackBar } from '@/components/playback';
import { CameraView, CameraLegend, type ColorMode } from '@/components/camera-view';
import { AXIS, GRID, InstrumentTooltip, timeOverlays } from '@/components/chart-kit';
import { useExperimentCtx } from '@/lib/experiment-context';
import { fmt, pct, METHOD_IDS } from '@/lib/format';
import { Trajectory3D } from '@/components/trajectory-3d';
import * as THREE from 'three';
import { Skeleton } from '@/components/ui/skeleton';

const AXIS_LABELS = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'];

function mean(xs: number[]) {
  const f = xs.filter((x) => Number.isFinite(x));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : NaN;
}

const PHASE_ROWS: { key: string; label: string; get: (f: Frame) => number; fmtFn: (v: number) => string }[] = [
  { key: 'goi', label: 'GOI mean', get: (f) => f.goiMean, fmtFn: (v) => fmt(v, 4) },
  { key: 'ig', label: 'Information gain', get: (f) => f.informationGain, fmtFn: (v) => fmt(v, 4) },
  { key: 'cond', label: 'Condition number', get: (f) => f.conditionNumber, fmtFn: (v) => fmt(v, 1) },
  { key: 'to', label: 'Translation observability', get: (f) => f.translationObservability, fmtFn: (v) => fmt(v, 3) },
  { key: 'ro', label: 'Rotation observability', get: (f) => f.rotationObservability, fmtFn: (v) => fmt(v, 3) },
  { key: 'inl', label: 'Inlier ratio', get: (f) => f.inlierRatio, fmtFn: (v) => pct(v) },
  { key: 'fc', label: 'Feature count', get: (f) => f.featureCount, fmtFn: (v) => fmt(v, 1) },
  { key: 'reproj', label: 'Reprojection error px', get: (f) => f.reprojectionError, fmtFn: (v) => fmt(v, 3) },
  { key: 'err', label: 'DriftGuard pos. error m', get: (f) => f.errors.driftguard ?? NaN, fmtFn: (v) => fmt(v, 4) },
  { key: 'sig', label: '√tr Σ position m', get: (f) => f.positionUncertainty, fmtFn: (v) => fmt(v, 4) },
];

interface GoiBin { label: string; lower: number; upper: number; before: number; during: number; after: number }

function readDistributions(meta: Record<string, unknown>): GoiBin[] {
  const raw = meta.goiDistributions;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (b): b is GoiBin =>
      !!b && typeof b === 'object' && typeof (b as GoiBin).label === 'string' && ['lower', 'upper', 'before', 'during', 'after'].every((k) => typeof (b as unknown as Record<string, unknown>)[k] === 'number'),
  );
}

const PHASE_COLORS = { before: '#7d9ccb', during: '#a68cff', after: '#55d3e0' };

function GoiInner({ experiment }: { experiment: Experiment }) {
  const { frameIndex, setFrameIndex } = useExperimentCtx();
  const [mode, setMode] = useState<ColorMode>('goi');
  const [sel, setSel] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<'goi' | 'residual' | 'weight'>('goi');
  const id = experiment.id;
  const frames = experiment.frames;
  const detail = useGetFrame(id, frameIndex, { query: { enabled: !!id, queryKey: getGetFrameQueryKey(id, frameIndex), placeholderData: (p) => p, staleTime: Infinity } });
  const d = detail.data;
  const f = frames[frameIndex];

  const series = useMemo(
    () => frames.map((x) => ({ t: x.time, goi: x.goiMean, ig: x.informationGain, cond: x.conditionNumber, to: x.translationObservability, ro: x.rotationObservability, kf: x.keyframe ? x.informationGain : null })),
    [frames],
  );

  const eig = useMemo(() => (f?.eigenvalues ?? []).map((v, i) => ({ i: `λ${i + 1}`, v: Math.max(v, 1e-12), raw: v })), [f]);
  const minEigIdx = eig.length ? eig.reduce((b, x, i) => (x.raw < eig[b].raw ? i : b), 0) : -1;

  const phases = useMemo(() => {
    const deg = frames.map((x, i) => (x.degraded ? i : -1)).filter((i) => i >= 0);
    if (!deg.length) return null;
    const a = deg[0];
    const b = deg[deg.length - 1];
    return { before: frames.slice(0, a), during: frames.slice(a, b + 1), after: frames.slice(b + 1) };
  }, [frames]);

  const feats = useMemo(() => [...(d?.features ?? [])].sort((a, b) => (a.missing === b.missing ? b[sortKey] - a[sortKey] : a.missing ? 1 : -1)), [d, sortKey]);
  const wd = d?.weakestDirection ?? [];
  const wdMax = Math.max(1e-12, ...wd.map((x) => Math.abs(x)));
  const [show3dTraj, setShow3dTraj] = useState(false);
  const noMethods = useMemo(() => Object.fromEntries(METHOD_IDS.map((m) => [m, show3dTraj && m === 'driftguard'])), [show3dTraj]);
  // Weakest eigenvector translation part is in the camera-centred frame; rotate to world with the estimated camera-to-world orientation.
  const weakWorld = useMemo(() => {
    if (!f || wd.length < 3) return null;
    const t = new THREE.Vector3(wd[0], wd[1], wd[2]);
    const tn = t.length();
    if (tn < 1e-9) return null;
    const q = (f.inlierRatio > 0 ? f.poses.vision : f.poses.driftguard).quaternion;
    const w = t.divideScalar(tn).applyQuaternion(new THREE.Quaternion(q[0], q[1], q[2], q[3]));
    return { dir: [w.x, w.y, w.z], transShare: tn };
  }, [f, wd]);
  const dist = useMemo(() => readDistributions(experiment.metadata), [experiment.metadata]);
  const [normDist, setNormDist] = useState(true);
  const distData = useMemo(() => {
    const tot = { before: 0, during: 0, after: 0 };
    dist.forEach((b) => { tot.before += b.before; tot.during += b.during; tot.after += b.after; });
    return dist.map((b) => ({
      label: b.label,
      range: `${fmt(b.lower, 4)} – ${fmt(b.upper, 4)}`,
      before: normDist ? (tot.before ? b.before / tot.before : 0) : b.before,
      during: normDist ? (tot.during ? b.during / tot.during : 0) : b.during,
      after: normDist ? (tot.after ? b.after / tot.after : 0) : b.after,
    }));
  }, [dist, normDist]);
  const distTotals = useMemo(() => dist.reduce((a, b) => ({ before: a.before + b.before, during: a.during + b.during, after: a.after + b.after }), { before: 0, during: 0, after: 0 }), [dist]);

  const onClick = (s: { activeTooltipIndex?: number } | null) => {
    if (s && typeof s.activeTooltipIndex === 'number') setFrameIndex(s.activeTooltipIndex);
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="px-4 py-2 border-b border-border text-[11.5px] text-muted-foreground">
        <span className="text-accent font-mono text-[10px] mr-2">NOTE</span>
        GOI-inspired Geometric Observability Index diagnostics: per-feature deletion influence in the observable subspace, ‖H† g_i‖ / max(1 − ℓ_i, 0.001). It attributes pose sensitivity and is
        <span className="text-foreground"> not an outlier score</span>: a perfectly valid feature can carry high influence, and a rejected outlier can carry none.
      </div>
      <div className="flex-1 overflow-auto grid grid-cols-1 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] gap-px bg-border auto-rows-min">
        <Panel
          title="Feature influence · image plane"
          meta={d ? `${d.features.length} features · frame ${frameIndex}` : 'loading'}
          className="border-0"
          actions={
            <div className="flex">
              {(['goi', 'residual', 'weight'] as ColorMode[]).map((m) => (
                <button key={m} onClick={() => setMode(m)} data-testid={`button-goi-mode-${m}`} className={`font-mono text-[10px] px-2 h-5 border -ml-px ${mode === m ? 'border-primary/60 text-primary z-10' : 'border-border text-muted-foreground'}`}>
                  {m}
                </button>
              ))}
            </div>
          }
        >
          <div className="p-2" style={{ aspectRatio: `${experiment.camera.width} / ${experiment.camera.height}` }}>
            {d ? <CameraView camera={experiment.camera} features={d.features} mode={mode} selected={sel} onSelect={setSel} /> : <Skeleton className="w-full h-full rounded-none" />}
          </div>
          <div className="px-3 pb-2"><CameraLegend residualScale={4} /></div>
        </Panel>

        <div className="flex flex-col gap-px bg-border">
          <Panel title="Observability at frame" meta={f ? `t ${fmt(f.time, 3)} s` : ''} className="border-0">
            <div className="grid grid-cols-3 divide-x divide-card-border border-b border-card-border">
              <Readout label="Observable rank" value={d ? `${d.observableRank}/6` : '—'} tone={d && d.observableRank < 6 ? '#a68cff' : undefined} testId="text-observable-rank" />
              <Readout label="Coverage" value={d ? pct(d.coverage) : '—'} />
              <Readout label="Parallax" value={d ? fmt(d.parallax, 4) : '—'} />
            </div>
            <div className="grid grid-cols-3 divide-x divide-card-border">
              <Readout label="Blur" value={d ? fmt(d.blur, 3) : '—'} />
              <Readout label="Cond. number" value={fmt(f?.conditionNumber, 1)} />
              <Readout label="Info gain" value={fmt(f?.informationGain, 4)} sub={f?.keyframe ? `keyframe · ${f.keyframeReason}` : 'not a keyframe'} />
            </div>
          </Panel>

          <Panel title="Information eigenspectrum" meta="log scale · weakest highlighted" className="border-0">
            <div className="h-[170px] p-2">
              {eig.length ? (
                <ResponsiveContainer>
                  <BarChart data={eig} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid {...GRID} vertical={false} />
                    <XAxis dataKey="i" {...AXIS} />
                    <YAxis {...AXIS} scale="log" domain={['auto', 'auto']} allowDataOverflow tickFormatter={(v) => fmt(v, 1)} width={52} />
                    <Tooltip content={<InstrumentTooltip labelFmt={(l) => String(l)} />} isAnimationActive={false} cursor={{ fill: 'rgba(85,211,224,0.05)' }} />
                    <Bar dataKey="v" name="eigenvalue" isAnimationActive={false}>
                      {eig.map((_, i) => <Cell key={i} fill={i === minEigIdx ? '#a68cff' : '#3f5a6e'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full grid place-items-center font-mono text-[11px] text-muted-foreground">no eigenvalues reported for this frame</div>
              )}
            </div>
          </Panel>

          <Panel title="Weakest direction" meta="eigenvector of smallest λ · camera-centred SE(3): t per 1 m, r in rad" className="border-0">
            <div className="p-3 space-y-1.5" data-testid="weakest-direction">
              {wd.length ? (
                wd.map((v, i) => (
                  <div key={i} className={`${i === 3 ? 'mt-2 pt-2 border-t border-card-border ' : ''}grid grid-cols-[28px_1fr_64px] items-center gap-2 font-mono text-[10.5px]`}>
                    <span className="text-muted-foreground">{AXIS_LABELS[i] ?? `d${i}`}</span>
                    <div className="relative h-2 bg-secondary">
                      <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border" />
                      <div
                        className="absolute top-0 bottom-0 bg-accent"
                        style={{ left: v >= 0 ? '50%' : `${50 - (Math.abs(v) / wdMax) * 50}%`, width: `${(Math.abs(v) / wdMax) * 50}%` }}
                      />
                    </div>
                    <span className="text-right tnum">{fmt(v, 3)}</span>
                  </div>
                ))
              ) : (
                <div className="font-mono text-[11px] text-muted-foreground">{d ? 'not reported' : 'loading'}</div>
              )}
            </div>
          </Panel>
        </div>

        <Panel
          title="3D geometry at frame"
          meta="camera, matched landmarks, weakest translation · click to select"
          className="border-0 xl:col-span-2"
          bodyClass="relative h-[420px]"
          actions={
            <button onClick={() => setShow3dTraj(!show3dTraj)} data-testid="toggle-goi-trajectory" className={`font-mono text-[10px] px-2 h-5 border ${show3dTraj ? 'border-primary/60 text-primary' : 'border-border text-muted-foreground'}`}>
              DGX path
            </button>
          }
        >
          {d ? (
            <div className="absolute inset-0">
              <Trajectory3D
                experiment={experiment}
                frameIndex={frameIndex}
                visible={noMethods}
                showLandmarks
                features={d.features}
                selectedFeature={sel}
                onSelectFeature={setSel}
                weakDirection={weakWorld?.dir ?? null}
                focusFrame
              />
            </div>
          ) : (
            <Skeleton className="absolute inset-0 rounded-none" />
          )}
          <div className="absolute top-2 left-2 bg-background/85 border border-border p-2 font-mono text-[10px] leading-[1.7] pointer-events-none">
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'linear-gradient(90deg,#465466,#a68cff,#55d3e0)' }} />color / size = GOI (influence)</div>
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full border-2 border-[#d9737e]" />ring = outlier (independent channel)</div>
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full border border-dashed border-muted-foreground" />wireframe = missing</div>
            <div className="flex items-center gap-2"><span className="w-3 h-0.5 bg-[#d4a24a]" />weakest translation, world frame</div>
          </div>
          {weakWorld && (
            <div className="absolute bottom-2 right-2 bg-background/85 border border-border px-2 py-1 font-mono text-[10px] pointer-events-none" data-testid="text-weak-world">
              <span className="text-muted-foreground">t_weak(world, OpenCV) </span>[{weakWorld.dir.map((v) => fmt(v, 3)).join(', ')}]
              <span className="text-muted-foreground"> · ‖t‖ in eigvec {fmt(weakWorld.transShare, 3)}</span>
            </div>
          )}
        </Panel>

        <Panel title="Frame influence over sequence" meta="GOI mean and information gain · keyframes as dots" className="border-0 xl:col-span-2">
          <div className="h-[200px] p-2">
            <ResponsiveContainer>
              <LineChart data={series} onClick={onClick} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis {...AXIS} tickFormatter={(v) => fmt(v, 2)} width={52} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex)}
                <Line dataKey="goi" name="GOI mean" stroke="#55d3e0" dot={false} isAnimationActive={false} />
                <Line dataKey="ig" name="info gain" stroke="#a68cff" dot={false} strokeWidth={1} isAnimationActive={false} />
                <Line dataKey="kf" name="keyframe" stroke="none" dot={{ r: 2.5, fill: '#e6dfcc', stroke: 'none' }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Conditioning & observability" meta="condition number (log, left) · observability (right)" className="border-0 xl:col-span-2">
          <div className="h-[200px] p-2">
            <ResponsiveContainer>
              <LineChart data={series} onClick={onClick} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis yAxisId="c" {...AXIS} scale="log" domain={['auto', 'auto']} allowDataOverflow tickFormatter={(v) => fmt(v, 0)} width={52} />
                <YAxis yAxisId="o" orientation="right" {...AXIS} tickFormatter={(v) => fmt(v, 2)} width={40} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex, 'c')}
                <Line yAxisId="c" dataKey="cond" name="cond. number" stroke="#d4a24a" dot={false} isAnimationActive={false} />
                <Line yAxisId="o" dataKey="to" name="transl. obs." stroke="#55d3e0" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
                <Line yAxisId="o" dataKey="ro" name="rot. obs." stroke="#a68cff" dot={false} strokeDasharray="1 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="Per-feature influence distribution"
          meta="metadata.goiDistributions · log-spaced GOI bins · before / during / after degradation"
          className="border-0 xl:col-span-2"
          actions={
            <button onClick={() => setNormDist(!normDist)} data-testid="toggle-dist-normalize" className={`font-mono text-[10px] px-2 h-5 border ${normDist ? 'border-primary/60 text-primary' : 'border-border text-muted-foreground'}`}>
              {normDist ? 'fraction of phase' : 'raw counts'}
            </button>
          }
        >
          {distData.length ? (
            <>
              <div className="h-[230px] p-2">
                <ResponsiveContainer>
                  <BarChart data={distData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} barCategoryGap="18%">
                    <CartesianGrid {...GRID} vertical={false} />
                    <XAxis dataKey="label" {...AXIS} interval={0} />
                    <YAxis {...AXIS} tickFormatter={(v) => (normDist ? `${(v * 100).toFixed(0)}%` : fmt(v, 0))} width={44} />
                    <Tooltip content={<InstrumentTooltip labelFmt={(l) => `GOI bin ${l} · ${distData.find((x) => x.label === l)?.range ?? ''}`} />} isAnimationActive={false} cursor={{ fill: 'rgba(85,211,224,0.05)' }} />
                    <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 10 }} />
                    <Bar dataKey="before" name={`before (n=${distTotals.before})`} fill={PHASE_COLORS.before} isAnimationActive={false} />
                    <Bar dataKey="during" name={`during (n=${distTotals.during})`} fill={PHASE_COLORS.during} isAnimationActive={false} />
                    <Bar dataKey="after" name={`after (n=${distTotals.after})`} fill={PHASE_COLORS.after} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="px-3 pb-2 font-mono text-[9px] text-muted-foreground">
                {normDist ? 'each phase normalised to its own feature-observation total so phases of different length compare as shapes' : 'raw per-feature observation counts; phases differ in length'}
              </div>
            </>
          ) : (
            <div className="p-4 font-mono text-[11px] text-muted-foreground">This run did not report metadata.goiDistributions. Re-run the experiment to compute per-feature influence distributions.</div>
          )}
        </Panel>

        <Panel title="Before / during / after · frame means" meta="means only; see distribution above" className="border-0" bodyClass="overflow-x-auto">
          {phases ? (
            <table className="w-full font-mono text-[11px] tnum" data-testid="table-phases">
              <thead>
                <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                  <th className="text-left font-normal px-3 py-1.5">diagnostic</th>
                  <th className="text-right font-normal px-2">before ({phases.before.length})</th>
                  <th className="text-right font-normal px-2 text-accent">during ({phases.during.length})</th>
                  <th className="text-right font-normal px-3">after ({phases.after.length})</th>
                </tr>
              </thead>
              <tbody>
                {PHASE_ROWS.map((r) => (
                  <tr key={r.key} className="border-t border-card-border">
                    <td className="px-3 py-1">{r.label}</td>
                    <td className="text-right px-2">{r.fmtFn(mean(phases.before.map(r.get)))}</td>
                    <td className="text-right px-2 text-accent">{r.fmtFn(mean(phases.during.map(r.get)))}</td>
                    <td className="text-right px-3">{r.fmtFn(mean(phases.after.map(r.get)))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-4 font-mono text-[11px] text-muted-foreground">No frames flagged as degraded in this run; phase comparison is not defined.</div>
          )}
        </Panel>

        <Panel
          title="Features at frame"
          meta="sorted, missing last"
          className="border-0"
          bodyClass="overflow-auto max-h-[360px]"
          actions={
            <div className="flex">
              {(['goi', 'residual', 'weight'] as const).map((k) => (
                <button key={k} onClick={() => setSortKey(k)} data-testid={`button-sort-${k}`} className={`font-mono text-[10px] px-2 h-5 border -ml-px ${sortKey === k ? 'border-primary/60 text-primary z-10' : 'border-border text-muted-foreground'}`}>
                  {k}
                </button>
              ))}
            </div>
          }
        >
          <table className="w-full font-mono text-[10.5px] tnum">
            <thead className="sticky top-0 bg-card">
              <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                <th className="text-left font-normal px-3 py-1.5">id</th>
                <th className="text-right font-normal px-2">GOI</th>
                <th className="text-right font-normal px-2">resid px</th>
                <th className="text-right font-normal px-2">weight</th>
                <th className="text-left font-normal px-3">status</th>
              </tr>
            </thead>
            <tbody>
              {feats.map((x) => (
                <tr
                  key={x.id}
                  onClick={() => setSel(x.id === sel ? null : x.id)}
                  className={`border-t border-card-border cursor-pointer ${x.id === sel ? 'bg-secondary' : 'hover:bg-secondary/50'} ${x.missing ? 'text-muted-foreground' : ''}`}
                  data-testid={`row-feature-${x.id}`}
                >
                  <td className="px-3 py-1">#{x.id}</td>
                  <td className="text-right px-2">{x.missing ? '—' : fmt(x.goi, 4)}</td>
                  <td className="text-right px-2">{x.missing ? '—' : fmt(x.residual, 3)}</td>
                  <td className="text-right px-2">{x.missing ? '—' : fmt(x.weight, 3)}</td>
                  <td className="px-3">
                    {x.missing ? 'missing' : x.outlier ? <span className="text-[#d9737e]">outlier</span> : <span className="text-primary/80">inlier</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
      <PlaybackBar experiment={experiment} />
    </div>
  );
}

export default function GoiPage() {
  return <ExperimentGate>{(e) => <GoiInner experiment={e} />}</ExperimentGate>;
}
