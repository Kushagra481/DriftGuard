import { useMemo, type ReactNode } from 'react';
import { Link } from 'wouter';
import { AlertTriangle, ArrowRight, Terminal } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  useGetFrame,
  getGetFrameQueryKey,
  useGetMixoModel,
  getGetMixoModelQueryKey,
  type Experiment,
} from '@workspace/api-client-react';
import { ExperimentGate, Panel, Readout, Swatch } from '@/components/instrument';
import { PlaybackBar } from '@/components/playback';
import { AXIS, GRID, InstrumentTooltip, timeOverlays } from '@/components/chart-kit';
import { CameraWindowStrip, ExpertDiagRows, ExpertPreview, MixoBaselineTable } from '@/components/mixo-kit';
import { useExperimentCtx } from '@/lib/experiment-context';
import { CAMERA_IDS, CAMERA_META, cameraColor, fmt, hasMixo } from '@/lib/format';
import { CITATION, mixoFrameAt, readModel, weightKeys, type ModelReport } from '@/lib/mixo';
import { Skeleton } from '@/components/ui/skeleton';

function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground">{k}</dt>
      <dd className="text-foreground/90 break-all">{v}</dd>
    </>
  );
}

function seeds(s: number[]) {
  if (!s.length) return '—';
  return s.length > 6 ? `${s.slice(0, 6).join(', ')} … (${s.length})` : s.join(', ');
}

function ModelEvidence({ m, label, loading, error, onRetry }: { m: ModelReport | null; label: string; loading?: boolean; error?: unknown; onRetry?: () => void }) {
  if (loading) return <div className="p-3 space-y-2">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-4 rounded-none" />)}</div>;
  if (error)
    return (
      <div className="p-3 font-mono text-[11px] text-destructive flex items-center gap-3">
        model report unavailable
        {onRetry && <button onClick={onRetry} className="border border-border px-2 h-6 text-muted-foreground hover:text-foreground" data-testid="button-retry-model">retry</button>}
      </div>
    );
  if (!m) return <div className="p-3 font-mono text-[11px] text-muted-foreground">{label}: not reported.</div>;
  if (!m.trained)
    return (
      <div className="p-3 space-y-2" data-testid="status-model-untrained">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[#e8b86a]">
          <AlertTriangle className="w-3.5 h-3.5" /> status {m.status} · no training results exist
        </div>
        <p className="text-[12px] text-muted-foreground leading-snug">
          The gate artifact has not been produced. Training is an offline step run from the command line, not from this interface. No losses,
          seeds or comparisons are shown because none exist.
        </p>
        {m.command && <pre className="font-mono text-[11px] bg-background border border-border px-2 py-1.5 overflow-x-auto"><Terminal className="inline w-3 h-3 mr-1.5" />{m.command}</pre>}
      </div>
    );
  const losses: [string, number | undefined][] = [
    ['train (motion MSE)', m.trainingLoss],
    ['validation', m.validationLoss],
    ['test · learned', m.testLoss],
    ['test · equal weighting', m.uniformTestLoss],
    ['test · matched count', m.matchTestLoss],
  ];
  const maxL = Math.max(1e-12, ...losses.map((l) => l[1] ?? 0));
  return (
    <div className="p-3 space-y-3">
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-0.5 font-mono text-[10.5px] tnum">
        <KV k="status" v={<span className="text-primary">{m.status}</span>} />
        <KV k="architecture" v={m.architecture.length ? m.architecture.join(' → ') : '—'} />
        <KV k="epochs" v={m.epochs ?? '—'} />
        <KV k="samples" v={`${m.trainingSamples ?? '—'} train · ${m.validationSamples ?? '—'} val · ${m.testSamples ?? '—'} test`} />
        <KV k="train seeds" v={seeds(m.trainingSeeds)} />
        <KV k="val seeds" v={seeds(m.validationSeeds)} />
        <KV k="test seeds" v={seeds(m.testSeeds)} />
        <KV k="model hash" v={m.modelHash ?? '—'} />
        {m.source && <KV k="source" v={m.source} />}
        {m.command && <KV k="offline CLI" v={m.command} />}
      </dl>
      <div>
        <div className="label-xs mb-1">held-out losses · lower is better · as reported</div>
        {losses.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[150px_minmax(0,1fr)_80px] items-center gap-2 font-mono text-[10.5px] tnum">
            <span className="text-muted-foreground">{k}</span>
            <div className="h-2 bg-secondary/60">{v !== undefined && <div className="h-full" style={{ width: `${(v / maxL) * 100}%`, background: k.startsWith('test · learned') ? '#e8b86a' : '#55606e' }} />}</div>
            <span className="text-right">{fmt(v, 5)}</span>
          </div>
        ))}
        <div className="mt-1 text-[10.5px] text-muted-foreground leading-snug">
          Losses are mean squared error on six-dimensional tangent increments of simulated sequences from seed namespaces disjoint from runtime seeds. They
          describe this simulator only.
        </div>
      </div>
      {m.featureNames.length > 0 && (
        <div>
          <div className="label-xs mb-1">gate input features · {m.featureNames.length}</div>
          <div className="flex flex-wrap gap-1">
            {m.featureNames.map((f) => <span key={f} className="font-mono text-[9.5px] px-1.5 border border-border text-muted-foreground">{f}</span>)}
          </div>
        </div>
      )}
      {m.namespaces && (
        <div>
          <div className="label-xs mb-1">seed namespaces</div>
          <pre className="font-mono text-[10px] text-muted-foreground whitespace-pre-wrap break-all">{JSON.stringify(m.namespaces, null, 1)}</pre>
        </div>
      )}
      {m.warnings.length > 0 && (
        <ul className="space-y-1">
          {m.warnings.map((w, i) => <li key={i} className="flex gap-2 text-[11px] text-[#e8b86a] leading-snug"><AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />{w}</li>)}
        </ul>
      )}
    </div>
  );
}

function NoMixo({ experiment }: { experiment: Experiment }) {
  const c = useExperimentCtx();
  const model = useGetMixoModel({ query: { queryKey: getGetMixoModelQueryKey(), staleTime: 60000 } });
  return (
    <div className="flex-1 overflow-auto grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_420px] gap-px bg-border">
      <div className="bg-background grid place-items-center p-8 grid-bg">
        <div className="max-w-lg border border-card-border bg-card p-6 fade-in" data-testid="status-no-mixo-experts">
          <div className="label-xs text-[#e8b86a]">run {experiment.id.slice(0, 8)} · no expert record</div>
          <h2 className="font-cond text-xl font-semibold mt-1">This experiment has no MIXO-inspired estimate</h2>
          <p className="mt-2 text-[13px] text-muted-foreground leading-relaxed">
            It was computed before the four-camera expert estimator existed, or the backend did not report it. Per-frame weights, camera previews and
            gate diagnostics require a new computation, so none are shown for this saved run. Re-run the same configuration below;
            the original experiment stays in history.
          </p>
          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <button
              onClick={() => c.derive(experiment.config)}
              className="inline-flex items-center gap-2 h-8 px-3 bg-primary text-primary-foreground text-[12px] font-medium"
              data-testid="button-rerun-mixo"
            >
              Re-run with MIXO experts <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <Link href="/sensors" className="text-[12px] text-primary underline underline-offset-4" data-testid="link-run-mixo">
              Edit settings in Sensor Laboratory
            </Link>
          </div>
        </div>
      </div>
      <Panel title="Gate model · current artifact" meta="GET /api/mixo/model" className="border-0">
        <ModelEvidence m={readModel(model.data)} label="model" loading={model.isLoading} error={model.error} onRetry={() => model.refetch()} />
      </Panel>
    </div>
  );
}

function ExpertsInner({ experiment }: { experiment: Experiment }) {
  const { frameIndex, setFrameIndex } = useExperimentCtx();
  const mixo = experiment.mixo!;
  const id = experiment.id;
  const detail = useGetFrame(id, frameIndex, {
    query: { enabled: !!id, queryKey: getGetFrameQueryKey(id, frameIndex), placeholderData: (p) => p, staleTime: Infinity },
  });
  const model = useGetMixoModel({ query: { queryKey: getGetMixoModelQueryKey(), staleTime: 60000 } });
  const mf = mixoFrameAt(experiment, frameIndex);
  const keys = useMemo(() => weightKeys(experiment), [experiment]);
  const runModel = useMemo(() => readModel(mixo.training), [mixo.training]);
  const liveModel = readModel(model.data);
  const hashMismatch = !!runModel?.modelHash && !!liveModel?.modelHash && runModel.modelHash !== liveModel.modelHash;

  const weightData = useMemo(
    () => mixo.frames.map((f) => ({ t: f.time, ...Object.fromEntries(keys.map((k) => [k, f.weights?.[k] ?? 0])) })),
    [mixo.frames, keys],
  );
  const gateData = useMemo(
    () => mixo.frames.map((f) => ({ t: f.time, H: f.entropy, D: f.disagreement, pU: f.positionUncertainty, oU: f.orientationUncertainty, n: f.activeExperts })),
    [mixo.frames],
  );
  // A retained query result may belong to the previous replay frame.
  const views = detail.data?.frame.index === frameIndex ? detail.data.expertViews : undefined;
  const duration = experiment.frames[experiment.frames.length - 1]?.time ?? 0;
  const cursorFrac = duration > 0 && experiment.frames[frameIndex] ? experiment.frames[frameIndex].time / duration : undefined;
  const onClick = (s: { activeTooltipIndex?: number } | null) => {
    if (s && typeof s.activeTooltipIndex === 'number') {
      const f = mixo.frames[s.activeTooltipIndex];
      if (f) setFrameIndex(f.index);
    }
  };
  const holdover = mf ? mf.experts.filter((e) => e.id !== 'imu').every((e) => !e.available) : false;
  const camIds = CAMERA_IDS.filter((c) => mixo.experts.some((e) => e.id === c) || views?.some((v) => v.id === c));
  const imuDiagnostic = mf?.experts.find((e) => e.id === 'imu');

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2 border-b border-border font-mono text-[11px]" data-testid="text-mixo-strip">
        <span className="text-[#e8b86a]">MIXO-inspired 6-DoF</span>
        <span><span className="text-muted-foreground">rig </span>{camIds.length} cam + IMU</span>
        <span><span className="text-muted-foreground">mode </span>{experiment.config.rigDegradation ?? 'staggered'}</span>
        <span><span className="text-muted-foreground">fingerprint </span>{mixo.measurementFingerprint.slice(0, 16)}</span>
        <span><span className="text-muted-foreground">gate </span>{runModel?.status ?? 'unreported'}{runModel?.modelHash ? ` · ${runModel.modelHash.slice(0, 10)}` : ''}</span>
        {hashMismatch && <span className="text-[#e8b86a]" data-testid="status-hash-mismatch">run used a different gate artifact than the current one</span>}
      </div>

      <div className="flex-1 min-h-0 overflow-auto grid grid-cols-1 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-px bg-border auto-rows-min">
        <Panel title="Frame gate state" meta={mf ? `frame ${mf.index} · t = ${fmt(mf.time, 3)} s` : `frame ${frameIndex} · no gate record`} className="border-0 xl:col-span-2">
          {mf ? (
            <div>
              <div className="px-3 py-2 border-b border-card-border flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px]">
                <span><span className="text-muted-foreground">state </span><span className={mf.state === 'NORMAL' ? 'text-primary' : 'text-accent'} data-testid="text-mixo-state">{mf.state}</span></span>
                <span className="truncate"><span className="text-muted-foreground">cause </span>{mf.cause || '—'}</span>
                {holdover && <span className="px-1.5 border border-[#d9737e]/60 text-[#d9737e]" data-testid="status-holdover">INERTIAL HOLDOVER · no valid camera increment · uncertainty growing</span>}
                {mf.index === 0 && <span className="px-1.5 border border-border text-muted-foreground">initial frame · initialised from common first PnP, no relative increment</span>}
              </div>
              <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-8 divide-x divide-card-border">
                <Readout label="Active experts" value={`${mf.activeExperts}/${mf.experts.length}`} testId="text-active-experts" />
                <Readout label="Gate entropy" value={fmt(mf.entropy, 3)} unit="nat" />
                <Readout label="Disagreement" value={fmt(mf.disagreement, 4)} />
                <Readout label="Pos. uncertainty" value={fmt(mf.positionUncertainty, 4)} unit="m" tone="#a68cff" />
                <Readout label="Ang. uncertainty" value={fmt(mf.orientationUncertainty, 3)} unit="deg" tone="#a68cff" />
                <Readout label="Fused Δt" value={fmt(Math.hypot(...(mf.fusedIncrement ?? []).slice(0, 3)), 4)} unit="m" />
                <Readout label="Fused Δr" value={fmt((Math.hypot(...(mf.fusedIncrement ?? []).slice(3, 6)) * 180) / Math.PI, 3)} unit="deg" />
                <Readout label="Latency" value={fmt(mf.latency, 2)} unit="ms" />
              </div>
            </div>
          ) : (
            <div className="p-3 font-mono text-[11px] text-muted-foreground">No MIXO record for this replay frame.</div>
          )}
        </Panel>

        <Panel title="Expert cameras · current frame" meta="actual observed features from GET frame · expertViews" className="border-0 xl:col-span-2">
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-px bg-border">
            {camIds.map((cid) => {
              const v = views?.find((x) => x.id === cid);
              const dg = mf?.experts.find((e) => e.id === cid);
              return (
                <div key={cid} className="bg-card">
                  <div className="p-2">
                    {detail.isFetching && !views ? <Skeleton className="w-full aspect-[4/3] rounded-none" /> : <ExpertPreview view={v} diag={dg} />}
                  </div>
                  <ExpertDiagRows d={dg} />
                </div>
              );
            })}
          </div>
          {imuDiagnostic && (
            <div className="border-t border-card-border bg-card px-3 py-2 font-mono text-[10px] flex flex-wrap gap-x-5 gap-y-1" data-testid="text-imu-expert">
              <span className="text-muted-foreground">IMU expert · no camera preview</span>
              <span>weight {fmt(imuDiagnostic.weight, 3)}</span>
              <span>health {fmt(imuDiagnostic.quality, 3)}</span>
              <span>{imuDiagnostic.reason}</span>
              <span>{fmt(imuDiagnostic.latency, 2)} ms</span>
            </div>
          )}
          <div className="px-3 py-1.5 border-t border-card-border font-mono text-[9px] text-muted-foreground flex flex-wrap gap-x-4">
            <span>dot = inlier, opacity falls with residual</span>
            <span className="text-[#d9737e]">x = outlier</span>
            <span>hatched = masked, weight forced to 0</span>
            <span>previews show up to 80 actual observations; featureCount is the full count</span>
            {detail.error ? <span className="text-destructive">frame detail unavailable</span> : null}
            {!detail.isFetching && !detail.error && !views && <span className="text-[#e8b86a]">backend returned no expertViews for this frame</span>}
          </div>
        </Panel>

        <Panel title="Gate weights over time" meta="stacked · sums to 1 over available experts" className="border-0">
          <div className="p-2 h-[220px]">
            <ResponsiveContainer>
              <AreaChart data={weightData} margin={{ top: 6, right: 12, left: 0, bottom: 0 }} onClick={onClick}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => fmt(v, 1)} />
                <YAxis {...AXIS} domain={[0, 1]} width={36} tickFormatter={(v) => fmt(v, 1)} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} />
                {keys.map((k) => (
                  <Area key={k} dataKey={k} name={CAMERA_META[k]?.label ?? k} stackId="w" type="stepAfter" stroke={cameraColor(k)} fill={cameraColor(k)} fillOpacity={0.35} strokeWidth={1} isAnimationActive={false} />
                ))}
                {timeOverlays(experiment.frames, frameIndex)}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="px-3 pb-2 flex flex-wrap gap-3 font-mono text-[10px]">
            {keys.map((k) => <span key={k} className="inline-flex items-center gap-1.5"><Swatch color={cameraColor(k)} />{CAMERA_META[k]?.label ?? k}</span>)}
          </div>
        </Panel>

        <Panel title="Gate entropy · disagreement · uncertainty" meta="click to seek" className="border-0">
          <div className="p-2 h-[220px]">
            <ResponsiveContainer>
              <LineChart data={gateData} margin={{ top: 6, right: 12, left: 0, bottom: 0 }} onClick={onClick}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => fmt(v, 1)} />
                <YAxis yAxisId="a" {...AXIS} width={40} tickFormatter={(v) => fmt(v, 2)} />
                <YAxis yAxisId="b" orientation="right" {...AXIS} width={44} tickFormatter={(v) => fmt(v, 2)} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} />
                <Line yAxisId="a" dataKey="H" name="entropy nat" stroke="#e8b86a" dot={false} strokeWidth={1.3} isAnimationActive={false} />
                <Line yAxisId="a" dataKey="D" name="disagreement" stroke="#d9737e" dot={false} strokeWidth={1.1} isAnimationActive={false} />
                <Line yAxisId="b" dataKey="pU" name="√trΣ pos m" stroke="#a68cff" dot={false} strokeWidth={1.1} isAnimationActive={false} />
                <Line yAxisId="b" dataKey="oU" name="√trΣ ori deg" stroke="#7d9ccb" dot={false} strokeWidth={1.1} strokeDasharray="3 3" isAnimationActive={false} />
                {timeOverlays(experiment.frames, frameIndex, 'a')}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="px-3 pb-2 flex flex-wrap gap-3 font-mono text-[10px]">
            <span className="inline-flex items-center gap-1.5"><Swatch color="#e8b86a" />entropy (left)</span>
            <span className="inline-flex items-center gap-1.5"><Swatch color="#d9737e" />disagreement (left)</span>
            <span className="inline-flex items-center gap-1.5"><Swatch color="#a68cff" />position unc. (right)</span>
            <span className="inline-flex items-center gap-1.5"><Swatch color="#7d9ccb" dashed />orientation unc. (right)</span>
          </div>
        </Panel>

        <Panel title="Camera degradation windows" meta="actual cameraWindows from this run · fraction of duration" className="border-0">
          <CameraWindowStrip windows={mixo.cameraWindows ?? {}} cursor={cursorFrac} />
          <div className="px-3 pb-2 text-[10.5px] text-muted-foreground leading-snug">
            Simplification: each camera is degraded over one contiguous window with hard on/off edges. Real rigs degrade gradually and in correlated ways.
          </div>
        </Panel>

        <Panel title="Expert extrinsics" meta="camera-to-body · quaternion xyzw · translation m" className="border-0" bodyClass="overflow-x-auto">
          <table className="w-full font-mono text-[10.5px] tnum">
            <thead>
              <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                <th className="text-left font-normal px-3 py-1.5">expert</th>
                <th className="text-right font-normal px-2">q (x y z w)</th>
                <th className="text-right font-normal px-3">t (m)</th>
              </tr>
            </thead>
            <tbody>
              {mixo.experts.map((e) => (
                <tr key={e.id} className="border-t border-card-border" data-testid={`row-expert-${e.id}`}>
                  <td className="px-3 py-1"><span className="inline-flex items-center gap-2"><Swatch color={cameraColor(e.id)} />{e.name}</span></td>
                  <td className="text-right px-2">{e.quaternion.map((v) => fmt(v, 3)).join(' ')}</td>
                  <td className="text-right px-3">{e.translation.map((v) => fmt(v, 3)).join(' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Same-rig gate baselines" meta="experiment.mixo.baselines · identical frozen input" className="border-0 xl:col-span-2" bodyClass="overflow-x-auto">
          <MixoBaselineTable rows={mixo.baselines ?? []} fingerprint={mixo.measurementFingerprint} />
        </Panel>

        <Panel title="Training evidence · as used by this run" meta="experiment.mixo.training" className="border-0">
          <ModelEvidence m={runModel} label="run training report" />
        </Panel>
        <Panel title="Training evidence · current artifact" meta="GET /api/mixo/model" className="border-0">
          <ModelEvidence m={liveModel} label="model" loading={model.isLoading} error={model.error} onRetry={() => model.refetch()} />
        </Panel>

        <Panel title="Limitations reported by the backend" meta={`${(mixo.limitations ?? []).length} items`} className="border-0 xl:col-span-2">
          <ul className="px-3 py-2 space-y-1">
            {(mixo.limitations ?? []).map((l, i) => (
              <li key={i} className="flex gap-2 text-[12px] text-muted-foreground leading-snug">
                <span className="font-mono text-[10px] text-[#e8b86a]">{String(i + 1).padStart(2, '0')}</span>{l}
              </li>
            ))}
            <li className="flex gap-2 text-[12px] text-muted-foreground leading-snug border-t border-card-border pt-2 mt-2">
              <span className="font-mono text-[10px] text-[#e8b86a]">ref</span>Inspired by, not a reproduction of: {CITATION}
            </li>
          </ul>
        </Panel>
      </div>
      <PlaybackBar experiment={experiment} />
    </div>
  );
}

export default function ExpertsPage() {
  return <ExperimentGate>{(e) => (hasMixo(e) ? <ExpertsInner experiment={e} /> : <NoMixo experiment={e} />)}</ExperimentGate>;
}
