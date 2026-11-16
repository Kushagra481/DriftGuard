import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useGetBenchmarks, getGetBenchmarksQueryKey, type Experiment, type MethodMetrics } from '@workspace/api-client-react';
import { ExperimentGate, Panel, ErrorPanel, SkeletonGrid, Swatch } from '@/components/instrument';
import { AXIS, GRID, InstrumentTooltip } from '@/components/chart-kit';
import { fmt, methodColor, methodShort, hasMixo } from '@/lib/format';
import { MixoBaselineTable } from '@/components/mixo-kit';
import { AlertTriangle } from 'lucide-react';
import { Link } from 'wouter';

type Key = 'ate' | 'rpe' | 'translationError' | 'rotationError' | 'drift' | 'availability' | 'meanLatency' | 'medianLatency' | 'p95Latency' | 'recoveryDuration' | 'keyframes';

const COLS: { key: Key; label: string; unit: string; better: 'low' | 'high' | 'none'; d: number }[] = [
  { key: 'ate', label: 'ATE', unit: 'm', better: 'low', d: 4 },
  { key: 'rpe', label: 'RPE', unit: 'm', better: 'low', d: 4 },
  { key: 'translationError', label: 'Transl. err', unit: 'm', better: 'low', d: 4 },
  { key: 'rotationError', label: 'Rot. err', unit: 'deg', better: 'low', d: 3 },
  { key: 'drift', label: 'Drift', unit: '% path', better: 'low', d: 2 },
  { key: 'availability', label: 'Availability', unit: '%', better: 'high', d: 1 },
  { key: 'recoveryDuration', label: 'Recovery', unit: 's', better: 'low', d: 2 },
  { key: 'meanLatency', label: 'Lat. mean', unit: 'ms', better: 'low', d: 2 },
  { key: 'medianLatency', label: 'Lat. p50', unit: 'ms', better: 'low', d: 2 },
  { key: 'p95Latency', label: 'Lat. p95', unit: 'ms', better: 'low', d: 2 },
  { key: 'keyframes', label: 'Keyframes', unit: 'n', better: 'none', d: 0 },
];

const ABLATIONS: { from: string; to: string; label: string }[] = [
  { from: 'vision', to: 'fixed', label: 'Add IMU fusion (fixed covariance)' },
  { from: 'fixed', to: 'adaptive', label: 'Residual-adaptive weighting' },
  { from: 'adaptive', to: 'driftguard', label: 'Geometry covariance + observability keyframing' },
  { from: 'imu', to: 'driftguard', label: 'Full stack vs inertial only' },
];

function MetricBars({ rows, k, unit }: { rows: MethodMetrics[]; k: Key; unit: string }) {
  const data = rows.map((r) => ({ id: r.id, name: methodShort(r.id), v: r[k] }));
  return (
    <div className="h-[170px]">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" {...AXIS} tickFormatter={(v) => fmt(v, 2)} />
          <YAxis type="category" dataKey="name" {...AXIS} width={40} />
          <Tooltip content={<InstrumentTooltip unit={unit} labelFmt={(l) => String(l)} />} isAnimationActive={false} cursor={{ fill: 'rgba(85,211,224,0.05)' }} />
          <Bar dataKey="v" name={k} isAnimationActive={false} barSize={12}>
            {data.map((d) => <Cell key={d.id} fill={methodColor(d.id)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const ISOLATION: Record<string, { label: string; note: string }> = {
  diagnostics_only: { label: 'Diagnostics only', note: 'Pose identical to Adaptive; GOI and observability are computed but never fed back. Isolates the compute overhead of the diagnostics.' },
  geometry_fixed: { label: 'Geometry covariance, fixed interval', note: 'Uses geometry-aware measurement covariance but keeps the fixed keyframe cadence. Isolates the weighting contribution.' },
  keyframes_basic: { label: 'Observability keyframes, basic covariance', note: 'Selects keyframes by observability but keeps the basic covariance. Isolates the keyframing contribution.' },
};

function isMetrics(x: unknown): x is MethodMetrics {
  return !!x && typeof x === 'object' && typeof (x as MethodMetrics).id === 'string' && typeof (x as MethodMetrics).ate === 'number';
}

function readAblations(meta: Record<string, unknown>) {
  const rows = Array.isArray(meta.ablationMetrics) ? meta.ablationMetrics.filter(isMetrics) : [];
  const notes = Array.isArray(meta.ablations) ? meta.ablations.filter((s): s is string => typeof s === 'string') : [];
  return { rows, notes };
}

function BenchInner({ experiment }: { experiment: Experiment }) {
  const id = experiment.id;
  const q = useGetBenchmarks(id, { query: { enabled: !!id, queryKey: getGetBenchmarksQueryKey(id), staleTime: Infinity } });
  const [chart, setChart] = useState<Key>('ate');
  const rows = useMemo(() => q.data ?? [], [q.data]);
  const best = useMemo(() => {
    const b: Partial<Record<Key, number>> = {};
    for (const c of COLS) {
      if (c.better === 'none') continue;
      const vals = rows.map((r) => r[c.key]).filter((v): v is number => v !== null && Number.isFinite(v));
      if (vals.length) b[c.key] = c.better === 'low' ? Math.min(...vals) : Math.max(...vals);
    }
    return b;
  }, [rows]);

  if (q.isLoading) return <SkeletonGrid />;
  if (q.error) return <ErrorPanel title="Benchmarks unavailable" error={q.error} onRetry={() => q.refetch()} />;
  if (!rows.length) return <div className="flex-1 grid place-items-center font-mono text-[12px] text-muted-foreground">No benchmark rows returned for this experiment.</div>;

  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  const col = COLS.find((c) => c.key === chart)!;
  const cfg = experiment.config;
  const iso = readAblations(experiment.metadata);
  const adaptive = byId.adaptive;
  const dgx = byId.driftguard;

  return (
    <div className="flex-1 overflow-auto grid grid-cols-1 xl:grid-cols-2 gap-px bg-border auto-rows-min">
      {byId.mixo ? (
        <div className="xl:col-span-2 bg-[#e8b86a]/[0.06] border-l-2 border-[#e8b86a] px-4 py-2.5 flex gap-3" data-testid="status-hardware-confound">
          <AlertTriangle className="w-4 h-4 text-[#e8b86a] shrink-0 mt-0.5" />
          <div className="text-[12px] leading-snug">
            <span className="font-cond font-semibold uppercase tracking-[0.12em] text-[#e8b86a] text-[11px]">Hardware-confounded row</span>
            <p className="text-muted-foreground mt-0.5">
              MIXO-inspired 6-DoF reads <span className="text-foreground">four cameras + IMU</span>; the other five methods read <span className="text-foreground">one front camera + IMU</span>.
              A lower MXO error in this table reflects extra sensors as much as the method, and is not evidence of equal-sensor superiority.
              For a fair same-rig comparison use the gate baselines below, which share one frozen input fingerprint.
            </p>
          </div>
        </div>
      ) : (
        <div className="xl:col-span-2 bg-secondary/40 px-4 py-2 font-mono text-[11px] text-muted-foreground flex flex-wrap gap-3" data-testid="status-no-mixo-bench">
          This run has no MIXO-inspired metrics; only the original five methods are compared.
          <Link href="/sensors" className="text-primary hover:underline">re-run to include it</Link>
        </div>
      )}
      <Panel
        title="Same-input comparison"
        meta={`identical front-camera + IMU streams${byId.mixo ? ' (MXO also reads left/right/rear)' : ''} · preset ${cfg.preset} · seed ${cfg.seed} · ${cfg.duration}s · ${cfg.motion}`}
        className="border-0 xl:col-span-2"
        bodyClass="overflow-x-auto"
      >
        <table className="w-full font-mono text-[11px] tnum" data-testid="table-benchmarks">
          <thead>
            <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
              <th className="text-left font-normal px-3 py-2">method</th>
              {COLS.map((c) => (
                <th key={c.key} className="text-right font-normal px-2 whitespace-nowrap">
                  <button onClick={() => setChart(c.key)} className={`uppercase ${chart === c.key ? 'text-primary' : 'hover:text-foreground'}`} data-testid={`button-chart-${c.key}`}>
                    {c.label}
                  </button>
                  <div className="normal-case tracking-normal text-[9px] opacity-70">{c.unit}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-card-border" data-testid={`row-benchmark-${r.id}`}>
                <td className="px-3 py-1.5 whitespace-nowrap">
                  <span className="inline-flex items-center gap-2"><Swatch color={methodColor(r.id)} />{r.name}</span>
                </td>
                {COLS.map((c) => {
                  const v = r[c.key];
                  const isBest = v !== null && best[c.key] !== undefined && v === best[c.key];
                  return (
                    <td key={c.key} className={`text-right px-2 ${isBest ? 'text-primary' : ''}`}>
                      {v === null ? <span className="text-muted-foreground">n/r</span> : fmt(v, c.d)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="px-3 py-1.5 border-t border-card-border font-mono text-[9px] text-muted-foreground">
          cyan = best in column for this single seed. One seed is one sample; re-run with other seeds before drawing conclusions. n/r = recovery null: no degraded episode, or no validated return to NORMAL before sequence end.
        </div>
      </Panel>

      <Panel title={`${col.label} by method`} meta={col.unit} className="border-0">
        <div className="p-2"><MetricBars rows={rows} k={chart} unit={col.unit} /></div>
      </Panel>

      <Panel title="Latency distribution" meta="mean / p50 / p95 ms" className="border-0">
        <div className="p-2 h-[186px]">
          <ResponsiveContainer>
            <BarChart data={rows.map((r) => ({ name: methodShort(r.id), mean: r.meanLatency, p50: r.medianLatency, p95: r.p95Latency }))} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} vertical={false} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis {...AXIS} tickFormatter={(v) => fmt(v, 1)} width={44} />
              <Tooltip content={<InstrumentTooltip unit="ms" labelFmt={(l) => String(l)} />} isAnimationActive={false} cursor={{ fill: 'rgba(85,211,224,0.05)' }} />
              <Bar dataKey="p50" name="p50" fill="#3f5a6e" isAnimationActive={false} />
              <Bar dataKey="mean" name="mean" fill="#55d3e0" isAnimationActive={false} />
              <Bar dataKey="p95" name="p95" fill="#a68cff" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel title="Ablation ladder" meta="Δ between methods on the same input · negative = improvement for error metrics" className="border-0 xl:col-span-2" bodyClass="overflow-x-auto">
        <table className="w-full font-mono text-[11px] tnum">
          <thead>
            <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
              <th className="text-left font-normal px-3 py-2">step</th>
              <th className="text-right font-normal px-2">Δ ATE m</th>
              <th className="text-right font-normal px-2">Δ ATE %</th>
              <th className="text-right font-normal px-2">Δ RPE m</th>
              <th className="text-right font-normal px-2">Δ rot °</th>
              <th className="text-right font-normal px-2">Δ avail pp</th>
              <th className="text-right font-normal px-3">Δ p95 lat ms</th>
            </tr>
          </thead>
          <tbody>
            {ABLATIONS.map((a) => {
              const A = byId[a.from];
              const B = byId[a.to];
              if (!A || !B) return null;
              const dA = B.ate - A.ate;
              const tone = (v: number, lowGood = true) => (Math.abs(v) < 1e-12 ? '' : (v < 0) === lowGood ? 'text-primary' : 'text-[#d9737e]');
              return (
                <tr key={a.label} className="border-t border-card-border" data-testid={`row-ablation-${a.from}-${a.to}`}>
                  <td className="px-3 py-1.5 whitespace-nowrap">
                    <span className="text-muted-foreground">{methodShort(a.from)} → {methodShort(a.to)}</span> <span className="ml-2 font-sans text-[12px]">{a.label}</span>
                  </td>
                  <td className={`text-right px-2 ${tone(dA)}`}>{dA >= 0 ? '+' : ''}{fmt(dA, 4)}</td>
                  <td className={`text-right px-2 ${tone(dA)}`}>{A.ate > 0 ? `${dA >= 0 ? '+' : ''}${fmt((dA / A.ate) * 100, 1)}%` : '—'}</td>
                  <td className={`text-right px-2 ${tone(B.rpe - A.rpe)}`}>{fmt(B.rpe - A.rpe, 4)}</td>
                  <td className={`text-right px-2 ${tone(B.rotationError - A.rotationError)}`}>{fmt(B.rotationError - A.rotationError, 3)}</td>
                  <td className={`text-right px-2 ${tone(B.availability - A.availability, false)}`}>{fmt(B.availability - A.availability, 1)}</td>
                  <td className={`text-right px-3 ${tone(B.p95Latency - A.p95Latency)}`}>{fmt(B.p95Latency - A.p95Latency, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>

      <Panel title="Isolation ablations" meta="metadata.ablationMetrics · same input, one mechanism changed at a time" className="border-0 xl:col-span-2" bodyClass="overflow-x-auto">
        {iso.rows.length ? (
          <table className="w-full font-mono text-[11px] tnum" data-testid="table-isolation">
            <thead>
              <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                <th className="text-left font-normal px-3 py-2">variant</th>
                <th className="text-right font-normal px-2">ATE m</th>
                <th className="text-right font-normal px-2">vs ADP</th>
                <th className="text-right font-normal px-2">vs DGX</th>
                <th className="text-right font-normal px-2">RPE m</th>
                <th className="text-right font-normal px-2">rot °</th>
                <th className="text-right font-normal px-2">avail %</th>
                <th className="text-right font-normal px-2">p95 ms</th>
                <th className="text-right font-normal px-2">recov s</th>
                <th className="text-right font-normal px-3">KF</th>
              </tr>
            </thead>
            <tbody>
              {[adaptive, ...iso.rows, dgx].filter(Boolean).map((r) => {
                const ref = !!byId[r.id];
                const info = ISOLATION[r.id];
                return (
                  <tr key={`iso-${r.id}`} className={`border-t border-card-border align-top ${ref ? 'text-muted-foreground' : ''}`} data-testid={`row-isolation-${r.id}`}>
                    <td className="px-3 py-1.5 max-w-[420px]">
                      <div className="flex items-center gap-2">
                        {ref ? <Swatch color={methodColor(r.id)} /> : <span className="w-3 h-[2px] bg-accent/70" />}
                        <span className={ref ? '' : 'text-foreground'}>{info?.label ?? r.name}</span>
                        <span className="text-[9px] opacity-60">{ref ? 'reference' : r.id}</span>
                      </div>
                      {info && <div className="font-sans text-[11px] text-muted-foreground leading-snug mt-0.5 pl-5">{info.note}</div>}
                    </td>
                    <td className="text-right px-2">{fmt(r.ate, 4)}</td>
                    <td className="text-right px-2">{adaptive && !ref ? `${r.ate - adaptive.ate >= 0 ? '+' : ''}${fmt(r.ate - adaptive.ate, 4)}` : '—'}</td>
                    <td className="text-right px-2">{dgx && !ref ? `${r.ate - dgx.ate >= 0 ? '+' : ''}${fmt(r.ate - dgx.ate, 4)}` : '—'}</td>
                    <td className="text-right px-2">{fmt(r.rpe, 4)}</td>
                    <td className="text-right px-2">{fmt(r.rotationError, 3)}</td>
                    <td className="text-right px-2">{fmt(r.availability, 1)}</td>
                    <td className="text-right px-2">{fmt(r.p95Latency, 2)}</td>
                    <td className="text-right px-2">{r.recoveryDuration === null ? 'n/r' : fmt(r.recoveryDuration, 2)}</td>
                    <td className="text-right px-3">{r.keyframes}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <div className="p-4 font-mono text-[11px] text-muted-foreground">This run did not report isolation ablations (metadata.ablationMetrics absent). Re-run to compute them.</div>
        )}
        {iso.notes.length > 0 && (
          <ul className="border-t border-card-border px-3 py-2 space-y-1">
            {iso.notes.map((n, i) => (
              <li key={i} className="text-[11.5px] text-muted-foreground leading-snug flex gap-2"><span className="font-mono text-[10px] text-accent">{String(i + 1).padStart(2, '0')}</span>{n}</li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Same-rig MIXO gate comparison" meta="experiment.mixo.baselines · learned vs equal vs matched-count" className="border-0 xl:col-span-2" bodyClass="overflow-x-auto">
        {hasMixo(experiment) ? (
          <MixoBaselineTable rows={experiment.mixo?.baselines ?? []} fingerprint={experiment.mixo?.measurementFingerprint} />
        ) : (
          <div className="p-4 font-mono text-[11px] text-muted-foreground">Not available for this run: it was computed without the four-camera rig.</div>
        )}
      </Panel>
    </div>
  );
}

export default function BenchmarksPage() {
  return <ExperimentGate>{(e) => <BenchInner experiment={e} />}</ExperimentGate>;
}
