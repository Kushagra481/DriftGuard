import type { ExpertDiagnostic, ExpertView, MethodMetrics } from '@workspace/api-client-react';
import { CAMERA_META, MIXO_BASELINE_META, cameraColor, fmt, pct } from '@/lib/format';
import { Swatch } from '@/components/instrument';

/** Same-rig comparison: learned gate vs equal and matched-count weighting on one frozen input. */
export function MixoBaselineTable({ rows, fingerprint }: { rows: MethodMetrics[]; fingerprint?: string }) {
  if (!rows.length)
    return <div className="p-4 font-mono text-[11px] text-muted-foreground">This run reported no same-rig MIXO baselines.</div>;
  const ref = rows.find((r) => r.id === 'mixo');
  const best = rows.reduce((a, b) => a.ate < b.ate ? a : b);
  return (
    <div>
      {ref && best.id !== ref.id && (
        <div className="px-3 py-2 border-b border-card-border text-[11px] text-[#e8b86a]" data-testid="text-mixo-drift-warning">
          The learned gate does not have the lowest position ATE on this run: {fmt(ref.ate, 4)} m versus {fmt(best.ate, 4)} m
          for {MIXO_BASELINE_META[best.id]?.label ?? best.name}. Lower held-out incremental loss does not guarantee lower accumulated trajectory drift.
        </div>
      )}
      <table className="w-full font-mono text-[11px] tnum" data-testid="table-mixo-baselines">
        <thead>
          <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
            <th className="text-left font-normal px-3 py-2">gate</th>
            <th className="text-right font-normal px-2">ATE m</th>
            <th className="text-right font-normal px-2">vs learned</th>
            <th className="text-right font-normal px-2">RPE m</th>
            <th className="text-right font-normal px-2">rot °</th>
            <th className="text-right font-normal px-2">avail %</th>
            <th className="text-right font-normal px-2">p95 ms</th>
            <th className="text-right font-normal px-3">recov s</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const meta = MIXO_BASELINE_META[r.id];
            const d = ref && r.id !== 'mixo' ? r.ate - ref.ate : null;
            return (
              <tr key={r.id} className="border-t border-card-border align-top" data-testid={`row-mixo-baseline-${r.id}`}>
                <td className="px-3 py-1.5">
                  <div className="flex items-center gap-2">
                    <Swatch color={meta?.color ?? '#8a97a6'} />
                    <span>{meta?.label ?? r.name}</span>
                    <span className="text-[9px] text-muted-foreground">{r.id}</span>
                  </div>
                  {meta && <div className="font-sans text-[11px] text-muted-foreground pl-5 leading-snug">{meta.note}</div>}
                </td>
                <td className="text-right px-2">{fmt(r.ate, 4)}</td>
                <td className="text-right px-2 text-muted-foreground">{d === null ? 'ref' : `${d >= 0 ? '+' : ''}${fmt(d, 4)}`}</td>
                <td className="text-right px-2">{fmt(r.rpe, 4)}</td>
                <td className="text-right px-2">{fmt(r.rotationError, 3)}</td>
                <td className="text-right px-2">{fmt(r.availability, 1)}</td>
                <td className="text-right px-2">{fmt(r.p95Latency, 2)}</td>
                <td className="text-right px-3">{r.recoveryDuration === null ? 'n/r' : fmt(r.recoveryDuration, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="px-3 py-2 border-t border-card-border text-[11px] text-muted-foreground leading-snug">
        All three rows consume the <span className="text-foreground">same frozen measurement set</span>
        {fingerprint ? <> (fingerprint <span className="font-mono text-foreground">{fingerprint.slice(0, 16)}</span>)</> : null}: identical four cameras, identical IMU,
        identical availability masks. Only the gate weights differ, so this is the fair same-rig comparison. One seed is one sample; a positive delta
        means the baseline had higher error on this run, not that the learned gate is generally better.
      </div>
    </div>
  );
}

/** Horizontal timeline of per-camera degradation windows (fractions of duration). */
export function CameraWindowStrip({
  windows,
  cursor,
  label,
}: {
  windows: Record<string, number[]>;
  cursor?: number;
  label?: string;
}) {
  const ids = Object.keys(windows);
  if (!ids.length) return <div className="px-3 py-2 font-mono text-[10px] text-muted-foreground">no windows reported</div>;
  return (
    <div className="px-3 py-2 space-y-1" data-testid="strip-camera-windows">
      {label && <div className="label-xs mb-1">{label}</div>}
      {ids.map((id) => {
        const w = windows[id] ?? [];
        const a = Math.max(0, Math.min(1, w[0] ?? 0));
        const b = Math.max(a, Math.min(1, w[1] ?? a));
        return (
          <div key={id} className="grid grid-cols-[52px_minmax(0,1fr)_86px] items-center gap-2 font-mono text-[10px]">
            <span style={{ color: cameraColor(id) }}>{CAMERA_META[id]?.label ?? id}</span>
            <div className="relative h-3 bg-secondary/60">
              <div className="absolute inset-y-0" style={{ left: `${a * 100}%`, width: `${(b - a) * 100}%`, background: cameraColor(id), opacity: 0.35 }} />
              <div className="absolute inset-y-0 border-x" style={{ left: `${a * 100}%`, width: `${(b - a) * 100}%`, borderColor: cameraColor(id) }} />
              {cursor !== undefined && <div className="absolute inset-y-[-2px] w-px bg-foreground" style={{ left: `${Math.max(0, Math.min(1, cursor)) * 100}%` }} />}
            </div>
            <span className="text-muted-foreground tnum text-right">{fmt(a, 2)}–{fmt(b, 2)}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Image-plane preview of one expert camera. Only actual observed points are drawn. */
export function ExpertPreview({ view, diag }: { view: ExpertView | undefined; diag: ExpertDiagnostic | undefined }) {
  const id = view?.id ?? diag?.id ?? '?';
  const color = cameraColor(id);
  const w = view?.width ?? 640;
  const h = view?.height ?? 480;
  const unavailable = diag ? !diag.available : false;
  const maxRes = Math.max(1, ...(view?.features ?? []).filter((p) => !p.outlier).map((p) => p.residual));
  return (
    <div className="relative bg-[#0b1016] border border-card-border" style={{ aspectRatio: `${w} / ${h}` }} data-testid={`preview-expert-${id}`}>
      <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <defs>
          <pattern id={`hatch-${id}`} width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="12" stroke="#d9737e" strokeOpacity="0.18" strokeWidth="4" />
          </pattern>
        </defs>
        <line x1={w / 2} y1={0} x2={w / 2} y2={h} stroke="#1d2631" strokeDasharray="4 6" />
        <line x1={0} y1={h / 2} x2={w} y2={h / 2} stroke="#1d2631" strokeDasharray="4 6" />
        {(view?.features ?? []).map((p) =>
          p.outlier ? (
            <g key={p.id} stroke="#d9737e" strokeWidth={w / 320}>
              <line x1={p.u - w / 110} y1={p.v - w / 110} x2={p.u + w / 110} y2={p.v + w / 110} />
              <line x1={p.u - w / 110} y1={p.v + w / 110} x2={p.u + w / 110} y2={p.v - w / 110} />
            </g>
          ) : (
            <circle key={p.id} cx={p.u} cy={p.v} r={w / 150} fill={color} fillOpacity={0.35 + 0.65 * (1 - Math.min(1, p.residual / maxRes))} />
          ),
        )}
        {unavailable && <rect x={0} y={0} width={w} height={h} fill={`url(#hatch-${id})`} />}
      </svg>
      <div className="absolute top-1 left-1.5 font-mono text-[10px] flex items-center gap-1.5">
        <span className="w-1.5 h-1.5" style={{ background: color }} />
        <span style={{ color }}>{view?.name ?? diag?.name ?? id}</span>
      </div>
      <div className="absolute top-1 right-1.5 font-mono text-[10px] tnum text-foreground/90 bg-background/70 px-1">
        w {diag ? fmt(diag.weight, 3) : '—'}
      </div>
      {view && (
        <div className="absolute bottom-1 left-1.5 font-mono text-[9px] text-muted-foreground bg-background/70 px-1" data-testid={`text-sample-${id}`}>
          {view.sampled ? `sampled ${view.features.length} of ${view.featureCount} observed` : `${view.features.length} of ${view.featureCount} observed (all shown)`}
        </div>
      )}
      {unavailable && (
        <div className="absolute inset-x-0 bottom-6 text-center font-mono text-[10px] text-[#d9737e]">
          UNAVAILABLE · {diag?.reason || 'no reason reported'}
        </div>
      )}
      {!view && !unavailable && <div className="absolute inset-0 grid place-items-center font-mono text-[10px] text-muted-foreground">no preview returned</div>}
    </div>
  );
}

export function ExpertDiagRows({ d }: { d: ExpertDiagnostic | undefined }) {
  if (!d) return <div className="px-2 py-1.5 font-mono text-[10px] text-muted-foreground">no diagnostic</div>;
  const inc = d.increment ?? [];
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 px-2 py-1.5 font-mono text-[10px] tnum">
      <dt className="text-muted-foreground">state</dt>
      <dd className={d.available ? 'text-primary' : 'text-[#d9737e]'}>{d.available ? 'available' : 'masked'}</dd>
      <dt className="text-muted-foreground">reason</dt>
      <dd className="truncate" title={d.reason}>{d.reason || '—'}</dd>
      <dt className="text-muted-foreground">features</dt>
      <dd>{d.featureCount}</dd>
      <dt className="text-muted-foreground">inliers</dt>
      <dd>{pct(d.inlierRatio)}</dd>
      <dt className="text-muted-foreground">reproj</dt>
      <dd>{fmt(d.reprojectionError, 3)} px</dd>
      <dt className="text-muted-foreground">quality</dt>
      <dd>{fmt(d.quality, 3)}</dd>
      <dt className="text-muted-foreground">latency</dt>
      <dd>{fmt(d.latency, 2)} ms</dd>
      <dt className="text-muted-foreground">Δt m</dt>
      <dd className="truncate">{inc.length >= 3 ? inc.slice(0, 3).map((v) => fmt(v, 4)).join(' ') : '—'}</dd>
      <dt className="text-muted-foreground">Δr rad</dt>
      <dd className="truncate">{inc.length >= 6 ? inc.slice(3, 6).map((v) => fmt(v, 4)).join(' ') : '—'}</dd>
    </dl>
  );
}
