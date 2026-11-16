import { useMemo, useState } from 'react';
import type { ExperimentCamera, Feature } from '@workspace/api-client-react';
import { fmt } from '@/lib/format';

export type ColorMode = 'goi' | 'residual' | 'weight';

function ramp(t: number) {
  // slate -> violet -> cyan; perceptually ordered, not red/green.
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [70, 84, 102],
    [166, 140, 255],
    [85, 211, 224],
  ];
  const s = x < 0.5 ? 0 : 1;
  const k = x < 0.5 ? x / 0.5 : (x - 0.5) / 0.5;
  const a = stops[s];
  const b = stops[s + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * k)},${Math.round(a[1] + (b[1] - a[1]) * k)},${Math.round(a[2] + (b[2] - a[2]) * k)})`;
}

export function CameraView({
  camera,
  features,
  mode,
  residualScale = 4,
  selected,
  onSelect,
}: {
  camera: ExperimentCamera;
  features: Feature[];
  mode: ColorMode;
  residualScale?: number;
  selected?: number | null;
  onSelect?: (id: number | null) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const range = useMemo(() => {
    const vals = features.filter((f) => !f.missing).map((f) => (mode === 'goi' ? f.goi : mode === 'residual' ? f.residual : f.weight));
    const finite = vals.filter((v) => Number.isFinite(v));
    return { lo: finite.length ? Math.min(...finite) : 0, hi: finite.length ? Math.max(...finite) : 1 };
  }, [features, mode]);
  const norm = (v: number) => (range.hi - range.lo > 1e-12 ? (v - range.lo) / (range.hi - range.lo) : 0.5);
  const W = camera.width;
  const H = camera.height;
  const r = Math.max(W, H) / 160;
  const focus = features.find((f) => f.id === (hover ?? selected));

  return (
    <div className="relative w-full h-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full block" preserveAspectRatio="xMidYMid meet" onClick={() => onSelect?.(null)}>
        <rect x={0} y={0} width={W} height={H} fill="#090d12" stroke="#24303d" />
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={t} stroke="#141c25">
            <line x1={W * t} y1={0} x2={W * t} y2={H} />
            <line x1={0} y1={H * t} x2={W} y2={H * t} />
          </g>
        ))}
        <g stroke="#3a4655">
          <line x1={camera.cx - r * 2} y1={camera.cy} x2={camera.cx + r * 2} y2={camera.cy} />
          <line x1={camera.cx} y1={camera.cy - r * 2} x2={camera.cx} y2={camera.cy + r * 2} />
        </g>
        {features.map((f) => {
          const isSel = f.id === selected || f.id === hover;
          const handlers = {
            onMouseEnter: () => setHover(f.id),
            onMouseLeave: () => setHover(null),
            onClick: (e: React.MouseEvent) => {
              e.stopPropagation();
              onSelect?.(f.id === selected ? null : f.id);
            },
            style: { cursor: 'pointer' },
          };
          if (f.missing) {
            return (
              <g key={f.id} {...handlers} data-testid={`feature-missing-${f.id}`}>
                <circle cx={f.predictedU} cy={f.predictedV} r={r * 1.2} fill="none" stroke="#7b8896" strokeDasharray={`${r * 0.6} ${r * 0.5}`} strokeWidth={r * 0.25} opacity={isSel ? 1 : 0.6} />
              </g>
            );
          }
          const val = mode === 'goi' ? f.goi : mode === 'residual' ? f.residual : f.weight;
          const c = ramp(norm(val));
          const du = (f.u - f.predictedU) * residualScale;
          const dv = (f.v - f.predictedV) * residualScale;
          return (
            <g key={f.id} {...handlers} data-testid={`feature-${f.id}`}>
              <line x1={f.predictedU} y1={f.predictedV} x2={f.predictedU + du} y2={f.predictedV + dv} stroke={f.outlier ? '#d9737e' : c} strokeWidth={r * 0.3} opacity={0.8} />
              {f.outlier ? (
                <g stroke="#d9737e" strokeWidth={r * 0.35}>
                  <line x1={f.u - r} y1={f.v - r} x2={f.u + r} y2={f.v + r} />
                  <line x1={f.u - r} y1={f.v + r} x2={f.u + r} y2={f.v - r} />
                </g>
              ) : (
                <circle cx={f.u} cy={f.v} r={r * (0.7 + norm(val) * 0.9)} fill={c} fillOpacity={0.9} />
              )}
              {isSel && <rect x={f.u - r * 2.6} y={f.v - r * 2.6} width={r * 5.2} height={r * 5.2} fill="none" stroke="#e6dfcc" strokeWidth={r * 0.2} />}
            </g>
          );
        })}
      </svg>
      {focus && (
        <div className="absolute top-2 left-2 border border-popover-border bg-popover/95 px-2.5 py-2 font-mono text-[10px] leading-[1.6] pointer-events-none" data-testid="text-feature-detail">
          <div className="text-primary">feature #{focus.id}{focus.missing ? ' · MISSING' : focus.outlier ? ' · OUTLIER' : ''}</div>
          <div>obs   ({fmt(focus.u, 1)}, {fmt(focus.v, 1)}) px</div>
          <div>pred  ({fmt(focus.predictedU, 1)}, {fmt(focus.predictedV, 1)}) px</div>
          <div>resid {fmt(focus.residual, 3)} px</div>
          <div>weight {fmt(focus.weight, 3)}</div>
          <div>GOI   {fmt(focus.goi, 4)}</div>
          <div>L     [{focus.landmark.map((x) => fmt(x, 2)).join(', ')}] m</div>
        </div>
      )}
      <div className="absolute bottom-2 right-2 flex items-center gap-2 font-mono text-[9px] text-muted-foreground bg-background/70 px-2 py-1">
        <span>{mode}</span>
        <span>{fmt(range.lo, 3)}</span>
        <span className="w-16 h-1.5" style={{ background: `linear-gradient(90deg, ${ramp(0)}, ${ramp(0.5)}, ${ramp(1)})` }} />
        <span>{fmt(range.hi, 3)}</span>
      </div>
    </div>
  );
}

export function CameraLegend({ residualScale }: { residualScale: number }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-muted-foreground">
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-primary" />observed (color = mode)</span>
      <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-primary" />residual to prediction ×{residualScale}</span>
      <span className="flex items-center gap-1.5"><span className="text-[#d9737e]">×</span>outlier (rejected)</span>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full border border-dashed border-muted-foreground" />missing (predicted only)</span>
    </div>
  );
}
