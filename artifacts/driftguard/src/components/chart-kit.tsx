import type { Frame } from '@workspace/api-client-react';
import { ReferenceArea, ReferenceLine } from 'recharts';
import { fmt } from '@/lib/format';

export const AXIS = { stroke: '#3a4655', fontSize: 10, fontFamily: 'IBM Plex Mono', tick: { fill: '#7b8896' } };
export const GRID = { stroke: '#1d2631', strokeDasharray: '2 4' };

interface TPayload {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
}

export function InstrumentTooltip({
  active,
  payload,
  label,
  unit,
  labelFmt,
}: {
  active?: boolean;
  payload?: TPayload[];
  label?: number | string;
  unit?: string;
  labelFmt?: (l: number | string) => string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="border border-popover-border bg-popover px-2.5 py-2 font-mono text-[11px] min-w-[150px]">
      <div className="text-muted-foreground mb-1">{label !== undefined ? (labelFmt ? labelFmt(label) : `t = ${fmt(Number(label), 3)} s`) : ''}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 shrink-0" style={{ background: p.color }} />
          <span className="text-foreground/80">{p.name}</span>
          <span className="ml-auto tnum">
            {typeof p.value === 'number' ? fmt(p.value, 4) : p.value}
            {unit ? ` ${unit}` : ''}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Shaded degradation window + current replay cursor. Returned as an array so Recharts 2 picks the elements up directly. */
export function timeOverlays(frames: Frame[], frameIndex: number, yAxisId?: string) {
  const ax = yAxisId ? { yAxisId } : {};
  const deg = frames.filter((f) => f.degraded);
  const t = frames[frameIndex]?.time;
  const out = [];
  if (deg.length > 0)
    out.push(
      <ReferenceArea key="deg" {...ax} x1={deg[0].time} x2={deg[deg.length - 1].time} fill="#a68cff" fillOpacity={0.07} stroke="#a68cff" strokeOpacity={0.25} strokeDasharray="2 3" ifOverflow="extendDomain" />,
    );
  if (t !== undefined) out.push(<ReferenceLine key="cur" {...ax} x={t} stroke="#55d3e0" strokeWidth={1} />);
  return out;
}
