import { useMemo, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Experiment } from '@workspace/api-client-react';
import { ExperimentGate, Panel, Swatch } from '@/components/instrument';
import { PlaybackBar } from '@/components/playback';
import { AXIS, GRID, InstrumentTooltip, timeOverlays } from '@/components/chart-kit';
import { useExperimentCtx } from '@/lib/experiment-context';
import { METHOD_IDS, METHOD_META, fmt } from '@/lib/format';

function quatAngleDeg(a: number[], b: number[]) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return (2 * Math.acos(Math.min(1, d)) * 180) / Math.PI;
}

function AnalysisInner({ experiment }: { experiment: Experiment }) {
  const { frameIndex, setFrameIndex } = useExperimentCtx();
  const [on, setOn] = useState<Record<string, boolean>>(() => Object.fromEntries(METHOD_IDS.map((m) => [m, true])));
  const [logY, setLogY] = useState(false);
  const frames = experiment.frames;

  const data = useMemo(
    () =>
      frames.map((f) => {
        const row: Record<string, number | null> = {
          t: f.time,
          posU: f.positionUncertainty,
          oriU: f.orientationUncertainty,
          vq: f.visualQuality,
          iq: f.imuQuality,
          inl: f.inlierRatio,
        };
        for (const m of METHOD_IDS) {
          const e = f.errors[m];
          row[`e_${m}`] = e === undefined || !Number.isFinite(e) ? null : logY ? Math.max(e, 1e-5) : e;
          const p = f.poses[m];
          row[`r_${m}`] = p ? quatAngleDeg(p.quaternion, f.truth.quaternion) : null;
        }
        return row;
      }),
    [frames, logY],
  );

  const cur = frames[frameIndex];
  const onClick = (s: { activeTooltipIndex?: number } | null) => {
    if (s && typeof s.activeTooltipIndex === 'number') setFrameIndex(s.activeTooltipIndex);
  };

  const legend = (
    <div className="flex flex-wrap gap-1">
      {METHOD_IDS.map((m) => (
        <button
          key={m}
          onClick={() => setOn({ ...on, [m]: !on[m] })}
          data-testid={`toggle-series-${m}`}
          className={`h-5 px-1.5 border font-mono text-[10px] inline-flex items-center gap-1.5 ${on[m] ? 'border-border' : 'border-transparent opacity-40'}`}
        >
          <Swatch color={METHOD_META[m].color} /> {METHOD_META[m].short}
        </button>
      ))}
    </div>
  );

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex-1 overflow-auto grid grid-cols-1 xl:grid-cols-2 gap-px bg-border auto-rows-min">
        <Panel
          title="Position error vs ground truth"
          meta="‖p̂ − p‖ per frame, m"
          className="border-0 xl:col-span-2"
          actions={
            <>
              {legend}
              <button
                onClick={() => setLogY(!logY)}
                className={`ml-2 h-5 px-1.5 border font-mono text-[10px] ${logY ? 'border-primary/60 text-primary' : 'border-border text-muted-foreground'}`}
                data-testid="toggle-log-scale"
              >
                log
              </button>
            </>
          }
        >
          <div className="h-[300px] p-2">
            <ResponsiveContainer>
              <LineChart data={data} onClick={onClick} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis {...AXIS} scale={logY ? 'log' : 'auto'} domain={logY ? ['auto', 'auto'] : [0, 'auto']} allowDataOverflow tickFormatter={(v) => fmt(v, 2)} width={52} />
                <Tooltip content={<InstrumentTooltip unit="m" />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex)}
                {METHOD_IDS.filter((m) => on[m]).map((m) => (
                  <Line key={m} dataKey={`e_${m}`} name={METHOD_META[m].short} stroke={METHOD_META[m].color} strokeWidth={m === 'driftguard' ? 1.8 : 1.1} dot={false} isAnimationActive={false} connectNulls={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="px-3 pb-2 font-mono text-[9px] text-muted-foreground">shaded = degradation window (frames flagged degraded) · cyan rule = replay cursor · click chart to seek</div>
        </Panel>

        <Panel title="Rotation error" meta="geodesic angle q̂ vs q, deg · computed client-side from returned quaternions" className="border-0">
          <div className="h-[240px] p-2">
            <ResponsiveContainer>
              <LineChart data={data} onClick={onClick} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis {...AXIS} tickFormatter={(v) => fmt(v, 1)} width={44} />
                <Tooltip content={<InstrumentTooltip unit="deg" />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex)}
                {METHOD_IDS.filter((m) => on[m]).map((m) => (
                  <Line key={m} dataKey={`r_${m}`} name={METHOD_META[m].short} stroke={METHOD_META[m].color} strokeWidth={1.1} dot={false} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Reported uncertainty" meta="DriftGuard √trace of covariance block · position m (left) / orientation deg (right) · not per-axis 1σ" className="border-0">
          <div className="h-[240px] p-2">
            <ResponsiveContainer>
              <ComposedChart data={data} onClick={onClick} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis yAxisId="p" {...AXIS} tickFormatter={(v) => fmt(v, 3)} width={52} />
                <YAxis yAxisId="o" orientation="right" {...AXIS} tickFormatter={(v) => fmt(v, 1)} width={40} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex, 'p')}
                <Area yAxisId="p" dataKey="posU" name="√tr Σ pos (m)" stroke="#a68cff" fill="#a68cff" fillOpacity={0.15} isAnimationActive={false} />
                <Line yAxisId="o" dataKey="oriU" name="√tr Σ ori (deg)" stroke="#55d3e0" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
                <Line yAxisId="p" dataKey="e_driftguard" name="DGX err (m)" stroke="#e6dfcc" strokeOpacity={0.6} dot={false} strokeWidth={1} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="px-3 pb-2 font-mono text-[9px] text-muted-foreground">compare √tr Σ pos to actual DGX error; error persistently above it suggests over-confidence (uncalibrated scalar, not a confidence interval)</div>
        </Panel>

        <Panel title="Sensor quality signals" meta="0..1" className="border-0 xl:col-span-2">
          <div className="h-[200px] p-2">
            <ResponsiveContainer>
              <LineChart data={data} onClick={onClick} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={(v) => `${fmt(v, 1)}s`} />
                <YAxis {...AXIS} domain={[0, 1]} width={36} />
                <Tooltip content={<InstrumentTooltip />} isAnimationActive={false} cursor={{ stroke: '#3a4655' }} />
                {timeOverlays(frames, frameIndex)}
                <Line dataKey="vq" name="visual quality" stroke="#55d3e0" dot={false} isAnimationActive={false} />
                <Line dataKey="iq" name="IMU quality" stroke="#d4a24a" dot={false} isAnimationActive={false} />
                <Line dataKey="inl" name="inlier ratio" stroke="#a68cff" dot={false} strokeDasharray="3 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="At replay cursor" meta={cur ? `frame ${frameIndex} · t ${fmt(cur.time, 3)} s` : ''} className="border-0 xl:col-span-2" bodyClass="overflow-x-auto">
          {cur && (
            <table className="w-full font-mono text-[11px] tnum">
              <thead>
                <tr className="text-muted-foreground text-[9px] uppercase tracking-wider">
                  <th className="text-left font-normal px-3 py-1.5">method</th>
                  <th className="text-right font-normal px-2">pos err m</th>
                  <th className="text-right font-normal px-2">rot err °</th>
                  <th className="text-right font-normal px-2">x̂</th>
                  <th className="text-right font-normal px-2">ŷ</th>
                  <th className="text-right font-normal px-3">ẑ</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-card-border text-[#e6dfcc]">
                  <td className="px-3 py-1">ground truth</td>
                  <td className="text-right px-2">—</td>
                  <td className="text-right px-2">—</td>
                  {cur.truth.position.map((v, i) => (
                    <td key={i} className={`text-right ${i === 2 ? 'px-3' : 'px-2'}`}>{fmt(v, 3)}</td>
                  ))}
                </tr>
                {METHOD_IDS.map((m) => {
                  const p = cur.poses[m];
                  return (
                    <tr key={m} className="border-t border-card-border" data-testid={`row-cursor-${m}`}>
                      <td className="px-3 py-1 flex items-center gap-2"><Swatch color={METHOD_META[m].color} />{METHOD_META[m].label}</td>
                      <td className="text-right px-2">{fmt(cur.errors[m], 4)}</td>
                      <td className="text-right px-2">{p ? fmt(quatAngleDeg(p.quaternion, cur.truth.quaternion), 2) : '—'}</td>
                      {(p?.position ?? [NaN, NaN, NaN]).map((v, i) => (
                        <td key={i} className={`text-right ${i === 2 ? 'px-3' : 'px-2'}`}>{fmt(v, 3)}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
      <PlaybackBar experiment={experiment} />
    </div>
  );
}

export default function AnalysisPage() {
  return <ExperimentGate>{(e) => <AnalysisInner experiment={e} />}</ExperimentGate>;
}
