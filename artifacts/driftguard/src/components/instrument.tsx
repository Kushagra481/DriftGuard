import type { ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useExperimentCtx, errMsg } from '@/lib/experiment-context';
import type { Experiment, ExperimentInput } from '@workspace/api-client-react';

export function Panel({
  title,
  meta,
  children,
  className,
  bodyClass,
  actions,
  testId,
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  actions?: ReactNode;
  testId?: string;
}) {
  return (
    <section data-testid={testId} className={cn('border border-card-border bg-card flex flex-col min-w-0', className)}>
      <header className="flex items-center gap-3 border-b border-card-border px-3 h-8 shrink-0">
        <h2 className="font-cond text-[11px] font-semibold uppercase tracking-[0.14em] text-foreground/90 truncate">{title}</h2>
        {meta && <div className="label-xs truncate normal-case tracking-normal">{meta}</div>}
        {actions && <div className="ml-auto flex items-center gap-1">{actions}</div>}
      </header>
      <div className={cn('flex-1 min-h-0', bodyClass)}>{children}</div>
    </section>
  );
}

export function Readout({
  label,
  value,
  unit,
  tone,
  testId,
  sub,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: string;
  testId?: string;
  sub?: ReactNode;
}) {
  return (
    <div className="min-w-0 px-3 py-2">
      <div className="label-xs truncate">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1 font-mono tnum">
        <span data-testid={testId} className="text-[15px] font-medium" style={tone ? { color: tone } : undefined}>
          {value}
        </span>
        {unit && <span className="text-[10px] text-muted-foreground">{unit}</span>}
      </div>
      {sub && <div className="text-[10px] text-muted-foreground font-mono mt-0.5 truncate">{sub}</div>}
    </div>
  );
}

export function Swatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return (
    <span
      className="inline-block w-3 h-[2px] shrink-0"
      style={dashed ? { backgroundImage: `linear-gradient(90deg, ${color} 50%, transparent 50%)`, backgroundSize: '4px 2px' } : { background: color }}
    />
  );
}

export function RunningPanel({ input, kind }: { input: ExperimentInput | null; kind: string }) {
  const p = input?.parameters ? Object.keys(input.parameters).length : 0;
  return (
    <div className="flex-1 grid place-items-center grid-bg p-6" data-testid="status-running">
      <div className="w-full max-w-md border border-card-border bg-card fade-in">
        <div className="h-[2px] bg-border overflow-hidden">
          <div className="h-full w-1/4 bg-primary scan" />
        </div>
        <div className="p-5">
          <div className="label-xs text-primary">
            {kind === 'initial' ? 'First session · running default experiment' : kind === 'derived' ? 'Deriving new run from current experiment' : 'Computing experiment'}
          </div>
          <div className="mt-2 font-cond text-xl font-semibold">Solving every estimator against ground truth</div>
          <p className="mt-2 text-[13px] text-muted-foreground leading-relaxed">
            The backend simulates the 30 Hz camera and 200 Hz IMU streams, injects the configured degradations across 30–70% of the
            sequence, and runs every method on identical inputs (MIXO additionally reads the side and rear cameras). This is synchronous and usually takes several seconds.
          </p>
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px]">
            <dt className="text-muted-foreground">preset</dt>
            <dd>{input?.preset ?? '—'}</dd>
            <dt className="text-muted-foreground">seed</dt>
            <dd>{input?.seed ?? '—'}</dd>
            <dt className="text-muted-foreground">duration</dt>
            <dd>{input?.duration !== undefined ? `${input.duration} s` : '—'}</dd>
            <dt className="text-muted-foreground">motion</dt>
            <dd>{input?.motion ?? '—'}</dd>
            <dt className="text-muted-foreground">param overrides</dt>
            <dd>{p}</dd>
            <dt className="text-muted-foreground">camera overrides</dt>
            <dd>{input?.cameraParameters ? Object.keys(input.cameraParameters).length : 0}</dd>
            <dt className="text-muted-foreground">rig degradation</dt>
            <dd>{input?.rigDegradation ?? 'staggered'}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

export function ErrorPanel({ title, error, onRetry }: { title: string; error: unknown; onRetry?: () => void }) {
  return (
    <div className="flex-1 grid place-items-center p-6" data-testid="status-error">
      <div className="max-w-md w-full border border-destructive/40 bg-card p-5">
        <div className="flex items-center gap-2 text-destructive">
          <AlertTriangle className="w-4 h-4" />
          <span className="font-cond font-semibold uppercase tracking-wider text-[12px]">{title}</span>
        </div>
        <pre className="mt-3 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap break-words">{errMsg(error)}</pre>
        {onRetry && (
          <button
            data-testid="button-retry"
            onClick={onRetry}
            className="mt-4 inline-flex items-center gap-2 h-8 px-3 border border-border text-[12px] hover:border-primary hover:text-primary transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Retry
          </button>
        )}
      </div>
    </div>
  );
}

export function SkeletonGrid() {
  return (
    <div className="flex-1 p-3 grid grid-cols-1 lg:grid-cols-3 gap-3" data-testid="status-loading">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="border border-card-border bg-card h-48 relative overflow-hidden">
          <div className="h-8 border-b border-card-border" />
          <div className="absolute inset-x-0 top-8 h-[2px] overflow-hidden">
            <div className="h-full w-1/4 bg-primary/40 scan" />
          </div>
        </div>
      ))}
    </div>
  );
}

/** Renders children only when an experiment is loaded; handles every other state explicitly. */
export function ExperimentGate({ children }: { children: (e: Experiment) => ReactNode }) {
  const c = useExperimentCtx();
  if (c.running) return <RunningPanel input={c.runInput} kind={c.running} />;
  if (c.historyError && !c.experiment) return <ErrorPanel title="Experiment history unavailable" error={c.historyError} onRetry={c.refetchHistory} />;
  if (c.experimentError) return <ErrorPanel title="Could not load experiment" error={c.experimentError} onRetry={c.refetchExperiment} />;
  if (c.runError && !c.experiment) return <ErrorPanel title="Experiment run failed" error={c.runError} onRetry={() => c.run({ preset: 'combined', seed: 42, duration: 8, motion: 'loop' })} />;
  if (!c.experiment) return <SkeletonGrid />;
  return <>{children(c.experiment)}</>;
}
