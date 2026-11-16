import { useState, type ReactNode } from 'react';
import { Link, useLocation } from 'wouter';
import { Activity, BookOpen, Box, Boxes, Download, FlaskConical, GitCompare, LineChart, Loader2, Radar, X } from 'lucide-react';
import { exportExperiment, useHealthCheck, getHealthCheckQueryKey } from '@workspace/api-client-react';
import { useExperimentCtx, errMsg } from '@/lib/experiment-context';
import { fmt, shortId } from '@/lib/format';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';

const NAV = [
  { href: '/', label: 'Live Experiment', code: '01', icon: Box },
  { href: '/sensors', label: 'Sensor Laboratory', code: '02', icon: FlaskConical },
  { href: '/analysis', label: 'Trajectory Analysis', code: '03', icon: LineChart },
  { href: '/goi', label: 'GOI Explorer', code: '04', icon: Radar },
  { href: '/benchmarks', label: 'Benchmarking', code: '05', icon: GitCompare },
  { href: '/experts', label: 'MIXO Experts', code: '06', icon: Boxes },
  { href: '/methodology', label: 'Methodology', code: '07', icon: BookOpen },
];

function Mark() {
  return (
    <svg viewBox="0 0 24 24" className="w-6 h-6" aria-hidden>
      <circle cx="12" cy="12" r="9.5" fill="none" stroke="#55d3e0" strokeWidth="1" />
      <ellipse cx="12" cy="12" rx="9.5" ry="4" fill="none" stroke="#a68cff" strokeWidth="1" transform="rotate(-28 12 12)" />
      <circle cx="12" cy="12" r="1.6" fill="#e6dfcc" />
    </svg>
  );
}

function HealthDot() {
  const h = useHealthCheck({ query: { refetchInterval: 30000, queryKey: getHealthCheckQueryKey() } });
  const ok = h.data?.status === 'ok' || h.data?.status === 'healthy';
  const label = h.isLoading ? 'probing' : h.error ? 'unreachable' : (h.data?.status ?? 'unknown');
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground" data-testid="status-health">
      <span className={`w-1.5 h-1.5 ${h.error ? 'bg-destructive' : ok ? 'bg-primary' : 'bg-muted-foreground'}`} />
      api {label}
    </div>
  );
}

function ExportButtons() {
  const { selectedId, experiment } = useExperimentCtx();
  const { toast } = useToast();
  const [busy, setBusy] = useState<'json' | 'csv' | null>(null);
  const go = async (format: 'json' | 'csv') => {
    if (!selectedId) return;
    setBusy(format);
    try {
      const data = await exportExperiment(selectedId, format);
      const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
      const blob = new Blob([text], { type: format === 'json' ? 'application/json' : 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `driftguard_${experiment?.config.preset ?? 'run'}_seed${experiment?.config.seed ?? ''}_${shortId(selectedId)}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      toast({ title: `Export ${format.toUpperCase()} failed`, description: errMsg(e), variant: 'destructive' });
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="flex items-center">
      {(['json', 'csv'] as const).map((f) => (
        <button
          key={f}
          data-testid={`button-export-${f}`}
          disabled={!selectedId || !!busy}
          onClick={() => go(f)}
          className="h-7 px-2.5 border border-border -ml-px first:ml-0 font-mono text-[10px] uppercase tracking-wider inline-flex items-center gap-1.5 hover:text-primary hover:border-primary/60 disabled:opacity-40 transition-colors"
        >
          {busy === f ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
          {f}
        </button>
      ))}
    </div>
  );
}

function ExperimentPicker() {
  const { history, selectedId, select, running } = useExperimentCtx();
  const sorted = [...(history ?? [])].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return (
    <Select value={selectedId ?? ''} onValueChange={select} disabled={!!running || !sorted.length}>
      <SelectTrigger data-testid="select-experiment" className="h-7 w-[300px] rounded-none bg-transparent font-mono text-[11px] border-border">
        <SelectValue placeholder={history ? 'no experiments' : 'loading history'} />
      </SelectTrigger>
      <SelectContent className="rounded-none font-mono text-[11px]">
        {sorted.map((s) => (
          <SelectItem key={s.id} value={s.id} className="font-mono text-[11px]" data-testid={`option-experiment-${s.id}`}>
            {shortId(s.id)} · {s.preset} · seed {s.seed} · {s.duration}s · ATE {fmt(s.ate, 3)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [loc] = useLocation();
  const c = useExperimentCtx();
  const current = NAV.find((n) => n.href === loc);
  return (
    <div className="min-h-[100dvh] md:h-[100dvh] md:overflow-hidden flex bg-background text-foreground">
      <aside className="hidden md:flex w-[212px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar sticky top-0 h-[100dvh]">
        <div className="h-12 flex items-center gap-2.5 px-4 border-b border-sidebar-border">
          <Mark />
          <div className="leading-tight">
            <div className="font-cond font-semibold text-[14px] tracking-wide">DriftGuard XR</div>
            <div className="font-mono text-[9px] text-muted-foreground tracking-wider">VIO FAILURE WORKBENCH</div>
          </div>
        </div>
        <nav className="flex-1 py-3">
          {NAV.map((n) => {
            const active = n.href === loc;
            const Icon = n.icon;
            return (
              <Link
                key={n.href}
                href={n.href}
                data-testid={`link-nav-${n.code}`}
                className={`relative flex items-center gap-3 h-9 px-4 text-[13px] transition-colors ${
                  active ? 'text-foreground bg-sidebar-accent' : 'text-sidebar-foreground hover:text-foreground'
                }`}
              >
                {active && <span className="absolute left-0 top-1.5 bottom-1.5 w-[2px] bg-primary" />}
                <span className="font-mono text-[9px] text-muted-foreground w-4">{n.code}</span>
                <Icon className={`w-3.5 h-3.5 ${active ? 'text-primary' : ''}`} />
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="p-4 border-t border-sidebar-border space-y-2">
          <HealthDot />
          <div className="font-mono text-[9px] leading-relaxed text-muted-foreground">
            Synthetic known-landmark benchmark. Results describe this simulator, not deployed hardware.
          </div>
        </div>
      </aside>
      <div className="flex-1 min-w-0 min-h-0 flex flex-col">
        <header className="h-12 shrink-0 border-b border-border flex items-center gap-3 px-3 md:px-4 bg-background/95 sticky top-0 z-20">
          <div className="md:hidden flex items-center gap-2">
            <Mark />
          </div>
          <div className="min-w-0">
            <div className="label-xs">{current?.code ?? '--'} / workspace</div>
            <div className="font-cond font-semibold text-[14px] leading-tight truncate">{current?.label ?? 'Not found'}</div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {c.running && (
              <span className="hidden sm:inline-flex items-center gap-1.5 font-mono text-[10px] text-primary" data-testid="status-run-pending">
                <Activity className="w-3 h-3" /> computing
              </span>
            )}
            <div className="hidden lg:block">
              <ExperimentPicker />
            </div>
            <ExportButtons />
          </div>
        </header>
        <div className="md:hidden flex overflow-x-auto border-b border-border">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} className={`px-3 h-9 flex items-center text-[12px] whitespace-nowrap ${n.href === loc ? 'text-primary border-b border-primary' : 'text-muted-foreground'}`}>
              {n.label}
            </Link>
          ))}
        </div>
        {c.runError && c.experiment && (
          <div className="flex items-center gap-3 px-4 py-2 border-b border-destructive/40 bg-destructive/10 text-[12px]" data-testid="status-run-error">
            <span className="font-mono text-destructive">RUN FAILED</span>
            <span className="text-foreground/80 truncate">{c.runError}</span>
            <span className="text-muted-foreground">Previous experiment kept.</span>
            <button onClick={c.clearRunError} className="ml-auto text-muted-foreground hover:text-foreground" data-testid="button-dismiss-error">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
        <main className="flex-1 min-h-0 flex flex-col">{children}</main>
      </div>
    </div>
  );
}
