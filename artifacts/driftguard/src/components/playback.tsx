import { useMemo } from 'react';
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack } from 'lucide-react';
import type { Experiment } from '@workspace/api-client-react';
import { useExperimentCtx } from '@/lib/experiment-context';
import { fmt } from '@/lib/format';

export function PlaybackBar({ experiment }: { experiment: Experiment }) {
  const { frameIndex, setFrameIndex, playing, setPlaying } = useExperimentCtx();
  const frames = experiment.frames;
  const f = frames[frameIndex];
  const n = frames.length;

  // Strip: degraded window and keyframes along the timeline.
  const strip = useMemo(() => {
    if (!n) return { deg: [] as number[], kf: [] as number[] };
    return {
      deg: frames.map((x, i) => (x.degraded ? i : -1)).filter((i) => i >= 0),
      kf: frames.map((x, i) => (x.keyframe ? i : -1)).filter((i) => i >= 0),
    };
  }, [frames, n]);

  if (!n) return null;
  const btn = 'h-7 w-7 grid place-items-center border border-border hover:border-primary/60 hover:text-primary transition-colors';
  return (
    <div className="border-t border-border bg-card/60 px-3 py-2 flex items-center gap-3 shrink-0" data-testid="playback-bar">
      <div className="flex items-center gap-1">
        <button className={btn} onClick={() => setFrameIndex(0)} data-testid="button-rewind" aria-label="First frame">
          <SkipBack className="w-3.5 h-3.5" />
        </button>
        <button className={btn} onClick={() => setFrameIndex(frameIndex - 1)} data-testid="button-prev-frame" aria-label="Previous frame">
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        <button
          className={`${btn} w-9 ${playing ? 'text-primary border-primary/60' : ''}`}
          onClick={() => setPlaying(!playing)}
          data-testid="button-play-pause"
          aria-label={playing ? 'Pause replay' : 'Play replay'}
        >
          {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        </button>
        <button className={btn} onClick={() => setFrameIndex(frameIndex + 1)} data-testid="button-next-frame" aria-label="Next frame">
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="hidden sm:block leading-tight">
        <div className="font-mono text-[9px] tracking-widest text-accent">REPLAY</div>
        <div className="font-mono text-[9px] text-muted-foreground whitespace-nowrap">
          {!playing && frameIndex === n - 1 ? 'final frame · full computed paths' : 'of computed sequence'}
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <div className="relative h-2">
          {strip.deg.length > 0 && (
            <div
              className="absolute top-0 h-1 bg-accent/40"
              style={{ left: `${(strip.deg[0] / (n - 1 || 1)) * 100}%`, width: `${((strip.deg[strip.deg.length - 1] - strip.deg[0]) / (n - 1 || 1)) * 100}%` }}
              title="Degradation window"
            />
          )}
          {strip.kf.map((i) => (
            <div key={i} className="absolute top-0 w-px h-2 bg-primary/50" style={{ left: `${(i / (n - 1 || 1)) * 100}%` }} />
          ))}
        </div>
        <input
          type="range"
          className="scrub"
          min={0}
          max={n - 1}
          step={1}
          value={frameIndex}
          onChange={(e) => {
            setPlaying(false);
            setFrameIndex(Number(e.target.value));
          }}
          data-testid="input-scrub"
          aria-label="Scrub frames"
        />
      </div>
      <div className="font-mono text-[11px] tnum text-right whitespace-nowrap" data-testid="text-frame-position">
        <span className="text-foreground">{String(frameIndex).padStart(String(n - 1).length, '0')}</span>
        <span className="text-muted-foreground">/{n - 1}</span>
        <span className="ml-3 text-foreground">{fmt(f?.time, 3)}</span>
        <span className="text-muted-foreground"> s</span>
        {f?.degraded && <span className="ml-3 text-accent">DEGRADED</span>}
      </div>
    </div>
  );
}
