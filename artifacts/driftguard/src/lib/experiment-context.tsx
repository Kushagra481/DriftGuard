import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  useListExperiments,
  useStartExperiment,
  useUpdateExperiment,
  useGetExperiment,
  getGetExperimentQueryKey,
  getListExperimentsQueryKey,
  type Experiment,
  type ExperimentInput,
  type ExperimentSummary,
} from '@workspace/api-client-react';

const STORAGE_KEY = 'driftguard.selectedExperiment';
export const DEFAULT_INPUT: ExperimentInput = { preset: 'combined', seed: 42, duration: 8, motion: 'loop' };

type RunKind = 'initial' | 'new' | 'derived';

interface Ctx {
  selectedId: string | null;
  select: (id: string) => void;
  experiment: Experiment | undefined;
  experimentLoading: boolean;
  experimentError: unknown;
  refetchExperiment: () => void;
  history: ExperimentSummary[] | undefined;
  historyLoading: boolean;
  historyError: unknown;
  refetchHistory: () => void;
  run: (input: ExperimentInput) => void;
  derive: (input: ExperimentInput) => void;
  running: RunKind | null;
  runInput: ExperimentInput | null;
  runError: string | null;
  clearRunError: () => void;
  frameIndex: number;
  setFrameIndex: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
}

const ExperimentCtx = createContext<Ctx | null>(null);

function errMsg(e: unknown) {
  if (!e) return 'Unknown error';
  if (e instanceof Error) return e.message;
  if (typeof e === 'object' && e && 'message' in e) return String((e as { message: unknown }).message);
  return String(e);
}

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY));
  const [running, setRunning] = useState<RunKind | null>(null);
  const [runInput, setRunInput] = useState<ExperimentInput | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  // Replay opens at the final frame so every computed trajectory is visible in full.
  const [frameIndex, setFrameIndexRaw] = useState(Number.MAX_SAFE_INTEGER);
  const [playing, setPlaying] = useState(false);
  const bootstrapped = useRef(false);

  const list = useListExperiments();
  const exp = useGetExperiment(selectedId ?? '', {
    query: { enabled: !!selectedId, queryKey: getGetExperimentQueryKey(selectedId ?? ''), staleTime: Infinity, retry: 1 },
  });
  const start = useStartExperiment();
  const update = useUpdateExperiment();

  const select = useCallback((id: string) => {
    setSelectedId(id);
    localStorage.setItem(STORAGE_KEY, id);
    setFrameIndexRaw(Number.MAX_SAFE_INTEGER);
    setPlaying(false);
  }, []);

  const onDone = useCallback(
    (e: Experiment) => {
      qc.setQueryData(getGetExperimentQueryKey(e.id), e);
      qc.invalidateQueries({ queryKey: getListExperimentsQueryKey() });
      select(e.id);
      setRunning(null);
    },
    [qc, select],
  );

  const startRef = useRef(start.mutate);
  startRef.current = start.mutate;
  const updateRef = useRef(update.mutate);
  updateRef.current = update.mutate;

  const runWith = useCallback(
    (input: ExperimentInput, kind: RunKind) => {
      setRunning(kind);
      setRunInput(input);
      setRunError(null);
      startRef.current(
        { data: input },
        { onSuccess: onDone, onError: (e) => { setRunning(null); setRunError(errMsg(e)); } },
      );
    },
    [onDone],
  );

  const run = useCallback((input: ExperimentInput) => runWith(input, 'new'), [runWith]);

  const derive = useCallback(
    (input: ExperimentInput) => {
      if (!selectedId) return runWith(input, 'new');
      setRunning('derived');
      setRunInput(input);
      setRunError(null);
      updateRef.current(
        { id: selectedId, data: input },
        { onSuccess: onDone, onError: (e) => { setRunning(null); setRunError(errMsg(e)); } },
      );
    },
    [selectedId, onDone, runWith],
  );

  // Bootstrap: reopen persisted / most recent, or auto-run default.
  useEffect(() => {
    if (bootstrapped.current || !list.data) return;
    bootstrapped.current = true;
    const items = list.data;
    if (items.length === 0) {
      localStorage.removeItem(STORAGE_KEY);
      setSelectedId(null);
      runWith(DEFAULT_INPUT, 'initial');
      return;
    }
    if (selectedId && items.some((x) => x.id === selectedId)) return;
    const recent = [...items].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
    select(recent.id);
  }, [list.data, selectedId, runWith, select]);

  const nFrames = exp.data?.frames.length ?? 0;
  const setFrameIndex = useCallback(
    (i: number) => setFrameIndexRaw(Math.max(0, Math.min(Math.max(0, nFrames - 1), Math.round(i)))),
    [nFrames],
  );

  // Replay ticker: advances through precomputed frames at their recorded timestamps.
  useEffect(() => {
    if (!playing || !exp.data) return;
    const frames = exp.data.frames;
    if (frames.length < 2) return;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let idx = Math.min(frameIndex, frames.length - 1);
    const tick = (now: number) => {
      acc += (now - last) / 1000;
      last = now;
      while (idx < frames.length - 1 && acc >= frames[idx + 1].time - frames[idx].time) {
        acc -= frames[idx + 1].time - frames[idx].time;
        idx++;
      }
      setFrameIndexRaw(idx);
      if (idx >= frames.length - 1) {
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, exp.data]);

  const value: Ctx = {
    selectedId,
    select,
    experiment: exp.data,
    experimentLoading: !!selectedId && exp.isLoading,
    experimentError: exp.error,
    refetchExperiment: () => void exp.refetch(),
    history: list.data,
    historyLoading: list.isLoading,
    historyError: list.error,
    refetchHistory: () => void list.refetch(),
    run,
    derive,
    running,
    runInput,
    runError,
    clearRunError: () => setRunError(null),
    frameIndex: Math.min(frameIndex, Math.max(0, nFrames - 1)),
    setFrameIndex,
    playing,
    setPlaying: (p: boolean) => {
      if (p && exp.data && frameIndex >= exp.data.frames.length - 1) setFrameIndexRaw(0);
      setPlaying(p);
    },
  };
  return <ExperimentCtx.Provider value={value}>{children}</ExperimentCtx.Provider>;
}

export function useExperimentCtx() {
  const c = useContext(ExperimentCtx);
  if (!c) throw new Error('ExperimentProvider missing');
  return c;
}

export { errMsg };
