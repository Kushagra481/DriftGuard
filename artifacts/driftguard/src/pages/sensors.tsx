import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation } from 'wouter';
import { Camera, GitBranch, Play, RotateCcw, Shuffle, Undo2 } from 'lucide-react';
import { useGetPresets, type ExperimentInput, type ExperimentInputMotion, type ExperimentInputRigDegradation, type Preset } from '@workspace/api-client-react';
import { useExperimentCtx, DEFAULT_INPUT } from '@/lib/experiment-context';
import { PARAM_SPEC, CAMERA_IDS, CAMERA_META, RIG_WINDOWS, cameraColor, fmt, hasMixo, type CameraId } from '@/lib/format';
import { CameraWindowStrip } from '@/components/mixo-kit';
import { Panel, ErrorPanel, RunningPanel } from '@/components/instrument';
import { Slider } from '@/components/ui/slider';
import { Skeleton } from '@/components/ui/skeleton';

const MOTIONS: { id: ExperimentInputMotion; label: string; note: string }[] = [
  { id: 'loop', label: 'Loop', note: 'Full 6-DoF loop, rich parallax' },
  { id: 'pure_rotation', label: 'Pure rotation', note: 'Translation unobservable from vision' },
  { id: 'low_parallax', label: 'Low parallax', note: 'Small baseline, weak depth' },
];

const MAX_SEED = 2147483647;
const CAM_SPEC = PARAM_SPEC.filter((s) => s.group === 'camera' && s.key !== 'parallax_scale');
const CAM_KEYS = new Set(CAM_SPEC.map((s) => s.key));

function cleanCamParams(src: Record<string, Record<string, number>> | undefined) {
  const out: Record<string, Record<string, number>> = {};
  for (const [cam, vals] of Object.entries(src ?? {})) {
    const o: Record<string, number> = {};
    for (const [k, v] of Object.entries(vals ?? {})) if (CAM_KEYS.has(k) && Number.isFinite(v)) o[k] = v;
    if (Object.keys(o).length) out[cam] = o;
  }
  return out;
}

function CamParamRow({
  spec,
  inherited,
  override,
  onChange,
  onInherit,
  cam,
}: {
  spec: (typeof PARAM_SPEC)[number];
  inherited: number;
  override: number | undefined;
  onChange: (v: number) => void;
  onInherit: () => void;
  cam: string;
}) {
  const value = override ?? inherited;
  const on = override !== undefined;
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_88px] gap-x-3 px-3 py-2 border-t border-card-border first:border-t-0">
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="text-[12.5px]">{spec.label}</span>
          {on ? (
            <>
              <span className="font-mono text-[9px]" style={{ color: cameraColor(cam) }}>override · global {fmt(inherited, 3)}</span>
              <button onClick={onInherit} className="ml-auto font-mono text-[9px] text-muted-foreground hover:text-foreground inline-flex items-center gap-1" data-testid={`button-inherit-${cam}-${spec.key}`}>
                <Undo2 className="w-2.5 h-2.5" /> inherit
              </button>
            </>
          ) : (
            <span className="font-mono text-[9px] text-muted-foreground">inherits global</span>
          )}
        </div>
        <Slider
          className={`mt-2 ${on ? '' : 'opacity-50'}`}
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={[Math.min(spec.max, Math.max(spec.min, value))]}
          onValueChange={(v) => onChange(v[0])}
          data-testid={`slider-cam-${cam}-${spec.key}`}
        />
      </div>
      <div>
        <input
          type="number"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={Number.isFinite(value) ? value : 0}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) onChange(Math.min(spec.max, Math.max(spec.min, v)));
          }}
          className={`w-full h-7 bg-background border px-2 font-mono text-[11px] tnum text-right focus:outline-none focus:border-primary ${on ? 'border-input' : 'border-input/50 text-muted-foreground'}`}
          data-testid={`input-cam-${cam}-${spec.key}`}
        />
        <div className="mt-1 font-mono text-[9px] text-muted-foreground text-right">{spec.unit}</div>
      </div>
    </div>
  );
}

function ParamRow({
  spec,
  value,
  base,
  onChange,
}: {
  spec: (typeof PARAM_SPEC)[number];
  value: number;
  base: number | undefined;
  onChange: (v: number) => void;
}) {
  const changed = base !== undefined && Math.abs(value - base) > 1e-12;
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_88px] gap-x-3 px-3 py-2.5 border-t border-card-border first:border-t-0">
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="text-[12.5px]">{spec.label}</span>
          <span className="font-mono text-[9px] text-muted-foreground">{spec.key}</span>
          {changed && <span className="font-mono text-[9px] text-accent">modified · preset {fmt(base, 3)}</span>}
        </div>
        <Slider
          className="mt-2"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={[Math.min(spec.max, Math.max(spec.min, value))]}
          onValueChange={(v) => onChange(v[0])}
          data-testid={`slider-${spec.key}`}
        />
        <div className="mt-1 text-[10.5px] text-muted-foreground leading-snug">{spec.hint}</div>
      </div>
      <div>
        <input
          type="number"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={Number.isFinite(value) ? value : 0}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) onChange(Math.min(spec.max, Math.max(spec.min, v)));
          }}
          className="w-full h-7 bg-background border border-input px-2 font-mono text-[11px] tnum text-right focus:outline-none focus:border-primary"
          data-testid={`input-${spec.key}`}
        />
        <div className="mt-1 font-mono text-[9px] text-muted-foreground text-right">{spec.unit}</div>
      </div>
    </div>
  );
}

export default function SensorsPage() {
  const c = useExperimentCtx();
  const [, navigate] = useLocation();
  const presets = useGetPresets();
  const [presetId, setPresetId] = useState<string>(DEFAULT_INPUT.preset!);
  const [seed, setSeed] = useState<number>(DEFAULT_INPUT.seed!);
  const [duration, setDuration] = useState<number>(DEFAULT_INPUT.duration!);
  const [motion, setMotion] = useState<ExperimentInputMotion>(DEFAULT_INPUT.motion!);
  const [params, setParams] = useState<Record<string, number>>({});
  const [camParams, setCamParams] = useState<Record<string, Record<string, number>>>({});
  const [rig, setRig] = useState<ExperimentInputRigDegradation>('staggered');
  const [activeCam, setActiveCam] = useState<CameraId>('front');
  const initFor = useRef<string | null>(null);

  const preset: Preset | undefined = presets.data?.find((p) => p.id === presetId);

  // Initialise once per experiment from its stored config.
  useEffect(() => {
    const e = c.experiment;
    if (!e || !presets.data || initFor.current === e.id) return;
    initFor.current = e.id;
    const cfg = e.config;
    const pid = cfg.preset ?? presets.data[0]?.id ?? 'combined';
    setPresetId(pid);
    setSeed(cfg.seed ?? 42);
    setDuration(cfg.duration ?? 8);
    setMotion(cfg.motion ?? 'loop');
    const p = presets.data.find((x) => x.id === pid);
    setParams({ ...(p?.parameters ?? {}), ...(cfg.parameters ?? {}) });
    setCamParams(cleanCamParams(cfg.cameraParameters));
    setRig(cfg.rigDegradation ?? 'staggered');
  }, [c.experiment, presets.data]);

  const choosePreset = (p: Preset) => {
    setPresetId(p.id);
    setParams({ ...p.parameters });
    // Camera overrides are absolute values chosen against the previous preset; drop them so every camera re-inherits the new preset.
    setCamParams({});
  };

  const setCam = (cam: string, key: string, v: number | undefined) =>
    setCamParams((prev) => {
      const cur = { ...(prev[cam] ?? {}) };
      if (v === undefined) delete cur[key];
      else cur[key] = v;
      const next = { ...prev };
      if (Object.keys(cur).length) next[cam] = cur;
      else delete next[cam];
      return next;
    });
  const nCamOver = Object.values(camParams).reduce((a, o) => a + Object.keys(o).length, 0);

  const overrides = useMemo(() => {
    const o: Record<string, number> = {};
    for (const s of PARAM_SPEC) {
      const v = params[s.key];
      if (v === undefined) continue;
      if (preset?.parameters[s.key] === undefined || Math.abs(preset.parameters[s.key] - v) > 1e-12) o[s.key] = v;
    }
    return o;
  }, [params, preset]);

  const seedValid = Number.isInteger(seed) && seed >= 0 && seed <= MAX_SEED;
  const durValid = duration >= 3 && duration <= 30;
  const valid = seedValid && durValid;

  const input = (): ExperimentInput => ({
    preset: presetId,
    seed,
    duration,
    motion,
    parameters: { ...params },
    rigDegradation: rig,
    ...(nCamOver > 0 ? { cameraParameters: cleanCamParams(camParams) } : {}),
  });

  const submit = (kind: 'new' | 'derive', destination = '/') => {
    if (!valid || !preset || (c.experiment && initFor.current !== c.experiment.id)) return;
    const i = input();
    if (kind === 'new') c.run(i);
    else c.derive(i);
    navigate(destination);
  };

  if (c.running) return <RunningPanel input={c.runInput} kind={c.running} />;
  if (presets.error) return <ErrorPanel title="Presets unavailable" error={presets.error} onRetry={() => presets.refetch()} />;

  const nOver = Object.keys(overrides).length;

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)] gap-px bg-border md:overflow-y-auto lg:overflow-hidden">
      <div className="bg-background flex flex-col gap-px lg:overflow-y-auto">
        <Panel title="Scenario presets" meta={presets.data ? `${presets.data.length} defined` : ''} className="border-0">
          {presets.isLoading ? (
            <div className="p-3 space-y-2">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-10 rounded-none" />)}</div>
          ) : (
            <div>
              {presets.data?.map((p) => (
                <button
                  key={p.id}
                  onClick={() => choosePreset(p)}
                  data-testid={`button-preset-${p.id}`}
                  className={`w-full text-left px-3 py-2 border-t border-card-border first:border-t-0 relative transition-colors ${p.id === presetId ? 'bg-secondary' : 'hover:bg-secondary/50'}`}
                >
                  {p.id === presetId && <span className="absolute left-0 inset-y-0 w-[2px] bg-primary" />}
                  <div className="flex items-baseline gap-2">
                    <span className="text-[12.5px] font-medium">{p.name}</span>
                    <span className="font-mono text-[9px] text-muted-foreground">{p.id}</span>
                  </div>
                  <div className="text-[11px] text-muted-foreground leading-snug mt-0.5">{p.description}</div>
                </button>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <div className="bg-background flex flex-col min-h-0 lg:overflow-y-auto">
        <div className="px-3 py-3 border-b border-border bg-card flex flex-wrap items-center gap-3" data-testid="status-sensor-mixo">
          <div className="flex-1 min-w-[200px]">
            <div className="text-[12px] font-medium">MIXO experts · four cameras + IMU</div>
            <p className="text-[11px] text-muted-foreground mt-1">
              {c.experiment && !hasMixo(c.experiment)
                ? 'This saved run predates MIXO and has no expert results. Generate a new run with the settings below; the original stays in history.'
                : 'Every new run computes MIXO expert weights and camera diagnostics using the settings below.'}
            </p>
          </div>
          {c.experiment && hasMixo(c.experiment) && (
            <Link href="/experts" className="text-[12px] text-primary underline underline-offset-4" data-testid="link-sensor-mixo-results">
              View current expert results
            </Link>
          )}
          <button
            onClick={() => submit('new', '/experts')}
            disabled={!valid || !preset || !!(c.experiment && initFor.current !== c.experiment.id)}
            className="h-8 px-3 bg-primary text-primary-foreground text-[12px] font-medium inline-flex items-center gap-2 disabled:opacity-40"
            data-testid="button-run-mixo"
          >
            <Play className="w-3.5 h-3.5" /> Run with MIXO experts
          </button>
        </div>
        <Panel title="Run configuration" meta="front/IMU degradation over 30–70% of duration · rig cameras per window below" className="border-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-3">
            <div>
              <div className="label-xs mb-1">Seed · integer 0..2147483647</div>
              <div className="flex">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Math.trunc(Number(e.target.value)))}
                  className={`flex-1 min-w-0 h-8 bg-background border px-2 font-mono text-[12px] tnum focus:outline-none ${seedValid ? 'border-input focus:border-primary' : 'border-destructive'}`}
                  data-testid="input-seed"
                />
                <button
                  onClick={() => setSeed(Math.floor(Math.random() * MAX_SEED))}
                  className="h-8 w-8 -ml-px border border-input grid place-items-center hover:text-primary"
                  title="Random seed"
                  data-testid="button-random-seed"
                >
                  <Shuffle className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <div>
              <div className="label-xs mb-1">Duration · {fmt(duration, 1)} s (3..30)</div>
              <Slider min={3} max={30} step={0.5} value={[duration]} onValueChange={(v) => setDuration(v[0])} className="mt-3" data-testid="slider-duration" />
              <div className="mt-1.5 font-mono text-[9px] text-muted-foreground">
                ≈ {Math.round(duration * 30)} camera frames · {Math.round(duration * 200)} IMU samples
              </div>
            </div>
            <div>
              <div className="label-xs mb-1">Motion profile</div>
              <div className="grid grid-cols-3">
                {MOTIONS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setMotion(m.id)}
                    title={m.note}
                    data-testid={`button-motion-${m.id}`}
                    className={`h-8 -ml-px first:ml-0 border text-[11px] ${motion === m.id ? 'border-primary/60 text-primary z-10 relative' : 'border-input text-muted-foreground hover:text-foreground'}`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              <div className="mt-1.5 text-[10.5px] text-muted-foreground">{MOTIONS.find((m) => m.id === motion)?.note}</div>
            </div>
          </div>
        </Panel>

        <div className="flex-1 grid grid-cols-1 xl:grid-cols-2 gap-px bg-border">
          {(['camera', 'imu'] as const).map((g) => (
            <Panel key={g} title={g === 'camera' ? 'Global camera channel · 30 Hz' : 'Shared inertial channel · 200 Hz'} meta={g === 'camera' ? 'front primary camera; inherited by every rig camera' : 'shared by all methods'} className="border-0">
              {PARAM_SPEC.filter((s) => s.group === g).map((s) => (
                <ParamRow
                  key={s.key}
                  spec={s}
                  value={params[s.key] ?? 0}
                  base={preset?.parameters[s.key]}
                  onChange={(v) => setParams((p) => ({ ...p, [s.key]: v }))}
                />
              ))}
            </Panel>
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] gap-px bg-border border-t border-border">
          <Panel
            title="Per-camera visual overrides"
            meta={`${nCamOver} override${nCamOver === 1 ? '' : 's'} · motion parallax stays global`}
            className="border-0"
            actions={
              <button
                onClick={() => setCamParams({})}
                disabled={nCamOver === 0}
                className="font-mono text-[10px] px-2 h-5 border border-border text-muted-foreground hover:text-foreground disabled:opacity-40 inline-flex items-center gap-1"
                data-testid="button-reset-camera-overrides"
              >
                <RotateCcw className="w-3 h-3" /> reset camera overrides
              </button>
            }
          >
            <div className="grid grid-cols-4 border-b border-card-border" role="tablist">
              {CAMERA_IDS.map((cam) => {
                const n = Object.keys(camParams[cam] ?? {}).length;
                const act = cam === activeCam;
                return (
                  <button
                    key={cam}
                    role="tab"
                    aria-selected={act}
                    onClick={() => setActiveCam(cam)}
                    data-testid={`button-camera-${cam}`}
                    className={`relative h-9 text-[12px] inline-flex items-center justify-center gap-1.5 border-l border-card-border first:border-l-0 ${act ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
                  >
                    {act && <span className="absolute inset-x-0 bottom-0 h-[2px]" style={{ background: cameraColor(cam) }} />}
                    <Camera className="w-3 h-3" style={{ color: cameraColor(cam) }} />
                    {CAMERA_META[cam].label}
                    {n > 0 && <span className="font-mono text-[9px] px-1 border" style={{ borderColor: cameraColor(cam), color: cameraColor(cam) }}>{n}</span>}
                  </button>
                );
              })}
            </div>
            <div className="px-3 py-1.5 text-[10.5px] text-muted-foreground leading-snug border-b border-card-border">
              {activeCam === 'front'
                ? 'Front overrides also change the primary camera read by the five single-camera methods.'
                : `The ${CAMERA_META[activeCam].label.toLowerCase()} camera is read only by the MIXO-inspired estimator; overrides here never touch the single-camera methods.`}{' '}
              Unset fields inherit the global resolved preset value.
            </div>
            {CAM_SPEC.map((spec) => (
              <CamParamRow
                key={`${activeCam}-${spec.key}`}
                cam={activeCam}
                spec={spec}
                inherited={params[spec.key] ?? 0}
                override={camParams[activeCam]?.[spec.key]}
                onChange={(v) => setCam(activeCam, spec.key, v)}
                onInherit={() => setCam(activeCam, spec.key, undefined)}
              />
            ))}
          </Panel>

          <Panel title="Rig degradation timing" meta="fractions of run duration" className="border-0">
            <div className="p-3 grid grid-cols-2">
              {(['staggered', 'common'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setRig(m)}
                  data-testid={`button-rig-${m}`}
                  className={`h-8 -ml-px first:ml-0 border text-[11px] ${rig === m ? 'border-primary/60 text-primary z-10 relative' : 'border-input text-muted-foreground hover:text-foreground'}`}
                >
                  {m === 'staggered' ? 'Staggered windows' : 'Common window'}
                </button>
              ))}
            </div>
            <CameraWindowStrip windows={RIG_WINDOWS[rig]} label="configured · nominal" />
            {c.experiment?.mixo?.cameraWindows ? (
              <CameraWindowStrip windows={c.experiment.mixo.cameraWindows} label={`actual · run ${c.experiment.id.slice(0, 8)} (${c.experiment.config.rigDegradation ?? 'staggered'})`} />
            ) : (
              <div className="px-3 py-2 font-mono text-[10px] text-muted-foreground">current run reports no cameraWindows (legacy run)</div>
            )}
            <div className="mx-3 mb-3 mt-1 border-l-2 border-[#e8b86a]/70 pl-2 text-[10.5px] text-muted-foreground leading-snug">
              Simplification: each camera degrades over one contiguous window with hard edges. The gate is trained offline from the documented command line;
              there is no training control here by design.
            </div>
          </Panel>
        </div>

        <div className="sticky bottom-0 border-t border-border bg-card px-3 py-2 flex flex-wrap items-center gap-3">
          <div className="font-mono text-[11px] text-muted-foreground" data-testid="text-override-count">
            {presetId} · {nOver} parameter{nOver === 1 ? '' : 's'} differ from preset · {nCamOver} camera override{nCamOver === 1 ? '' : 's'} · {rig}
            {!valid && <span className="ml-3 text-destructive">invalid {!seedValid ? 'seed' : 'duration'}</span>}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => preset && setParams({ ...preset.parameters })}
              disabled={!preset}
              className="h-8 px-3 border border-border text-[12px] inline-flex items-center gap-2 hover:text-foreground text-muted-foreground disabled:opacity-40"
              data-testid="button-reset-params"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset to preset
            </button>
            <button
              onClick={() => submit('derive')}
              disabled={!valid || !preset || !c.selectedId}
              title="PATCH: creates a new run derived from the current one; the original run remains in history"
              className="h-8 px-3 border border-accent/60 text-accent text-[12px] inline-flex items-center gap-2 hover:bg-accent/10 disabled:opacity-40"
              data-testid="button-derive-run"
            >
              <GitBranch className="w-3.5 h-3.5" /> Derive from current run
            </button>
            <button
              onClick={() => submit('new')}
              disabled={!valid || !preset}
              className="h-8 px-4 bg-primary text-primary-foreground text-[12px] font-medium inline-flex items-center gap-2 hover:brightness-110 disabled:opacity-40"
              data-testid="button-run-experiment"
            >
              <Play className="w-3.5 h-3.5" /> Run new experiment
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
