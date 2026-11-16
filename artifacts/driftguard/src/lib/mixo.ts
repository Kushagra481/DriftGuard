import type { Experiment, MixoFrame, MixoModel } from '@workspace/api-client-react';

export function num(x: unknown): number | undefined {
  return typeof x === 'number' && Number.isFinite(x) ? x : undefined;
}
export function str(x: unknown): string | undefined {
  return typeof x === 'string' && x.length > 0 ? x : undefined;
}
export function strArr(x: unknown): string[] {
  return Array.isArray(x) ? x.filter((s): s is string => typeof s === 'string') : [];
}
export function numArr(x: unknown): number[] {
  return Array.isArray(x) ? x.filter((s): s is number => typeof s === 'number' && Number.isFinite(s)) : [];
}
export function obj(x: unknown): Record<string, unknown> | undefined {
  return x && typeof x === 'object' && !Array.isArray(x) ? (x as Record<string, unknown>) : undefined;
}

export interface ModelReport {
  status: string;
  trained: boolean;
  architecture: number[];
  trainingSamples?: number;
  validationSamples?: number;
  testSamples?: number;
  trainingLoss?: number;
  validationLoss?: number;
  testLoss?: number;
  uniformTestLoss?: number;
  matchTestLoss?: number;
  modelHash?: string;
  featureNames: string[];
  epochs?: number;
  trainingSeeds: number[];
  validationSeeds: number[];
  testSeeds: number[];
  namespaces?: Record<string, unknown>;
  citation?: string;
  source?: string;
  command?: string;
  warnings: string[];
}

export function readModel(m: MixoModel | undefined): ModelReport | null {
  if (!m) return null;
  const status = str(m.status) ?? 'unknown';
  return {
    status,
    trained: status === 'trained',
    architecture: numArr(m.architecture),
    trainingSamples: num(m.trainingSamples),
    validationSamples: num(m.validationSamples),
    testSamples: num(m.testSamples),
    trainingLoss: num(m.trainingLoss),
    validationLoss: num(m.validationLoss),
    testLoss: num(m.testLoss),
    uniformTestLoss: num(m.uniformTestLoss),
    matchTestLoss: num(m.matchTestLoss),
    modelHash: str(m.modelHash),
    featureNames: strArr(m.featureNames),
    epochs: num(m.epochs),
    trainingSeeds: numArr(m.trainingSeeds),
    validationSeeds: numArr(m.validationSeeds),
    testSeeds: numArr(m.testSeeds),
    namespaces: obj(m.namespaces),
    citation: str(m.citation),
    source: str(m.source),
    command: str(m.command) ?? str(m.trainCommand) ?? str(m.cli),
    warnings: strArr(m.warnings),
  };
}

export function mixoFrameAt(e: Experiment, frameIndex: number): MixoFrame | undefined {
  const fr = e.mixo?.frames;
  if (!fr || !fr.length) return undefined;
  const direct = fr[frameIndex];
  if (direct && direct.index === frameIndex) return direct;
  return fr.find((f) => f.index === frameIndex) ?? undefined;
}

/** Weight keys in a stable order: declared experts first, then any extra (e.g. imu). */
export function weightKeys(e: Experiment): string[] {
  const keys: string[] = [];
  for (const x of e.mixo?.experts ?? []) if (!keys.includes(x.id)) keys.push(x.id);
  for (const f of e.mixo?.frames ?? []) for (const k of Object.keys(f.weights ?? {})) if (!keys.includes(k)) keys.push(k);
  return keys;
}

export const CITATION =
  'Morra, L., et al. "MIXO: Mixture of Experts-based Visual Odometry for Multicamera Autonomous Systems." IEEE Transactions on Consumer Electronics, 2023. DOI 10.1109/TCE.2023.3238655';
