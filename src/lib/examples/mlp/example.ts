import { generate, type GeneratorName, type PointDataset } from '../../data/points';
import { cloneNetwork } from '../../network/factory';
import type { Network } from '../../network/types';
import type { StorageKeys } from '../../persist/storage';
import { MLP_NETWORK } from './networks/mlp';

export const DEFAULT_GENERATOR: GeneratorName = 'twoGaussians';
export const DEFAULT_POINT_COUNT = 200;
export const DEFAULT_SEED = 1;
export const CLASS_LABELS: [string, string] = ['blue', 'pink'];

export const MLP_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:network:v1',
  dataset: 'visnet:mlp:dataset:v1'
};

export const MLP_WEIGHTS_ID = 'main';

export function createMlpNetwork(): Network {
  return cloneNetwork(MLP_NETWORK);
}

export function defaultDataset(): PointDataset {
  return generate(DEFAULT_GENERATOR, DEFAULT_POINT_COUNT, DEFAULT_SEED);
}
