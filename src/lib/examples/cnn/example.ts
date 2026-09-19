import { cloneNetwork } from '../../network/factory';
import type { Network } from '../../network/types';
import type { StorageKeys } from '../../persist/storage';
import { CNN_NETWORK } from './networks/cnn';

export const CNN_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:cnn:network:v1',
  dataset: 'visnet:cnn:dataset:v1'
};

export const CNN_WEIGHTS_ID = 'cnn';

export const SAMPLE_GRID_SIZE = 40;
export const SAMPLE_GRID_COLUMNS = 8;
export const SAMPLE_GRID_CELL_WIDTH = 36;
export const SAMPLE_GRID_CELL_HEIGHT = 48;
export const SAMPLE_GRID_GAP = 4;

export function createCnnNetwork(): Network {
  return cloneNetwork(CNN_NETWORK);
}

export function defaultSampleIndices(count: number = SAMPLE_GRID_SIZE): number[] {
  const size = Math.max(0, Math.min(count, SAMPLE_GRID_SIZE));
  return Array.from({ length: size }, (_, index) => index);
}
