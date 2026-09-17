import { createBlock } from '../../network/factory';
import type { BlockKind, InputBlock, LinearBlock, Network, OutputBlock } from '../../network/types';
import type { StorageKeys } from '../../persist/storage';

export const CNN_PALETTE: BlockKind[] = [
  'conv2d',
  'maxpool2d',
  'flatten',
  'linear',
  'relu',
  'sigmoid',
  'tanh',
  'softmax'
];

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
  const input: InputBlock = { ...(createBlock('input') as InputBlock), shape: [28, 28, 1] };
  const conv = createBlock('conv2d');
  const relu = createBlock('relu');
  const flatten = createBlock('flatten');
  const dense: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 10 };
  const softmax = createBlock('softmax');
  const output: OutputBlock = { ...(createBlock('output') as OutputBlock), units: 10 };

  return {
    version: 2,
    blocks: [input, conv, relu, flatten, dense, softmax, output],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
    positions: {}
  };
}

export function defaultSampleIndices(count: number = SAMPLE_GRID_SIZE): number[] {
  const size = Math.max(0, Math.min(count, SAMPLE_GRID_SIZE));
  return Array.from({ length: size }, (_, index) => index);
}
