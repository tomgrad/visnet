import { describe, expect, it } from 'vitest';
import { findProblems } from '../../network/problems';
import {
  CNN_STORAGE_KEYS,
  CNN_WEIGHTS_ID,
  SAMPLE_GRID_COLUMNS,
  SAMPLE_GRID_SIZE,
  createCnnNetwork,
  defaultSampleIndices
} from './example';

describe('CNN_STORAGE_KEYS', () => {
  it('is namespaced away from the MLP', () => {
    expect(CNN_STORAGE_KEYS.network).toBe('visnet:cnn:network:v1');
    expect(CNN_STORAGE_KEYS.network).not.toBe('visnet:network:v1');
    expect(CNN_WEIGHTS_ID).toBe('cnn');
  });
});

describe('createCnnNetwork', () => {
  it('builds a convolutional network with no validation errors', () => {
    const network = createCnnNetwork();
    expect(network.positions).toEqual({});
    expect(network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'conv2d',
      'relu',
      'flatten',
      'linear',
      'softmax',
      'output'
    ]);
    expect(findProblems(network, { expectedClasses: 10 })).toEqual([]);
  });

  it('reads a 28 by 28 image and predicts ten classes', () => {
    const network = createCnnNetwork();
    expect(network.blocks[0]).toMatchObject({ kind: 'input', shape: [28, 28, 1] });
    expect(network.blocks[4]).toMatchObject({ kind: 'linear', units: 10 });
    expect(network.blocks[6]).toMatchObject({ kind: 'output', shape: [10] });
  });
});

describe('defaultSampleIndices', () => {
  it('is the first forty images', () => {
    const indices = defaultSampleIndices();
    expect(indices).toHaveLength(SAMPLE_GRID_SIZE);
    expect(indices[0]).toBe(0);
    expect(indices[SAMPLE_GRID_SIZE - 1]).toBe(SAMPLE_GRID_SIZE - 1);
    expect(SAMPLE_GRID_SIZE % SAMPLE_GRID_COLUMNS).toBe(0);
  });

  it('never asks for more images than the dataset holds', () => {
    expect(defaultSampleIndices(5)).toEqual([0, 1, 2, 3, 4]);
  });

  it('still caps at the grid size for a large dataset', () => {
    expect(defaultSampleIndices(1000)).toHaveLength(SAMPLE_GRID_SIZE);
  });

  it('returns nothing for an empty dataset', () => {
    expect(defaultSampleIndices(0)).toEqual([]);
  });
});
