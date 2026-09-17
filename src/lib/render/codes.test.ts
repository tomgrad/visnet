import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { buildModel } from '../tf/buildModel';
import { codeScatter } from './codes';

let models: tf.Sequential[] = [];

const AUTOENCODER: Network = {
  blocks: [
    { id: 'in', kind: 'input', shape: [4, 4, 1] },
    { id: 'conv', kind: 'conv2d', filters: 1, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'flat', kind: 'flatten' },
    { id: 'code', kind: 'linear', units: 3 },
    { id: 'out', kind: 'output', shape: [3] }
  ],
  training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
  positions: {}
};

const FLAT_INPUT: Network = {
  blocks: [
    { id: 'in', kind: 'input', shape: [4] },
    { id: 'code', kind: 'linear', units: 2 },
    { id: 'out', kind: 'output', shape: [2] }
  ],
  training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 2 },
  positions: {}
};

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('codeScatter', () => {
  it('projects each sample onto the chosen dimension pair', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const sample = codeScatter(model, new Uint8Array(3 * 4 * 4).fill(120), 4, 4, 3, 2, 0);
    expect(sample.points).toHaveLength(6);
    expect(Number.isFinite(sample.bounds.minX)).toBe(true);
  });

  it('does not leak tensors', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const before = tf.memory().numTensors;
    codeScatter(model, new Uint8Array(4 * 4).fill(50), 4, 4, 1, 2, 0);
    expect(tf.memory().numTensors).toBe(before);
  });

  it('supports the input source on a flat-input model', () => {
    const model = buildModel(FLAT_INPUT);
    models.push(model);
    const sample = codeScatter(model, new Uint8Array(2 * 4).fill(120), 1, 4, 2, 'input', 0);
    const allFinite = [
      sample.bounds.minX,
      sample.bounds.maxX,
      sample.bounds.minY,
      sample.bounds.maxY
    ];
    expect(sample.points).toHaveLength(4);
    expect(allFinite.every(Number.isFinite)).toBe(true);
  });
});
