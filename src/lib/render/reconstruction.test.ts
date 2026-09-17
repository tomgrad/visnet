import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { buildModel } from '../tf/buildModel';
import { reconstruct } from './reconstruction';

let models: tf.Sequential[] = [];

const AUTOENCODER: Network = {
  version: 2,
  blocks: [
    { id: 'in', kind: 'input', shape: [4, 4, 1] },
    { id: 'conv', kind: 'conv2d', filters: 1, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'out', kind: 'output', shape: [4, 4, 1] }
  ],
  training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
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

describe('reconstruct', () => {
  it('returns one grayscale image per input', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const images = reconstruct(model, new Uint8Array(4 * 4).fill(200), 4, 4, 1);
    expect(images).toHaveLength(4 * 4);
    expect(Math.max(...images)).toBeLessThanOrEqual(255);
  });

  it('does not leak tensors', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const before = tf.memory().numTensors;
    reconstruct(model, new Uint8Array(4 * 4).fill(10), 4, 4, 1);
    expect(tf.memory().numTensors).toBe(before);
  });
});
