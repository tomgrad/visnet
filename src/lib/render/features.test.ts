import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createCnnNetwork } from '../examples/cnn/example';
import { buildModel } from '../tf/buildModel';
import { featureMaps } from './features';

let models: tf.Sequential[] = [];

function cnn(): tf.Sequential {
  const model = buildModel(createCnnNetwork());
  models.push(model);
  return model;
}

const FLAT = new Uint8Array(28 * 28).fill(128);
const VARIED = Uint8Array.from({ length: 28 * 28 }, (_, index) => (index * 7) % 256);

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('featureMaps', () => {
  it('returns one map per convolution channel', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 0);
    expect(maps).toHaveLength(8);
    expect(maps[0].width).toBe(28);
    expect(maps[0].height).toBe(28);
    expect(maps[0].values).toHaveLength(28 * 28);
  });

  it('returns one square per dense unit', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 3);
    expect(maps).toHaveLength(10);
    expect(maps[0].width).toBe(1);
    expect(maps[0].height).toBe(1);
  });

  it('normalises a map between its own minimum and maximum', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 0);
    const values = Array.from(maps[0].values);
    expect(Math.min(...values)).toBe(0);
    expect(Math.max(...values)).toBe(255);
  });

  it('shows a constant map as mid-grey', () => {
    const maps = featureMaps(cnn(), FLAT, 28, 28, 'input');
    expect(Array.from(maps[0].values)).toEqual(new Array(28 * 28).fill(128));
  });

  it('groups values by channel rather than contiguous memory', () => {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        filters: 2,
        kernelSize: 1,
        inputShape: [2, 2, 1],
        padding: 'same',
        activation: 'linear'
      })
    );
    model.layers[0].setWeights([tf.tensor4d([1, -1], [1, 1, 1, 2]), tf.tensor1d([0, 0])]);
    models.push(model);

    const pixels = Uint8Array.of(0, 85, 170, 255);
    const maps = featureMaps(model, pixels, 2, 2, 0);

    expect(Array.from(maps[0].values)).toEqual([0, 85, 170, 255]);
    expect(Array.from(maps[1].values)).toEqual([255, 170, 85, 0]);
  });

  it('does not leak tensors', () => {
    const model = cnn();
    const before = tf.memory().numTensors;
    featureMaps(model, VARIED, 28, 28, 1);
    expect(tf.memory().numTensors).toBe(before);
  });
});
