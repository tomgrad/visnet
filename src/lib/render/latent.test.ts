import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { LATENT_GRID_SIZE, gridInputs, projectLatent } from './latent';

let models: tf.Sequential[] = [];

function model(): tf.Sequential {
  const built = buildModel(createEmptyNetwork());
  models.push(built);
  return built;
}

const POINTS = new Float32Array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6]);

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((entry) => entry.dispose());
  models = [];
});

describe('gridInputs', () => {
  it('builds a size-by-size grid of coordinate pairs', () => {
    const cells = gridInputs(4);
    expect(cells).toHaveLength(4 * 4 * 2);
    expect(cells[0]).toBeCloseTo(-0.75, 5);
    expect(cells[1]).toBeCloseTo(0.75, 5);
  });
});

describe('projectLatent', () => {
  it('projects the grid and points onto the chosen dimensions', () => {
    const sample = projectLatent(model(), gridInputs(LATENT_GRID_SIZE), POINTS, 0, 0);
    expect(sample.grid).toHaveLength(LATENT_GRID_SIZE * LATENT_GRID_SIZE * 2);
    expect(sample.gridClasses).toHaveLength(LATENT_GRID_SIZE * LATENT_GRID_SIZE);
    expect(sample.points).toHaveLength(POINTS.length);
    expect(sample.bounds.maxX).toBeGreaterThanOrEqual(sample.bounds.minX);
  });

  it('returns the raw input coordinates for the input source', () => {
    const cells = gridInputs(4);
    const sample = projectLatent(model(), cells, POINTS, 'input', 0);
    expect(Array.from(sample.grid.slice(0, 2))).toEqual([cells[0], cells[1]]);
  });

  it('colours each grid cell by the model argmax', () => {
    const built = model();
    const cells = gridInputs(4);
    const sample = projectLatent(built, cells, POINTS, 0, 0);
    const expected = tf.tidy(() => {
      const input = tf.tensor2d(cells, [cells.length / 2, 2]);
      const logits = built.predict(input) as tf.Tensor;
      return Array.from(tf.argMax(logits, 1).dataSync());
    });
    expect(Array.from(sample.gridClasses)).toEqual(expected);
  });

  it('leaves the training model usable and does not leak tensors', () => {
    const built = model();
    const before = tf.memory().numTensors;
    projectLatent(built, gridInputs(8), POINTS, 1, 0);
    expect(tf.memory().numTensors).toBe(before);
    const prediction = tf.tidy(() => built.predict(tf.tensor2d([[0.1, 0.2]])) as tf.Tensor);
    expect(prediction.shape).toEqual([1, 2]);
  });
});
