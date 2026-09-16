import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { generate } from './points';
import { toTensors } from './tensors';

let created: tf.Tensor[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  created.forEach((tensor) => tensor.dispose());
  created = [];
});

describe('toTensors', () => {
  it('produces float32 tensors with the documented shapes', () => {
    const { xs, ys } = toTensors(generate('xor', 12, 1));
    created.push(xs, ys);
    expect(xs.shape).toEqual([12, 2]);
    expect(ys.shape).toEqual([12, 2]);
    expect(xs.dtype).toBe('float32');
    expect(ys.dtype).toBe('float32');
  });

  it('one-hot encodes the labels', () => {
    const { ys } = toTensors(generate('xor', 6, 1));
    created.push(ys);
    const rows = ys.arraySync() as number[][];
    for (const row of rows) {
      expect(row.reduce((total, value) => total + value, 0)).toBe(1);
      expect([0, 1]).toContain(row[0]);
    }
  });

  it('handles an empty dataset', () => {
    const { xs, ys } = toTensors({ points: [], numClasses: 2 });
    created.push(xs, ys);
    expect(xs.shape).toEqual([0, 2]);
    expect(ys.shape).toEqual([0, 2]);
  });
});
