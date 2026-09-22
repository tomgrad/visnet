import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { EmojiDataset } from './emoji';
import type { ImageDataset } from './images';
import { generate } from './points';
import {
  colourImagesToReconstruction,
  imagesToReconstruction,
  imagesToTensors,
  toTensors
} from './tensors';

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

function images(count = 3, rows = 2, cols = 2, numClasses = 3): ImageDataset {
  const size = rows * cols;
  const pixels = new Uint8Array(count * size);
  for (let i = 0; i < pixels.length; i++) pixels[i] = i === 0 ? 255 : i % 256;
  return {
    count,
    rows,
    cols,
    numClasses,
    pixels,
    labels: Uint8Array.from({ length: count }, (_, i) => i % numClasses)
  };
}

describe('imagesToTensors', () => {
  it('produces a 4d image batch and a one-hot label batch', () => {
    const { xs, ys } = imagesToTensors(images());
    created.push(xs, ys);

    expect(xs.shape).toEqual([3, 2, 2, 1]);
    expect(ys.shape).toEqual([3, 3]);
    expect(xs.dtype).toBe('float32');
    expect(ys.dtype).toBe('float32');
  });

  it('scales pixels into 0..1', () => {
    const { xs } = imagesToTensors(images());
    created.push(xs);

    const values = Array.from(xs.dataSync());
    expect(values[0]).toBe(1);
    expect(values[1]).toBeCloseTo(1 / 255, 6);
  });

  it('maps a zero pixel to zero', () => {
    const dataset = images();
    dataset.pixels[0] = 0;
    const { xs } = imagesToTensors(dataset);
    created.push(xs);

    expect(Array.from(xs.dataSync())[0]).toBe(0);
  });

  it('one-hot encodes the labels', () => {
    const { ys } = imagesToTensors(images());
    created.push(ys);

    const rows = ys.arraySync() as number[][];
    expect(rows).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]);
  });

  it('selects a subset in the given order', () => {
    const dataset = images();
    const { xs, ys } = imagesToTensors(dataset, [2, 0]);
    created.push(xs, ys);

    expect(xs.shape).toEqual([2, 2, 2, 1]);
    expect(ys.arraySync()).toEqual([
      [0, 0, 1],
      [1, 0, 0]
    ]);
    expect(Array.from(xs.dataSync()).slice(0, 4)).toEqual(
      Array.from(dataset.pixels.slice(8, 12), (value) => Math.fround(value / 255))
    );
  });
});

describe('imagesToReconstruction', () => {
  const dataset = {
    count: 2,
    rows: 2,
    cols: 2,
    numClasses: 10,
    pixels: Uint8Array.of(0, 255, 128, 64, 10, 20, 30, 40),
    labels: Uint8Array.of(3, 7)
  };

  it('builds matching input and target tensors', () => {
    const { xs, ys } = imagesToReconstruction(dataset);
    expect(xs.shape).toEqual([2, 2, 2, 1]);
    expect(ys.shape).toEqual([2, 2, 2, 1]);
    expect(Array.from(xs.dataSync())).toEqual(Array.from(ys.dataSync()));
    xs.dispose();
    ys.dispose();
  });

  it('normalises pixels to the zero-to-one range', () => {
    const { xs, ys } = imagesToReconstruction(dataset);
    expect(Array.from(xs.dataSync())[1]).toBeCloseTo(1, 5);
    xs.dispose();
    ys.dispose();
  });
});

function colourDataset(): EmojiDataset {
  const pixels = new Uint8Array([0, 51, 102, 153, 204, 255, 255, 0, 0, 0, 255, 0]);
  return { count: 2, rows: 1, cols: 2, channels: 3, pixels };
}

describe('colourImagesToReconstruction', () => {
  it('builds three-channel [n, rows, cols, 3] tensors normalised to [0, 1]', () => {
    const { xs, ys } = colourImagesToReconstruction(colourDataset());
    created.push(xs, ys);

    expect(xs.shape).toEqual([2, 1, 2, 3]);
    expect(ys.shape).toEqual([2, 1, 2, 3]);
    const expected = [0, 0.2, 0.4, 0.6, 0.8, 1, 1, 0, 0, 0, 1, 0].map((value) =>
      Math.fround(value)
    );
    expect(Array.from(xs.dataSync())).toEqual(expected);
    expect(Array.from(ys.dataSync())).toEqual(expected);
  });

  it('selects a subset by index', () => {
    const { xs, ys } = colourImagesToReconstruction(colourDataset(), [1]);
    created.push(xs, ys);

    expect(xs.shape).toEqual([1, 1, 2, 3]);
    expect(Array.from(xs.dataSync())).toEqual(
      [1, 0, 0, 0, 1, 0].map((value) => Math.fround(value))
    );
  });
});
