import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { BACKGROUND_RGB, CLASS_COLOURS } from './palette';
import { GRID_SIZE, cellCentre, classesToRgba, clientToDomain, sampleGrid } from './boundary';

const RECT = { left: 10, top: 20, width: 200, height: 100 };

let models: tf.LayersModel[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('palette', () => {
  it('matches the design tokens', () => {
    expect(CLASS_COLOURS[0].hex).toBe('#38bdf8');
    expect(CLASS_COLOURS[1].hex).toBe('#fb7185');
  });

  it('provides rgb triples', () => {
    expect(CLASS_COLOURS[0].rgb).toEqual([56, 189, 248]);
    expect(CLASS_COLOURS[1].rgb).toEqual([251, 113, 133]);
    expect(BACKGROUND_RGB).toHaveLength(3);
  });
});

describe('clientToDomain', () => {
  it('maps the rect corners to the domain corners', () => {
    expect(clientToDomain(10, 120, RECT)).toEqual({ x: -1, y: -1 });
    expect(clientToDomain(210, 20, RECT)).toEqual({ x: 1, y: 1 });
  });

  it('maps the centre to the origin', () => {
    const centre = clientToDomain(110, 70, RECT);
    expect(centre.x).toBeCloseTo(0, 10);
    expect(centre.y).toBeCloseTo(0, 10);
  });

  it('flips the y axis so up is positive', () => {
    expect(clientToDomain(110, 20, RECT).y).toBeCloseTo(1, 10);
    expect(clientToDomain(110, 120, RECT).y).toBeCloseTo(-1, 10);
  });
});

describe('cellCentre', () => {
  it('centres the first cell in the top-left', () => {
    const centre = cellCentre(0, 0, 2);
    expect(centre.x).toBeCloseTo(-0.5, 10);
    expect(centre.y).toBeCloseTo(0.5, 10);
  });

  it('centres the last cell in the bottom-right', () => {
    const centre = cellCentre(1, 1, 2);
    expect(centre.x).toBeCloseTo(0.5, 10);
    expect(centre.y).toBeCloseTo(-0.5, 10);
  });

  it('stays inside the domain for the default grid', () => {
    for (const gx of [0, 1, GRID_SIZE - 1]) {
      for (const gy of [0, 1, GRID_SIZE - 1]) {
        const centre = cellCentre(gx, gy);
        expect(Math.abs(centre.x)).toBeLessThan(1);
        expect(Math.abs(centre.y)).toBeLessThan(1);
      }
    }
  });
});

describe('classesToRgba', () => {
  const colours: Array<[number, number, number]> = [
    [10, 20, 30],
    [40, 50, 60]
  ];

  it('produces four bytes per cell', () => {
    expect(classesToRgba(new Int32Array(4), colours, 2)).toHaveLength(2 * 2 * 4);
  });

  it('writes opaque colours in row-major order', () => {
    const rgba = classesToRgba(Int32Array.from([0, 1, 1, 0]), colours, 2);
    expect(Array.from(rgba.slice(0, 4))).toEqual([10, 20, 30, 255]);
    expect(Array.from(rgba.slice(4, 8))).toEqual([40, 50, 60, 255]);
    expect(Array.from(rgba.slice(8, 12))).toEqual([40, 50, 60, 255]);
    expect(Array.from(rgba.slice(12, 16))).toEqual([10, 20, 30, 255]);
  });

  it('falls back to black for an unknown class', () => {
    const rgba = classesToRgba(Int32Array.from([9]), colours, 1);
    expect(Array.from(rgba)).toEqual([0, 0, 0, 255]);
  });
});

describe('sampleGrid', () => {
  it('classifies every cell', () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [2], useBias: false }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    const classes = sampleGrid(model, 8);
    expect(classes).toHaveLength(64);
    expect(Array.from(classes).every((value) => value === 0 || value === 1)).toBe(true);
  });

  it('uses the default grid size when none is given', () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    expect(sampleGrid(model)).toHaveLength(GRID_SIZE * GRID_SIZE);
  });
});
