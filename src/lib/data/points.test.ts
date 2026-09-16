import { describe, expect, it } from 'vitest';
import {
  GENERATOR_DESCRIPTIONS,
  GENERATOR_NAMES,
  addPoint,
  clearPoints,
  generate
} from './points';

describe('GENERATOR_DESCRIPTIONS', () => {
  it.each(GENERATOR_NAMES)('describes %s', (name) => {
    expect(GENERATOR_DESCRIPTIONS[name]).toBeTruthy();
  });
});

describe('generate', () => {
  it.each(GENERATOR_NAMES)('produces the requested number of points for %s', (name) => {
    const dataset = generate(name, 120, 1);
    expect(dataset.points).toHaveLength(120);
    expect(dataset.numClasses).toBe(2);
  });

  it.each(GENERATOR_NAMES)('keeps %s inside the domain', (name) => {
    for (const point of generate(name, 200, 3).points) {
      expect(point.x).toBeGreaterThanOrEqual(-1);
      expect(point.x).toBeLessThanOrEqual(1);
      expect(point.y).toBeGreaterThanOrEqual(-1);
      expect(point.y).toBeLessThanOrEqual(1);
      expect([0, 1]).toContain(point.label);
    }
  });

  it('uses both classes', () => {
    const labels = new Set(generate('twoGaussians', 100, 5).points.map((p) => p.label));
    expect(labels).toEqual(new Set([0, 1]));
  });

  it('is reproducible for a given seed', () => {
    expect(generate('spirals', 50, 9)).toEqual(generate('spirals', 50, 9));
  });

  it('changes when the seed changes', () => {
    expect(generate('spirals', 50, 9)).not.toEqual(generate('spirals', 50, 10));
  });

  it('labels xor quadrants in a checkerboard', () => {
    for (const point of generate('xor', 400, 11).points) {
      const expected = (point.x > 0) !== (point.y > 0) ? 1 : 0;
      expect(point.label).toBe(expected);
    }
  });

  it('labels circles by distance from the centre', () => {
    for (const point of generate('circles', 400, 13).points) {
      const radius = Math.hypot(point.x, point.y);
      if (point.label === 0) expect(radius).toBeLessThanOrEqual(0.3);
      else expect(radius).toBeGreaterThanOrEqual(0.6);
    }
  });
});

describe('addPoint', () => {
  it('appends without mutating the original', () => {
    const original = generate('xor', 4, 1);
    const updated = addPoint(original, 0.25, -0.25, 1);
    expect(original.points).toHaveLength(4);
    expect(updated.points).toHaveLength(5);
    expect(updated.points[4]).toEqual({ x: 0.25, y: -0.25, label: 1 });
    expect(updated.numClasses).toBe(2);
  });
});

describe('clearPoints', () => {
  it('empties the dataset without mutating the original', () => {
    const original = generate('xor', 4, 1);
    const cleared = clearPoints(original);
    expect(original.points).toHaveLength(4);
    expect(cleared.points).toEqual([]);
    expect(cleared.numClasses).toBe(2);
  });
});
