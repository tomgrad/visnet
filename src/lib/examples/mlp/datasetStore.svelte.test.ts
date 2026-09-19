import { describe, expect, it } from 'vitest';
import { GENERATOR_NAMES } from '../../data/points';
import { DatasetStore } from './datasetStore.svelte';
import { CLASS_LABELS, DEFAULT_GENERATOR } from './example';

describe('example configuration', () => {
  it('names both classes', () => {
    expect(CLASS_LABELS).toHaveLength(2);
    expect(CLASS_LABELS[0]).toBeTruthy();
    expect(CLASS_LABELS[1]).toBeTruthy();
  });

  it('starts from a known generator', () => {
    expect(GENERATOR_NAMES).toContain(DEFAULT_GENERATOR);
  });
});

describe('DatasetStore', () => {
  it('starts with a generated dataset', () => {
    const store = new DatasetStore();
    expect(store.dataset.points.length).toBeGreaterThan(0);
    expect(store.dataset.numClasses).toBe(2);
  });

  it('regenerates when the generator changes', () => {
    const store = new DatasetStore();
    store.generator = 'circles';
    store.regenerate();
    for (const point of store.dataset.points) {
      const radius = Math.hypot(point.x, point.y);
      if (point.label === 0) expect(radius).toBeLessThanOrEqual(0.3);
      else expect(radius).toBeGreaterThanOrEqual(0.6);
    }
  });

  it('respects the point count', () => {
    const store = new DatasetStore();
    store.pointCount = 40;
    store.regenerate();
    expect(store.dataset.points).toHaveLength(40);
  });

  it('reseeds to a different sample of the same size', () => {
    const store = new DatasetStore();
    const before = store.dataset;
    store.reseed();
    expect(store.dataset.points).toHaveLength(before.points.length);
    expect(store.dataset).not.toEqual(before);
  });

  it('adds a point with the selected label', () => {
    const store = new DatasetStore();
    store.clear();
    store.selectLabel(1);
    store.addPoint(0.25, -0.5);
    expect(store.dataset.points).toEqual([{ x: 0.25, y: -0.5, label: 1 }]);
  });

  it('clamps a click outside the canvas into the domain', () => {
    const store = new DatasetStore();
    store.clear();
    store.addPoint(4, -9);
    expect(store.dataset.points[0]).toEqual({ x: 1, y: -1, label: 0 });
  });

  it('clears every point but keeps the settings', () => {
    const store = new DatasetStore();
    store.generator = 'xor';
    store.pointCount = 25;
    store.clear();
    expect(store.dataset.points).toEqual([]);
    expect(store.dataset.numClasses).toBe(2);
    expect(store.generator).toBe('xor');
    expect(store.pointCount).toBe(25);
  });
});
