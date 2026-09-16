import { beforeEach, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { PointDataset } from '../data/points';
import { createStorage, DATASET_KEY, NETWORK_KEY, type KeyValueStore } from './storage';

function fakeStore(): KeyValueStore & { entries: Map<string, string> } {
  const entries = new Map<string, string>();
  return {
    entries,
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => void entries.set(key, value),
    removeItem: (key) => void entries.delete(key)
  };
}

let backing: ReturnType<typeof fakeStore>;

beforeEach(() => {
  backing = fakeStore();
});

describe('network storage', () => {
  it('round-trips a network', () => {
    const storage = createStorage(backing);
    const network = createEmptyNetwork();
    storage.saveNetwork(network);
    expect(storage.loadNetwork()).toEqual(network);
    expect(backing.entries.has(NETWORK_KEY)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });

  it('returns null for corrupt data', () => {
    backing.setItem(NETWORK_KEY, '{not json');
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });

  it('returns null for an unsupported version', () => {
    backing.setItem(NETWORK_KEY, JSON.stringify({ version: 99, network: createEmptyNetwork() }));
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });
});

describe('dataset storage', () => {
  const dataset: PointDataset = {
    points: [
      { x: 0.1, y: -0.2, label: 0 },
      { x: -0.3, y: 0.4, label: 1 }
    ],
    numClasses: 2
  };

  it('round-trips a dataset', () => {
    const storage = createStorage(backing);
    storage.saveDataset(dataset);
    expect(storage.loadDataset()).toEqual(dataset);
    expect(backing.entries.has(DATASET_KEY)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects a payload that is not a list of points', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: 'a', y: 0, label: 0 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects a label that is not 0 or 1', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: 0, y: 0, label: 7 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects non-finite coordinates', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: null, y: 0, label: 0 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('accepts an empty dataset', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [] }));
    expect(createStorage(backing).loadDataset()).toEqual({ points: [], numClasses: 2 });
  });
});

describe('clear', () => {
  it('removes both entries', () => {
    const storage = createStorage(backing);
    storage.saveNetwork(createEmptyNetwork());
    storage.saveDataset({ points: [], numClasses: 2 });
    storage.clear();
    expect(backing.entries.size).toBe(0);
  });
});
