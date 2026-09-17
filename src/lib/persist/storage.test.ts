import { beforeEach, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { PointDataset } from '../data/points';
import { createStorage, type KeyValueStore, type StorageKeys } from './storage';

const KEYS: StorageKeys = { network: 'test:network', dataset: 'test:dataset' };

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
    const storage = createStorage(backing, KEYS);
    const network = createEmptyNetwork();
    storage.saveNetwork(network);
    expect(storage.loadNetwork()).toEqual(network);
    expect(backing.entries.has(KEYS.network)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing, KEYS).loadNetwork()).toBeNull();
  });

  it('returns null for corrupt data', () => {
    backing.setItem(KEYS.network, '{not json');
    expect(createStorage(backing, KEYS).loadNetwork()).toBeNull();
  });

  it('distinguishes an unreadable saved network from no saved network', () => {
    const storage = createStorage(backing, KEYS);
    expect(storage.hasStoredNetwork()).toBe(false);

    storage.saveNetwork(createEmptyNetwork());
    expect(storage.hasStoredNetwork()).toBe(true);

    backing.setItem(KEYS.network, '{not json');
    expect(storage.hasStoredNetwork()).toBe(true);
    expect(storage.loadNetwork()).toBeNull();
  });

  it('rejects a legacy versioned envelope', () => {
    backing.setItem(KEYS.network, JSON.stringify({ version: 99, network: createEmptyNetwork() }));
    expect(createStorage(backing, KEYS).loadNetwork()).toBeNull();
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
    const storage = createStorage(backing, KEYS);
    storage.saveDataset(dataset);
    expect(storage.loadDataset()).toEqual(dataset);
    expect(backing.entries.has(KEYS.dataset)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing, KEYS).loadDataset()).toBeNull();
  });

  it('rejects a payload that is not a list of points', () => {
    backing.setItem(KEYS.dataset, JSON.stringify({ points: [{ x: 'a', y: 0, label: 0 }] }));
    expect(createStorage(backing, KEYS).loadDataset()).toBeNull();
  });

  it('rejects a label that is not 0 or 1', () => {
    backing.setItem(KEYS.dataset, JSON.stringify({ points: [{ x: 0, y: 0, label: 7 }] }));
    expect(createStorage(backing, KEYS).loadDataset()).toBeNull();
  });

  it('rejects non-finite coordinates', () => {
    backing.setItem(KEYS.dataset, JSON.stringify({ points: [{ x: null, y: 0, label: 0 }] }));
    expect(createStorage(backing, KEYS).loadDataset()).toBeNull();
  });

  it('accepts an empty dataset', () => {
    backing.setItem(KEYS.dataset, JSON.stringify({ points: [] }));
    expect(createStorage(backing, KEYS).loadDataset()).toEqual({ points: [], numClasses: 2 });
  });
});

describe('key namespacing', () => {
  it('does not read another key set data', () => {
    const first = createStorage(backing, { network: 'a:network', dataset: 'a:dataset' });
    const second = createStorage(backing, { network: 'b:network', dataset: 'b:dataset' });

    first.saveNetwork(createEmptyNetwork());

    expect(first.loadNetwork()).not.toBeNull();
    expect(second.loadNetwork()).toBeNull();
    expect(second.hasStoredNetwork()).toBe(false);
  });
});
