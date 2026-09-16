import type { PointDataset } from '../data/points';
import type { Network } from '../network/types';
import { fromJSON, toJSON } from '../network/serialize';

export const NETWORK_KEY = 'visnet:network:v1';
export const DATASET_KEY = 'visnet:mlp:dataset:v1';

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface NetworkStorage {
  saveNetwork(net: Network): void;
  loadNetwork(): Network | null;
  hasStoredNetwork(): boolean;
  saveDataset(dataset: PointDataset): void;
  loadDataset(): PointDataset | null;
  clear(): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isPoint(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (typeof value.x !== 'number' || !Number.isFinite(value.x)) return false;
  if (typeof value.y !== 'number' || !Number.isFinite(value.y)) return false;
  return value.label === 0 || value.label === 1;
}

function parseDataset(raw: string): PointDataset | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || !Array.isArray(parsed.points)) return null;
    if (!parsed.points.every(isPoint)) return null;
    return { points: parsed.points as PointDataset['points'], numClasses: 2 };
  } catch {
    return null;
  }
}

export function createStorage(backing: KeyValueStore): NetworkStorage {
  return {
    saveNetwork(net) {
      backing.setItem(NETWORK_KEY, toJSON(net));
    },
    loadNetwork() {
      const raw = backing.getItem(NETWORK_KEY);
      return raw === null ? null : fromJSON(raw);
    },
    hasStoredNetwork() {
      return backing.getItem(NETWORK_KEY) !== null;
    },
    saveDataset(dataset) {
      backing.setItem(DATASET_KEY, JSON.stringify({ points: dataset.points }));
    },
    loadDataset() {
      const raw = backing.getItem(DATASET_KEY);
      return raw === null ? null : parseDataset(raw);
    },
    clear() {
      backing.removeItem(NETWORK_KEY);
      backing.removeItem(DATASET_KEY);
    }
  };
}

export function createBrowserStorage(): NetworkStorage | null {
  try {
    const probe = '__visnet_probe__';
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return createStorage(window.localStorage);
  } catch {
    return null;
  }
}
