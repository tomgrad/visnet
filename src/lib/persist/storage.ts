import type { PointDataset } from '../data/points';
import type { Network } from '../network/types';
import { decodeNetwork, encodeNetwork } from './networkCodec';

export interface StorageKeys {
  network: string;
  dataset: string;
}

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

export function createStorage(backing: KeyValueStore, keys: StorageKeys): NetworkStorage {
  return {
    saveNetwork(net) {
      backing.setItem(keys.network, encodeNetwork(net));
    },
    loadNetwork() {
      const raw = backing.getItem(keys.network);
      return raw === null ? null : decodeNetwork(raw);
    },
    hasStoredNetwork() {
      return backing.getItem(keys.network) !== null;
    },
    saveDataset(dataset) {
      backing.setItem(keys.dataset, JSON.stringify({ points: dataset.points }));
    },
    loadDataset() {
      const raw = backing.getItem(keys.dataset);
      return raw === null ? null : parseDataset(raw);
    }
  };
}

export function createBrowserStorage(keys: StorageKeys): NetworkStorage | null {
  try {
    const probe = '__visnet_probe__';
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return createStorage(window.localStorage, keys);
  } catch {
    return null;
  }
}
