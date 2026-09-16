import type { Block, Network, TrainingConfig } from './types';

const SUPPORTED_VERSION = 1;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isBlock(value: unknown): value is Block {
  if (!isRecord(value)) return false;
  return typeof value.id === 'string' && typeof value.kind === 'string';
}

function isTrainingConfig(value: unknown): value is TrainingConfig {
  if (!isRecord(value)) return false;
  return (
    (value.loss === 'mse' || value.loss === 'crossEntropy') &&
    (value.optimizer === 'sgd' || value.optimizer === 'adam') &&
    typeof value.learningRate === 'number' &&
    typeof value.batchSize === 'number'
  );
}

function isNetwork(value: unknown): value is Network {
  if (!isRecord(value)) return false;
  if (value.version !== SUPPORTED_VERSION) return false;
  if (!Array.isArray(value.blocks) || !value.blocks.every(isBlock)) return false;
  return isTrainingConfig(value.training);
}

export function migrate(raw: unknown): Network | null {
  if (!isRecord(raw)) return null;
  if (raw.version !== SUPPORTED_VERSION) return null;
  return isNetwork(raw.network) ? raw.network : null;
}

export function toJSON(net: Network): string {
  return JSON.stringify({ version: net.version, network: net });
}

export function fromJSON(raw: string): Network | null {
  try {
    return migrate(JSON.parse(raw));
  } catch {
    return null;
  }
}
