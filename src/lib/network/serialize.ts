import type { Block, Network, TrainingConfig } from './types';

const SUPPORTED_VERSION = 1;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFinitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function isInputShape(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every(isFinitePositive);
}

function isBlock(value: unknown): value is Block {
  if (!isRecord(value)) return false;
  if (typeof value.id !== 'string') return false;
  switch (value.kind) {
    case 'input':
      return isInputShape(value.shape);
    case 'linear':
      return isFinitePositive(value.units);
    case 'conv2d':
      return (
        isFinitePositive(value.filters) &&
        isFinitePositive(value.kernelSize) &&
        isFinitePositive(value.stride) &&
        (value.padding === 'same' || value.padding === 'valid')
      );
    case 'output':
      return isFinitePositive(value.units);
    case 'flatten':
    case 'relu':
    case 'sigmoid':
    case 'softmax':
      return true;
    default:
      return false;
  }
}

function isTrainingConfig(value: unknown): value is TrainingConfig {
  if (!isRecord(value)) return false;
  return (
    (value.loss === 'mse' || value.loss === 'crossEntropy') &&
    (value.optimizer === 'sgd' || value.optimizer === 'adam') &&
    isFinitePositive(value.learningRate) &&
    isFinitePositive(value.batchSize)
  );
}

function isNetwork(value: unknown): value is Network {
  if (!isRecord(value)) return false;
  if (value.version !== SUPPORTED_VERSION) return false;
  if (!Array.isArray(value.blocks)) return false;
  const blocks: unknown[] = value.blocks;
  if (!blocks.every(isBlock)) return false;
  if (!isTrainingConfig(value.training)) return false;
  if (blocks.length < 2) return false;
  if (blocks[0].kind !== 'input') return false;
  if (blocks[blocks.length - 1].kind !== 'output') return false;
  if (blocks.filter((block) => block.kind === 'input').length !== 1) return false;
  if (blocks.filter((block) => block.kind === 'output').length !== 1) return false;
  return true;
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
