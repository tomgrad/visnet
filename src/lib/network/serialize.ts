import type { Block, Network, NodePosition, TrainingConfig } from './types';

const SUPPORTED_VERSION = 2;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFinitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function isInputShape(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every(isPositiveInteger);
}

function isBlock(value: unknown): value is Block {
  if (!isRecord(value)) return false;
  if (typeof value.id !== 'string' || value.id.length === 0) return false;
  switch (value.kind) {
    case 'input':
      return isInputShape(value.shape);
    case 'linear':
      return isPositiveInteger(value.units);
    case 'conv2d':
      return (
        isPositiveInteger(value.filters) &&
        isPositiveInteger(value.kernelSize) &&
        isPositiveInteger(value.stride) &&
        (value.padding === 'same' || value.padding === 'valid')
      );
    case 'maxpool2d':
      return (
        isPositiveInteger(value.poolSize) &&
        isPositiveInteger(value.stride) &&
        (value.padding === 'same' || value.padding === 'valid')
      );
    case 'output':
      return isPositiveInteger(value.units);
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
    isPositiveInteger(value.batchSize)
  );
}

function isNodePosition(value: unknown): value is NodePosition {
  if (!isRecord(value)) return false;
  return (
    typeof value.x === 'number' &&
    Number.isFinite(value.x) &&
    typeof value.y === 'number' &&
    Number.isFinite(value.y)
  );
}

function readPositions(value: unknown, blocks: Block[]): Record<string, NodePosition> | null {
  if (value === undefined) return {};
  if (!isRecord(value)) return null;
  const known = new Set(blocks.map((block) => block.id));
  const positions: Record<string, NodePosition> = {};
  for (const [id, position] of Object.entries(value)) {
    if (!known.has(id)) continue;
    if (!isNodePosition(position)) return null;
    positions[id] = { x: position.x, y: position.y };
  }
  return positions;
}

function normalize(value: unknown): Network | null {
  if (!isRecord(value)) return null;
  if (value.version !== SUPPORTED_VERSION) return null;
  if (!Array.isArray(value.blocks)) return null;
  const blocks: unknown[] = value.blocks;
  if (!blocks.every(isBlock)) return null;
  if (!isTrainingConfig(value.training)) return null;
  if (blocks.length < 2) return null;
  if (blocks[0].kind !== 'input') return null;
  if (blocks[blocks.length - 1].kind !== 'output') return null;
  if (blocks.filter((block) => block.kind === 'input').length !== 1) return null;
  if (blocks.filter((block) => block.kind === 'output').length !== 1) return null;
  if (new Set(blocks.map((block) => block.id)).size !== blocks.length) return null;

  const positions = readPositions(value.positions, blocks);
  if (positions === null) return null;

  return {
    version: SUPPORTED_VERSION,
    blocks: blocks.map((block) => ({ ...block })),
    training: { ...value.training },
    positions
  };
}

export function migrate(raw: unknown): Network | null {
  if (!isRecord(raw)) return null;

  if (raw.version === 1) {
    if (!isRecord(raw.network)) return null;
    return normalize({ ...raw.network, version: SUPPORTED_VERSION, positions: {} });
  }

  if (raw.version !== SUPPORTED_VERSION) return null;
  return normalize(raw.network);
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
