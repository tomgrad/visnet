import {
  isBlockColour,
  type Block,
  type Network,
  type NodePosition,
  type TrainingConfig
} from '../network/types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isBlock(value: unknown): value is Block {
  if (!isRecord(value)) return false;
  if (typeof value.id !== 'string' || typeof value.kind !== 'string') return false;
  if (value.colour !== undefined && !isBlockColour(value.colour)) return false;
  switch (value.kind) {
    case 'input':
      return (
        Array.isArray(value.shape) &&
        value.shape.length > 0 &&
        value.shape.every((n) => typeof n === 'number')
      );
    case 'linear':
      return typeof value.units === 'number';
    case 'output':
      return (
        Array.isArray(value.shape) &&
        value.shape.length > 0 &&
        value.shape.every((n) => typeof n === 'number')
      );
    case 'conv2d':
    case 'conv2dtranspose':
      return (
        typeof value.filters === 'number' &&
        typeof value.kernelSize === 'number' &&
        typeof value.stride === 'number'
      );
    case 'maxpool2d':
      return typeof value.poolSize === 'number' && typeof value.stride === 'number';
    case 'upsampling2d':
      return typeof value.size === 'number';
    case 'batchnorm':
      return true;
    case 'reshape':
      return (
        Array.isArray(value.shape) &&
        value.shape.length > 0 &&
        value.shape.every((n) => typeof n === 'number')
      );
    default:
      return true;
  }
}

function isTraining(value: unknown): value is TrainingConfig {
  return (
    isRecord(value) &&
    (value.loss === 'mse' || value.loss === 'crossEntropy') &&
    (value.optimizer === 'sgd' || value.optimizer === 'adam') &&
    typeof value.learningRate === 'number' &&
    typeof value.batchSize === 'number'
  );
}

function readPositions(value: unknown): Record<string, NodePosition> {
  if (!isRecord(value)) return {};
  const positions: Record<string, NodePosition> = {};
  for (const [id, position] of Object.entries(value)) {
    if (isRecord(position) && typeof position.x === 'number' && typeof position.y === 'number') {
      positions[id] = { x: position.x, y: position.y };
    }
  }
  return positions;
}

export function encodeNetwork(net: Network): string {
  return JSON.stringify(net);
}

export function decodeNetwork(raw: string): Network | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return null;
    if (!Array.isArray(parsed.blocks)) return null;
    const blocks = parsed.blocks.map((block) =>
      isRecord(block) &&
      block.kind === 'output' &&
      block.shape === undefined &&
      typeof block.units === 'number'
        ? { ...block, shape: [block.units] }
        : block
    );
    if (!blocks.every(isBlock)) return null;
    if (!isTraining(parsed.training)) return null;
    return {
      blocks,
      training: parsed.training,
      positions: readPositions(parsed.positions)
    };
  } catch {
    return null;
  }
}
