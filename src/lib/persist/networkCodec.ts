import type { Block, Network, NodePosition, TrainingConfig } from '../network/types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isBlock(value: unknown): value is Block {
  return isRecord(value) && typeof value.id === 'string' && typeof value.kind === 'string';
}

function isTraining(value: unknown): value is TrainingConfig {
  return isRecord(value) && typeof value.loss === 'string' && typeof value.optimizer === 'string';
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
    if (!Array.isArray(parsed.blocks) || !parsed.blocks.every(isBlock)) return null;
    if (!isTraining(parsed.training)) return null;
    return {
      version: 2,
      blocks: parsed.blocks,
      training: parsed.training,
      positions: readPositions(parsed.positions)
    };
  } catch {
    return null;
  }
}
