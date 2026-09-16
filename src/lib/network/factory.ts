import type { Block, BlockKind, LinearBlock, Network, TrainingConfig } from './types';

const DEFAULT_TRAINING: TrainingConfig = {
  loss: 'crossEntropy',
  optimizer: 'adam',
  learningRate: 0.01,
  batchSize: 32
};

export function newBlockId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `b_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

export function createBlock(kind: BlockKind): Block {
  switch (kind) {
    case 'input':
      return { id: newBlockId(), kind: 'input', shape: [2] };
    case 'linear':
      return { id: newBlockId(), kind: 'linear', units: 8 };
    case 'conv2d':
      return {
        id: newBlockId(),
        kind: 'conv2d',
        filters: 8,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      };
    case 'flatten':
      return { id: newBlockId(), kind: 'flatten' };
    case 'relu':
    case 'sigmoid':
    case 'softmax':
      return { id: newBlockId(), kind };
    case 'output':
      return { id: newBlockId(), kind: 'output', units: 2 };
  }
}

export function createEmptyNetwork(): Network {
  const input = createBlock('input');
  const hidden = createBlock('linear');
  const activation = createBlock('relu');
  const outputLayer: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 2 };
  const softmax = createBlock('softmax');
  const output = createBlock('output');

  return {
    version: 2,
    blocks: [input, hidden, activation, outputLayer, softmax, output],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
}

export function cloneNetwork(net: Network): Network {
  return {
    version: net.version,
    blocks: net.blocks.map((block) => ({ ...block })),
    training: { ...net.training },
    positions: Object.fromEntries(
      Object.entries(net.positions).map(([id, position]) => [id, { ...position }])
    )
  };
}
