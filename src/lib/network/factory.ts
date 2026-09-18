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
    case 'maxpool2d':
      return { id: newBlockId(), kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' };
    case 'upsampling2d':
      return { id: newBlockId(), kind: 'upsampling2d', size: 2 };
    case 'conv2dtranspose':
      return {
        id: newBlockId(),
        kind: 'conv2dtranspose',
        filters: 8,
        kernelSize: 3,
        stride: 2,
        padding: 'same'
      };
    case 'flatten':
      return { id: newBlockId(), kind: 'flatten' };
    case 'reshape':
      return { id: newBlockId(), kind: 'reshape', shape: [1] };
    case 'relu':
    case 'sigmoid':
    case 'tanh':
    case 'softmax':
      return { id: newBlockId(), kind };
    case 'output':
      return { id: newBlockId(), kind: 'output', shape: [2] };
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
    blocks: [input, hidden, activation, outputLayer, softmax, output],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
}

export function cloneNetwork(net: Network): Network {
  return {
    blocks: net.blocks.map((block) => ({ ...block })),
    training: { ...net.training },
    positions: Object.fromEntries(
      Object.entries(net.positions).map(([id, position]) => [id, { ...position }])
    )
  };
}
