import { describe, expect, it } from 'vitest';
import { BLOCK_KINDS, type Block, type Network } from './types';

describe('BLOCK_KINDS', () => {
  it('lists the twelve block kinds in pipeline order', () => {
    expect([...BLOCK_KINDS]).toEqual([
      'input',
      'linear',
      'conv2d',
      'maxpool2d',
      'upsampling2d',
      'flatten',
      'reshape',
      'relu',
      'sigmoid',
      'tanh',
      'softmax',
      'output'
    ]);
  });

  it('contains no duplicates', () => {
    expect(new Set(BLOCK_KINDS).size).toBe(BLOCK_KINDS.length);
  });
});

describe('domain types', () => {
  it('has a valid block representation for every kind', () => {
    const blocks: Block[] = [
      { id: 'a', kind: 'input', shape: [2] },
      { id: 'b', kind: 'linear', units: 8 },
      { id: 'c', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'i', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      { id: 'k', kind: 'upsampling2d', size: 2 },
      { id: 'd', kind: 'flatten' },
      { id: 'r', kind: 'reshape', shape: [4] },
      { id: 'e', kind: 'relu' },
      { id: 'f', kind: 'sigmoid' },
      { id: 'j', kind: 'tanh' },
      { id: 'g', kind: 'softmax' },
      { id: 'h', kind: 'output', units: 2 }
    ];

    const network: Network = {
      version: 2,
      blocks,
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };

    expect(new Set(network.blocks.map((block) => block.kind))).toEqual(new Set(BLOCK_KINDS));
  });
});
