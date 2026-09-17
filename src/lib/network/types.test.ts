import { describe, expect, it } from 'vitest';
import { BLOCK_KINDS, type Block, type Network } from './types';

describe('BLOCK_KINDS', () => {
  it('lists the nine block kinds in pipeline order', () => {
    expect([...BLOCK_KINDS]).toEqual([
      'input',
      'linear',
      'conv2d',
      'maxpool2d',
      'flatten',
      'relu',
      'sigmoid',
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
      { id: 'd', kind: 'flatten' },
      { id: 'e', kind: 'relu' },
      { id: 'f', kind: 'sigmoid' },
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
