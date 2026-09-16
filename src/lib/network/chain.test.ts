import { describe, expect, it } from 'vitest';
import { insertAt, moveBlock, removeBlock, replaceBlock } from './chain';
import { createBlock } from './factory';
import type { Network } from './types';

function net(): Network {
  return {
    version: 2,
    blocks: [
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'a', kind: 'linear', units: 8 },
      { id: 'b', kind: 'relu' },
      { id: 'c', kind: 'linear', units: 2 },
      { id: 'out', kind: 'output', units: 2 }
    ],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
    positions: {}
  };
}

const ids = (n: Network) => n.blocks.map((b) => b.id);

describe('insertAt', () => {
  it('inserts at an interior index', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 2, block))).toEqual(['in', 'a', block.id, 'b', 'c', 'out']);
  });

  it('clamps an index past the end to just before the output', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 99, block))).toEqual(['in', 'a', 'b', 'c', block.id, 'out']);
  });

  it('clamps an index before the input to just after it', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 0, block))).toEqual(['in', block.id, 'a', 'b', 'c', 'out']);
    expect(ids(insertAt(net(), -5, block))).toEqual(['in', block.id, 'a', 'b', 'c', 'out']);
  });

  it('does not mutate the original network', () => {
    const original = net();
    insertAt(original, 2, createBlock('relu'));
    expect(ids(original)).toEqual(['in', 'a', 'b', 'c', 'out']);
  });
});

describe('moveBlock', () => {
  it('moves a block earlier', () => {
    expect(ids(moveBlock(net(), 3, 1))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });

  it('moves a block later', () => {
    expect(ids(moveBlock(net(), 1, 3))).toEqual(['in', 'b', 'c', 'a', 'out']);
  });

  it('refuses to move the input or the output', () => {
    const original = net();
    expect(moveBlock(original, 0, 2)).toBe(original);
    expect(moveBlock(original, 4, 2)).toBe(original);
  });

  it('returns the same reference when the move is a no-op', () => {
    const original = net();
    expect(moveBlock(original, 2, 2)).toBe(original);
  });

  it('clamps the destination into the movable range', () => {
    expect(ids(moveBlock(net(), 1, 99))).toEqual(['in', 'b', 'c', 'a', 'out']);
    expect(ids(moveBlock(net(), 3, -99))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });
});

describe('removeBlock', () => {
  it('removes an interior block', () => {
    expect(ids(removeBlock(net(), 'b'))).toEqual(['in', 'a', 'c', 'out']);
  });

  it('refuses to remove the input, the output, or an unknown id', () => {
    const original = net();
    expect(removeBlock(original, 'in')).toBe(original);
    expect(removeBlock(original, 'out')).toBe(original);
    expect(removeBlock(original, 'missing')).toBe(original);
  });
});

describe('replaceBlock', () => {
  it('merges the patch into the matching block', () => {
    const updated = replaceBlock(net(), 'a', { units: 16 });
    expect(updated.blocks[1]).toMatchObject({ id: 'a', kind: 'linear', units: 16 });
    expect(updated.blocks[2]).toEqual({ id: 'b', kind: 'relu' });
  });

  it('returns the same reference for an unknown id', () => {
    const original = net();
    expect(replaceBlock(original, 'missing', { units: 16 })).toBe(original);
  });

  it('does not mutate the original block', () => {
    const original = net();
    replaceBlock(original, 'a', { units: 16 });
    expect(original.blocks[1]).toMatchObject({ units: 8 });
  });
});
