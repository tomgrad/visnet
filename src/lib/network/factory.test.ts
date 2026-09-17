import { describe, expect, it } from 'vitest';
import { cloneNetwork, createBlock, createEmptyNetwork, newBlockId } from './factory';
import { BLOCK_KINDS } from './types';

describe('createBlock', () => {
  it.each([...BLOCK_KINDS])('creates a %s block with a unique id', (kind) => {
    const a = createBlock(kind);
    const b = createBlock(kind);
    expect(a.kind).toBe(kind);
    expect(a.id).toBeTruthy();
    expect(a.id).not.toBe(b.id);
  });

  it('applies the documented defaults', () => {
    expect(createBlock('input')).toMatchObject({ kind: 'input', shape: [2] });
    expect(createBlock('linear')).toMatchObject({ kind: 'linear', units: 8 });
    expect(createBlock('conv2d')).toMatchObject({
      kind: 'conv2d',
      filters: 8,
      kernelSize: 3,
      stride: 1,
      padding: 'same'
    });
    expect(createBlock('maxpool2d')).toMatchObject({
      kind: 'maxpool2d',
      poolSize: 2,
      stride: 2,
      padding: 'valid'
    });
    expect(createBlock('reshape')).toMatchObject({ kind: 'reshape', shape: [1] });
    expect(createBlock('output')).toMatchObject({ kind: 'output', units: 2 });
  });
});

describe('newBlockId', () => {
  it('returns a different id on every call', () => {
    const ids = new Set(Array.from({ length: 50 }, () => newBlockId()));
    expect(ids.size).toBe(50);
  });
});

describe('createEmptyNetwork', () => {
  it('builds the default MLP', () => {
    const net = createEmptyNetwork();
    expect(net.version).toBe(2);
    expect(net.positions).toEqual({});
    expect(net.blocks.map((b) => b.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
    expect(net.blocks[3]).toMatchObject({ kind: 'linear', units: 2 });
    expect(net.training).toEqual({
      loss: 'crossEntropy',
      optimizer: 'adam',
      learningRate: 0.01,
      batchSize: 32
    });
  });

  it('gives every block a distinct id', () => {
    const net = createEmptyNetwork();
    expect(new Set(net.blocks.map((b) => b.id)).size).toBe(net.blocks.length);
  });

  it('returns a fresh object each time', () => {
    const a = createEmptyNetwork();
    const b = createEmptyNetwork();
    expect(a).not.toBe(b);
    expect(a.blocks[0].id).not.toBe(b.blocks[0].id);
  });
});

describe('cloneNetwork', () => {
  it('deep-copies blocks so mutations do not leak', () => {
    const net = createEmptyNetwork();
    const copy = cloneNetwork(net);
    expect(copy).not.toBe(net);
    expect(copy.blocks).not.toBe(net.blocks);
    expect(copy.blocks[0]).not.toBe(net.blocks[0]);
    expect(copy.blocks[0]).toEqual(net.blocks[0]);
    expect(copy.training).not.toBe(net.training);
    expect(copy.training).toEqual(net.training);
  });
});
