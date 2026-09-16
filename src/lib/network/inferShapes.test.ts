import { describe, expect, it } from 'vitest';
import { convOutputSize, inferShapes } from './inferShapes';
import { createEmptyNetwork } from './factory';
import type { Network } from './types';

function net(blocks: Network['blocks']): Network {
  return {
    version: 1,
    blocks,
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

describe('convOutputSize', () => {
  it('keeps the size with same padding', () => {
    expect(convOutputSize(28, 3, 1, 'same')).toBe(28);
    expect(convOutputSize(28, 3, 2, 'same')).toBe(14);
  });

  it('shrinks with valid padding', () => {
    expect(convOutputSize(28, 3, 1, 'valid')).toBe(26);
    expect(convOutputSize(28, 5, 2, 'valid')).toBe(12);
  });
});

describe('inferShapes on the default MLP', () => {
  const result = inferShapes(createEmptyNetwork());

  it('produces one entry per block', () => {
    expect(result.perBlock).toHaveLength(6);
  });

  it('tracks shapes through the pipeline', () => {
    expect(result.perBlock.map((p) => p.inShape)).toEqual([
      null,
      [2],
      [8],
      [8],
      [2],
      [2]
    ]);
    expect(result.perBlock.map((p) => p.outShape)).toEqual([
      [2],
      [8],
      [8],
      [2],
      [2],
      [2]
    ]);
  });

  it('counts parameters', () => {
    expect(result.perBlock.map((p) => p.paramCount)).toEqual([0, 24, 0, 18, 0, 0]);
    expect(result.totalParamCount).toBe(42);
  });

  it('labels every edge with the shape travelling along it', () => {
    expect(result.edges).toHaveLength(5);
    expect(result.edges.map((e) => e.shape)).toEqual([[2], [8], [8], [2], [2]]);
  });
});

describe('inferShapes on a convolutional chain', () => {
  const result = inferShapes(
    net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'out', kind: 'output', units: 10 }
    ])
  );

  it('computes the convolution output shape', () => {
    expect(result.perBlock[1].outShape).toEqual([28, 28, 4]);
  });

  it('flattens to a single dimension', () => {
    expect(result.perBlock[2].outShape).toEqual([28 * 28 * 4]);
  });

  it('counts convolution parameters from the input channels', () => {
    expect(result.perBlock[1].paramCount).toBe(3 * 3 * 1 * 4 + 4);
  });
});

describe('inferShapes error propagation', () => {
  const result = inferShapes(
    net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'dense', kind: 'linear', units: 8 },
      { id: 'out', kind: 'output', units: 2 }
    ])
  );

  it('records the offending input shape but no output shape', () => {
    expect(result.perBlock[1].inShape).toEqual([28, 28, 1]);
    expect(result.perBlock[1].outShape).toBeNull();
    expect(result.perBlock[1].paramCount).toBeNull();
  });

  it('nulls out every later shape', () => {
    expect(result.perBlock[2].inShape).toBeNull();
    expect(result.perBlock[2].outShape).toBeNull();
  });

  it('excludes unknown counts from the total', () => {
    expect(result.totalParamCount).toBe(0);
  });
});

describe('inferShapes with an impossible convolution', () => {
  it('returns a null output shape', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [2, 2, 1] },
        { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 3, padding: 'valid' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].inShape).toEqual([2, 2, 1]);
    expect(result.perBlock[1].outShape).toBeNull();
  });
});
