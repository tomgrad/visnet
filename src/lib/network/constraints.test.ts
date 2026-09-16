import { describe, expect, it } from 'vitest';
import { clampBlockPatch, clampNetwork, parameterBounds } from './constraints';
import type { Network } from './types';

function net(blocks: Network['blocks']): Network {
  return {
    version: 1,
    blocks,
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

const IMAGE_NETWORK = net([
  { id: 'in', kind: 'input', shape: [28, 28, 1] },
  { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
  { id: 'flat', kind: 'flatten' },
  { id: 'dense', kind: 'linear', units: 10 },
  { id: 'out', kind: 'output', units: 10 }
]);

const FLAT_NETWORK = net([
  { id: 'in', kind: 'input', shape: [2] },
  { id: 'dense', kind: 'linear', units: 8 },
  { id: 'out', kind: 'output', units: 2 }
]);

describe('parameterBounds', () => {
  it('returns no bounds for flat input', () => {
    expect(parameterBounds([2])).toBeNull();
    expect(parameterBounds(null)).toBeNull();
    expect(parameterBounds([4, 4])).toBeNull();
  });

  it('offers kernel and stride choices up to the smaller image dimension', () => {
    expect(parameterBounds([28, 28, 1])).toEqual({
      kernelSize: Array.from({ length: 28 }, (_, i) => i + 1),
      stride: Array.from({ length: 28 }, (_, i) => i + 1)
    });
    expect(parameterBounds([6, 4, 3])?.kernelSize).toEqual([1, 2, 3, 4]);
  });

  it('returns no bounds when a spatial dimension is not positive', () => {
    expect(parameterBounds([0, 4, 1])).toBeNull();
  });
});

describe('clampBlockPatch', () => {
  it('leaves a valid patch untouched and says nothing', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'dense', { units: 16 })).toEqual({
      patch: { units: 16 },
      announcement: null
    });
  });

  it('raises units to at least one and announces it', () => {
    const result = clampBlockPatch(FLAT_NETWORK, 'dense', { units: 0 });
    expect(result.patch).toEqual({ units: 1 });
    expect(result.announcement).toBe(
      'Units changed from 0 to 1. A layer must produce at least one number.'
    );
  });

  it('rounds units down to an integer and announces it', () => {
    const result = clampBlockPatch(FLAT_NETWORK, 'dense', { units: 7.6 });
    expect(result.patch).toEqual({ units: 7 });
    expect(result.announcement).toBe(
      'Units changed from 7.6 to 7. A layer must produce at least one number.'
    );
  });

  it('clamps filters and announces it with the filters wording', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { filters: 0 });
    expect(result.patch).toEqual({ filters: 1 });
    expect(result.announcement).toBe(
      'Filters changed from 0 to 1. A layer must produce at least one number.'
    );
  });

  it('clamps a kernel larger than the image and names the dimensions', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { kernelSize: 40 });
    expect(result.patch).toEqual({ kernelSize: 28 });
    expect(result.announcement).toBe(
      'Kernel size changed from 40 to 28 because the incoming data is 28×28.'
    );
  });

  it('clamps a stride larger than the image', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { stride: 99 });
    expect(result.patch).toEqual({ stride: 28 });
    expect(result.announcement).toBe(
      'Stride changed from 99 to 28 because the incoming data is 28×28.'
    );
  });

  it('leaves a kernel alone when the incoming data is not an image', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'dense', { units: 4 }).announcement).toBeNull();
  });

  it('says nothing for an unknown block', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'missing', { units: 0 })).toEqual({
      patch: { units: 0 },
      announcement: null
    });
  });

  it('clamps several parameters in one patch and reports the first correction', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { kernelSize: 40, stride: 50 });
    expect(result.patch).toEqual({ kernelSize: 28, stride: 28 });
    expect(result.announcement).toBe(
      'Kernel size changed from 40 to 28 because the incoming data is 28×28.'
    );
  });
});

describe('clampNetwork', () => {
  it('leaves a valid network untouched and returns the same reference', () => {
    const result = clampNetwork(IMAGE_NETWORK);
    expect(result.network).toBe(IMAGE_NETWORK);
    expect(result.announcements).toEqual([]);
  });

  it('re-clamps a downstream convolution when the input shape shrinks', () => {
    const shrunk = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 28, stride: 1, padding: 'same' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'out', kind: 'output', units: 10 }
    ]);

    const result = clampNetwork(shrunk);

    expect(result.network.blocks[1]).toMatchObject({ kernelSize: 4, stride: 1 });
    expect(result.announcements).toEqual([
      'Kernel size changed from 28 to 4 because the incoming data is 4×4.'
    ]);
  });

  it('leaves convolution parameters alone when the input is flat', () => {
    const flat = net([
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 28, stride: 1, padding: 'same' },
      { id: 'out', kind: 'output', units: 2 }
    ]);

    const result = clampNetwork(flat);

    expect(result.network).toBe(flat);
    expect(result.announcements).toEqual([]);
  });
});
