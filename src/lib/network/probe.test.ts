import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import { probeTargetFor } from './probe';
import type { Network } from './types';

const net = createEmptyNetwork();
const TRAINING = net.training;

describe('probeTargetFor', () => {
  it('maps the input block to the raw input', () => {
    expect(probeTargetFor(net, net.blocks[0].id)).toEqual({ source: 'input', dims: [2] });
  });

  it('maps a real layer to its model layer index and output shape', () => {
    expect(probeTargetFor(net, net.blocks[1].id)).toEqual({ source: 0, dims: [8] });
    expect(probeTargetFor(net, net.blocks[2].id)).toEqual({ source: 1, dims: [8] });
    expect(probeTargetFor(net, net.blocks[3].id)).toEqual({ source: 2, dims: [2] });
  });

  it('maps the output marker to the last real layer', () => {
    expect(probeTargetFor(net, net.blocks[5].id)).toEqual({ source: 3, dims: [2] });
  });

  it('returns null without a selection or for an unknown id', () => {
    expect(probeTargetFor(net, null)).toBeNull();
    expect(probeTargetFor(net, 'nope')).toBeNull();
  });

  it('maps the input but returns null for the output when there are no real layers', () => {
    const bare: Network = {
      version: 2,
      blocks: [net.blocks[0], net.blocks[5]],
      training: TRAINING,
      positions: {}
    };
    expect(probeTargetFor(bare, bare.blocks[0].id)).toEqual({ source: 'input', dims: [2] });
    expect(probeTargetFor(bare, bare.blocks[1].id)).toBeNull();
  });

  it('reports unknown dimensions when an earlier block is invalid', () => {
    const broken: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'l', kind: 'linear', units: 8 },
        { id: 'out', kind: 'output', units: 2 }
      ],
      training: TRAINING,
      positions: {}
    };
    expect(probeTargetFor(broken, 'l')).toEqual({ source: 0, dims: [] });
  });
});
