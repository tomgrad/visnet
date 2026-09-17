import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { decodeNetwork, encodeNetwork } from './networkCodec';

describe('networkCodec', () => {
  it('round-trips a network', () => {
    const net = createEmptyNetwork();
    expect(decodeNetwork(encodeNetwork(net))).toEqual(net);
  });

  it('returns null for invalid JSON', () => {
    expect(decodeNetwork('not json')).toBeNull();
  });

  it('returns null when the blocks array is missing', () => {
    expect(decodeNetwork(JSON.stringify({ training: {} }))).toBeNull();
  });

  it('returns null when a block lacks an id or kind', () => {
    expect(decodeNetwork(JSON.stringify({ blocks: [{ id: 'a' }] }))).toBeNull();
  });

  it('returns null when training is missing', () => {
    const net = createEmptyNetwork();
    expect(decodeNetwork(JSON.stringify({ blocks: net.blocks }))).toBeNull();
  });

  it('accepts a payload with no positions and defaults them to empty', () => {
    const net = createEmptyNetwork();
    const restored = decodeNetwork(
      JSON.stringify({ blocks: net.blocks, training: net.training })
    );
    expect(restored?.positions).toEqual({});
  });
});
