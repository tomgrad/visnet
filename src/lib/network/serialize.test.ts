import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import { fromJSON, migrate, toJSON } from './serialize';
import type { Network } from './types';

describe('toJSON', () => {
  it('wraps the network in a versioned envelope', () => {
    const envelope = JSON.parse(toJSON(createEmptyNetwork()));
    expect(envelope.version).toBe(1);
    expect(envelope.network.blocks).toHaveLength(6);
  });
});

describe('fromJSON', () => {
  it('round-trips a network', () => {
    const original = createEmptyNetwork();
    const restored = fromJSON(toJSON(original));
    expect(restored).toEqual(original);
  });

  it('preserves training configuration', () => {
    const original = createEmptyNetwork();
    original.training.optimizer = 'sgd';
    original.training.learningRate = 0.25;
    const restored = fromJSON(toJSON(original));
    expect(restored?.training).toEqual(original.training);
  });

  it('returns null for invalid JSON', () => {
    expect(fromJSON('not json')).toBeNull();
  });

  it('returns null for a missing envelope', () => {
    expect(fromJSON(JSON.stringify({ blocks: [] }))).toBeNull();
  });

  it('returns null for an unsupported future version', () => {
    expect(fromJSON(JSON.stringify({ version: 99, network: createEmptyNetwork() }))).toBeNull();
  });

  it('returns null when the network is structurally wrong', () => {
    expect(fromJSON(JSON.stringify({ version: 1, network: { blocks: 'nope' } }))).toBeNull();
    expect(fromJSON(JSON.stringify({ version: 1, network: { blocks: [{ id: 1 }] } }))).toBeNull();
    expect(
      fromJSON(JSON.stringify({ version: 1, network: { blocks: [], training: {} } }))
    ).toBeNull();
  });
});

describe('migrate', () => {
  it('accepts version 1', () => {
    const original: Network = createEmptyNetwork();
    expect(migrate({ version: 1, network: original })).toEqual(original);
  });

  it('rejects unknown versions', () => {
    expect(migrate({ version: 2, network: createEmptyNetwork() })).toBeNull();
  });

  it('rejects non-objects', () => {
    expect(migrate(null)).toBeNull();
    expect(migrate('nope')).toBeNull();
    expect(migrate(42)).toBeNull();
  });
});
