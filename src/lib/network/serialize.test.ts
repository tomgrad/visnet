import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import { fromJSON, migrate, toJSON } from './serialize';
import type { Network } from './types';

const TRAINING = {
  loss: 'crossEntropy',
  optimizer: 'adam',
  learningRate: 0.01,
  batchSize: 32
} as const;

const envelope = (blocks: unknown[], training: unknown = TRAINING) => ({
  version: 1,
  network: { version: 1, blocks, training }
});

const INPUT = { id: 'in', kind: 'input', shape: [2] };
const OUTPUT = { id: 'out', kind: 'output', units: 2 };
const LINEAR = { id: 'l', kind: 'linear', units: 2 };

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

  it('rejects a linear block with no units', () => {
    expect(
      fromJSON(JSON.stringify(envelope([INPUT, { id: 'l', kind: 'linear' }, OUTPUT])))
    ).toBeNull();
  });

  it('rejects a conv2d block with an invalid padding', () => {
    const conv = { id: 'c', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'half' };
    expect(fromJSON(JSON.stringify(envelope([INPUT, conv, OUTPUT])))).toBeNull();
  });

  it('rejects a network whose first block is not an input', () => {
    expect(fromJSON(JSON.stringify(envelope([LINEAR, OUTPUT])))).toBeNull();
  });

  it('rejects a network whose last block is not an output', () => {
    expect(fromJSON(JSON.stringify(envelope([INPUT, LINEAR])))).toBeNull();
  });

  it('rejects a network with two input blocks', () => {
    expect(
      fromJSON(JSON.stringify(envelope([INPUT, { id: 'in2', kind: 'input', shape: [2] }, OUTPUT])))
    ).toBeNull();
  });

  it('still accepts the default network', () => {
    expect(fromJSON(toJSON(createEmptyNetwork()))).not.toBeNull();
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

  it('rejects a non-finite learning rate', () => {
    expect(
      migrate({
        version: 1,
        network: {
          version: 1,
          blocks: [INPUT, LINEAR, OUTPUT],
          training: { ...TRAINING, learningRate: Infinity }
        }
      })
    ).toBeNull();
  });

  it('rejects a non-finite batch size', () => {
    expect(
      migrate({
        version: 1,
        network: {
          version: 1,
          blocks: [INPUT, LINEAR, OUTPUT],
          training: { ...TRAINING, batchSize: NaN }
        }
      })
    ).toBeNull();
  });

  it('rejects non-objects', () => {
    expect(migrate(null)).toBeNull();
    expect(migrate('nope')).toBeNull();
    expect(migrate(42)).toBeNull();
  });
});
