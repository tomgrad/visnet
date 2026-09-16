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
  version: 2,
  network: { version: 2, blocks, training, positions: {} }
});

const INPUT = { id: 'in', kind: 'input', shape: [2] };
const OUTPUT = { id: 'out', kind: 'output', units: 2 };
const LINEAR = { id: 'l', kind: 'linear', units: 2 };

describe('toJSON', () => {
  it('wraps the network in a versioned envelope', () => {
    const envelope = JSON.parse(toJSON(createEmptyNetwork()));
    expect(envelope.version).toBe(2);
    expect(envelope.network.blocks).toHaveLength(6);
    expect(envelope.network.positions).toEqual({});
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
    expect(fromJSON(JSON.stringify({ version: 2, network: { blocks: 'nope' } }))).toBeNull();
    expect(fromJSON(JSON.stringify({ version: 2, network: { blocks: [{ id: 1 }] } }))).toBeNull();
    expect(
      fromJSON(JSON.stringify({ version: 2, network: { blocks: [], training: {} } }))
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
  it('accepts the current version', () => {
    const original: Network = createEmptyNetwork();
    expect(migrate({ version: 2, network: original })).toEqual(original);
  });

  it('upgrades a version 1 payload by giving it empty positions', () => {
    const legacy = createEmptyNetwork();
    const upgraded = migrate({
      version: 1,
      network: { version: 1, blocks: legacy.blocks, training: legacy.training }
    });
    expect(upgraded).not.toBeNull();
    expect(upgraded?.version).toBe(2);
    expect(upgraded?.positions).toEqual({});
    expect(upgraded?.blocks).toEqual(legacy.blocks);
  });

  it('rejects a version 1 payload that is structurally broken', () => {
    expect(migrate({ version: 1, network: { blocks: 'nope' } })).toBeNull();
  });

  it('rejects unknown versions', () => {
    expect(migrate({ version: 3, network: createEmptyNetwork() })).toBeNull();
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

describe('positions', () => {
  it('round-trips stored positions', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[1].id;
    const restored = fromJSON(toJSON({ ...net, positions: { [id]: { x: 12, y: 34 } } }));
    expect(restored?.positions).toEqual({ [id]: { x: 12, y: 34 } });
  });

  it('rejects a position that is not a finite pair', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[1].id;
    const payload = envelope(net.blocks);
    payload.network.positions = { [id]: { x: 'nope', y: 0 } };
    expect(fromJSON(JSON.stringify(payload))).toBeNull();
  });

  it('rejects a non-finite coordinate', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[1].id;
    const payload = envelope(net.blocks);
    payload.network.positions = { [id]: { x: 0, y: Number.POSITIVE_INFINITY } };
    expect(fromJSON(JSON.stringify(payload))).toBeNull();
  });

  it('drops positions for blocks that are not in the network', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[1].id;
    const payload = envelope(net.blocks);
    payload.network.positions = { [id]: { x: 1, y: 2 }, ghost: { x: 9, y: 9 } };
    expect(fromJSON(JSON.stringify(payload))?.positions).toEqual({ [id]: { x: 1, y: 2 } });
  });

  it('rejects a current-version payload with no positions field', () => {
    const net = createEmptyNetwork();
    const payload = {
      version: 2,
      network: { version: 2, blocks: net.blocks, training: TRAINING }
    };
    expect(fromJSON(JSON.stringify(payload))).toBeNull();
  });
});
