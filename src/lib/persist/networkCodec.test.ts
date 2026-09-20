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

  it('returns null when an input block lacks a numeric shape', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'a', kind: 'input' }], training }))
    ).toBeNull();
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: [{ id: 'a', kind: 'input', shape: 'square' }], training })
      )
    ).toBeNull();
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: [{ id: 'a', kind: 'input', shape: [2, 'x'] }], training })
      )
    ).toBeNull();
  });

  it('returns null when a training field is not one of the known choices', () => {
    const net = createEmptyNetwork();
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: net.blocks, training: { ...net.training, loss: 'x' } })
      )
    ).toBeNull();
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: net.blocks, training: { ...net.training, optimizer: 'y' } })
      )
    ).toBeNull();
  });

  it('rejects an upsampling2d block without a numeric size', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'u', kind: 'upsampling2d' }], training }))
    ).toBeNull();
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: [{ id: 'u', kind: 'upsampling2d', size: 'big' }], training })
      )
    ).toBeNull();
  });

  it('rejects a reshape block without a numeric shape', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'r', kind: 'reshape' }], training }))
    ).toBeNull();
  });

  it('rejects blocks with an empty shape array', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'a', kind: 'input', shape: [] }], training }))
    ).toBeNull();
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'a', kind: 'output', shape: [] }], training }))
    ).toBeNull();
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'r', kind: 'reshape', shape: [] }], training }))
    ).toBeNull();
  });

  it('loads a legacy output block that stores units', () => {
    const net = createEmptyNetwork();
    const blocks = net.blocks.map((block) =>
      block.kind === 'output' ? { id: block.id, kind: 'output', units: 2 } : block
    );
    const restored = decodeNetwork(JSON.stringify({ blocks, training: net.training }));
    expect(restored?.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [2] });
  });

  it('rejects a conv2dtranspose block without numeric fields', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(
        JSON.stringify({ blocks: [{ id: 'ct', kind: 'conv2dtranspose', filters: 4 }], training })
      )
    ).toBeNull();
  });

  it('round-trips an upsampling2d block', () => {
    const net = createEmptyNetwork();
    const block = { id: 'u', kind: 'upsampling2d', size: 3 };
    const restored = decodeNetwork(
      JSON.stringify({ blocks: [...net.blocks, block], training: net.training })
    );
    expect(restored?.blocks.at(-1)).toEqual(block);
  });

  it('round-trips a batchnorm block', () => {
    const net = createEmptyNetwork();
    const block = { id: 'bn', kind: 'batchnorm' };
    const restored = decodeNetwork(
      JSON.stringify({ blocks: [...net.blocks, block], training: net.training })
    );
    expect(restored?.blocks.at(-1)).toEqual(block);
  });

  it('accepts a payload with no positions and defaults them to empty', () => {
    const net = createEmptyNetwork();
    const restored = decodeNetwork(JSON.stringify({ blocks: net.blocks, training: net.training }));
    expect(restored?.positions).toEqual({});
  });

  it('round-trips a block colour and rejects an unknown one', () => {
    const net = createEmptyNetwork();
    const blocks = net.blocks.map((block, index) =>
      index === 1 ? { ...block, colour: 'blue' } : block
    );
    const restored = decodeNetwork(JSON.stringify({ blocks, training: net.training }));
    expect(restored?.blocks[1]).toMatchObject({ colour: 'blue' });

    const bogus = net.blocks.map((block, index) =>
      index === 1 ? { ...block, colour: 'chartreuse' } : block
    );
    expect(decodeNetwork(JSON.stringify({ blocks: bogus, training: net.training }))).toBeNull();
  });
});
