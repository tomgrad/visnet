import { describe, expect, it } from 'vitest';
import { createVaeNetwork } from '../network/vae';
import { decodeVaeNetwork, encodeVaeNetwork } from './vaeCodec';

describe('vaeCodec', () => {
  it('round-trips a VAE network', () => {
    const vae = createVaeNetwork(4);
    const restored = decodeVaeNetwork(encodeVaeNetwork(vae));
    expect(restored).not.toBeNull();
    expect(restored?.latentSize).toBe(4);
    expect(restored?.encoder.blocks.length).toBe(vae.encoder.blocks.length);
    expect(restored?.decoder.blocks.length).toBe(vae.decoder.blocks.length);
    expect(restored?.training).toEqual(vae.training);
  });

  it('rejects a malformed payload', () => {
    expect(decodeVaeNetwork('{')).toBeNull();
    expect(decodeVaeNetwork(JSON.stringify({ latentSize: 2 }))).toBeNull();
    expect(
      decodeVaeNetwork(JSON.stringify({ encoder: {}, decoder: {}, latentSize: 2 }))
    ).toBeNull();
  });
});
