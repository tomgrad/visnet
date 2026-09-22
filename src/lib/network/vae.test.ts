import { describe, expect, it } from 'vitest';
import { findProblems } from './problems';
import { createVaeNetwork, syncLatentSize, syncTraining } from './vae';

describe('createVaeNetwork', () => {
  it('builds an encoder ending at the latent and a decoder starting from it', () => {
    const vae = createVaeNetwork(2);
    expect(vae.latentSize).toBe(2);
    const encoderOutput = vae.encoder.blocks.at(-1);
    const decoderInput = vae.decoder.blocks[0];
    expect(encoderOutput).toMatchObject({ kind: 'output', shape: [2] });
    expect(decoderInput).toMatchObject({ kind: 'input', shape: [2] });
    expect(vae.decoder.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [64, 64, 3] });
    expect(vae.encoder.blocks[0]).toMatchObject({ kind: 'input', shape: [64, 64, 3] });
  });

  it('defaults to a latent size of 2', () => {
    expect(createVaeNetwork().latentSize).toBe(2);
  });
});

describe('syncLatentSize', () => {
  it('moves the encoder latent layer and both boundary blocks', () => {
    const vae = createVaeNetwork(2);
    const next = syncLatentSize(vae, 8);
    expect(next.latentSize).toBe(8);
    expect(next.encoder.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [8] });
    expect(next.decoder.blocks[0]).toMatchObject({ kind: 'input', shape: [8] });
    expect(next.encoder.blocks.find((block) => block.id === 'latent')).toMatchObject({
      kind: 'linear',
      units: 8
    });
    const others = (net: { blocks: { id: string }[] }) =>
      net.blocks.filter((block) => block.id !== 'latent');
    expect(others(next.encoder).slice(0, -1)).toEqual(others(vae.encoder).slice(0, -1));
    expect(next.decoder.blocks.slice(1)).toEqual(vae.decoder.blocks.slice(1));
  });

  it('leaves both chains with no errors after a resize', () => {
    const next = syncLatentSize(createVaeNetwork(2), 8);
    const errors = (net: Parameters<typeof findProblems>[0]) =>
      findProblems(net).filter((problem) => problem.severity === 'error');
    expect(errors(next.encoder)).toEqual([]);
    expect(errors(next.decoder)).toEqual([]);
  });
});

describe('syncTraining', () => {
  it('mirrors the pair training config onto both chains', () => {
    const vae = createVaeNetwork(2);
    const next = syncTraining({ ...vae, training: { ...vae.training, learningRate: 0.05 } });
    expect(next.encoder.training.learningRate).toBe(0.05);
    expect(next.decoder.training.learningRate).toBe(0.05);
  });
});
