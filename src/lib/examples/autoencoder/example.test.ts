import { describe, expect, it } from 'vitest';
import { findProblems } from '../../network/problems';
import { PRESETS, presetFor } from './example';

describe('autoencoder presets', () => {
  it('offers a dense, a convolutional, and an upsampling preset', () => {
    expect(PRESETS.map((preset) => preset.id)).toEqual(['dense', 'conv', 'upsample']);
  });

  it.each(PRESETS.map((preset) => preset.id))('%s builds with no errors', (id) => {
    const preset = PRESETS.find((candidate) => candidate.id === id)!;
    expect(findProblems(preset.create(), { task: 'reconstruction' })).toEqual([]);
  });

  it('gives every preset a rank-one code layer for the scatter', () => {
    for (const preset of PRESETS) {
      const code = preset.create().blocks.find((block) => block.id === 'code');
      expect(code).toMatchObject({ kind: 'linear' });
      expect((code as { units: number }).units).toBeGreaterThanOrEqual(2);
    }
  });

  it('rebuilds with upsampling and convolution instead of transposed convolution', () => {
    const preset = PRESETS.find((candidate) => candidate.id === 'upsample')!;
    const kinds = preset.create().blocks.map((block) => block.kind);
    expect(kinds).toContain('upsampling2d');
    expect(kinds).not.toContain('conv2dtranspose');
  });
});

describe('presetFor', () => {
  it('recognises each preset by its blocks', () => {
    for (const preset of PRESETS) {
      expect(presetFor(preset.create())).toBe(preset.id);
    }
  });

  it('returns null for a network that matches no preset', () => {
    const network = PRESETS[0].create();
    expect(presetFor({ ...network, blocks: network.blocks.slice(0, -1) })).toBeNull();
  });

  it('returns null once a block field is edited', () => {
    const network = PRESETS[0].create();
    const blocks = network.blocks.map((block) =>
      block.kind === 'linear' ? { ...block, units: block.units + 1 } : block
    );
    expect(presetFor({ ...network, blocks })).toBeNull();
  });

  it('returns null once the training config is edited', () => {
    const network = PRESETS[1].create();
    const training = { ...network.training, learningRate: network.training.learningRate + 0.001 };
    expect(presetFor({ ...network, training })).toBeNull();
  });
});
