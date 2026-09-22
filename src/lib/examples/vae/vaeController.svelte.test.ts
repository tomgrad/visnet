import { describe, expect, it } from 'vitest';
import { VaeController } from './vaeController.svelte';

describe('VaeController', () => {
  it('starts from the default VAE and keeps the boundary blocks in sync', () => {
    const controller = new VaeController();
    expect(controller.latentSize).toBe(2);
    controller.setLatentSize(8);
    expect(controller.encoderStore.network.blocks.at(-1)).toMatchObject({ shape: [8] });
    expect(controller.decoderStore.network.blocks[0]).toMatchObject({ shape: [8] });
    expect(controller.isValid).toBe(true);
  });

  it('reports invalid when a chain is broken', () => {
    const controller = new VaeController();
    controller.encoderStore.updateBlock(controller.encoderStore.network.blocks[0].id, {
      shape: [2]
    });
    expect(controller.isValid).toBe(false);
  });
});
