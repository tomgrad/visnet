import { NetworkStore } from '../../editor/networkStore.svelte';
import type { Block, TrainingConfig } from '../../network/types';
import { createVaeNetwork, vaeProblems } from '../../network/vae';

export class VaeController {
  readonly encoderStore: NetworkStore;
  readonly decoderStore: NetworkStore;

  constructor(latentSize = 2) {
    const vae = createVaeNetwork(latentSize);
    this.encoderStore = new NetworkStore(vae.encoder);
    this.decoderStore = new NetworkStore(vae.decoder);
  }

  get latentSize(): number {
    const output = this.encoderStore.network.blocks.find((block) => block.kind === 'output');
    return output && output.kind === 'output' ? (output.shape[0] ?? 0) : 0;
  }

  setLatentSize(size: number): void {
    const encoder = this.encoderStore.network;
    const latentIndex = encoder.blocks.reduce(
      (found, block, index) => (block.kind === 'linear' ? index : found),
      -1
    );
    if (latentIndex >= 0) {
      this.encoderStore.updateBlock(encoder.blocks[latentIndex].id, {
        units: size
      } as Partial<Block>);
    }
    const output = encoder.blocks.find((block) => block.kind === 'output');
    if (output) {
      this.encoderStore.updateBlock(output.id, { shape: [size] } as Partial<Block>);
    }
    const decoderInput = this.decoderStore.network.blocks.find((block) => block.kind === 'input');
    if (decoderInput) {
      this.decoderStore.updateBlock(decoderInput.id, { shape: [size] } as Partial<Block>);
    }
  }

  get training(): TrainingConfig {
    return this.encoderStore.network.training;
  }

  updateTraining(patch: Partial<TrainingConfig>): void {
    this.encoderStore.updateTraining(patch);
    this.decoderStore.updateTraining(patch);
  }

  get problems() {
    return vaeProblems({
      encoder: this.encoderStore.network,
      decoder: this.decoderStore.network,
      latentSize: this.latentSize,
      training: this.training
    });
  }

  get isValid(): boolean {
    return this.problems.every((problem) => problem.severity !== 'error');
  }
}
