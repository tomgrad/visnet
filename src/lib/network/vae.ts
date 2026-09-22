import type { Block, Network, TrainingConfig } from './types';

export interface VaeNetwork {
  encoder: Network;
  decoder: Network;
  latentSize: number;
  training: TrainingConfig;
}

const DEFAULT_TRAINING: TrainingConfig = {
  loss: 'mse',
  optimizer: 'adam',
  learningRate: 0.001,
  batchSize: 32
};

export function createVaeNetwork(latentSize = 2): VaeNetwork {
  const encoder: Network = {
    blocks: [
      { id: 'input', kind: 'input', shape: [64, 64, 3] },
      {
        id: 'conv-encode-1',
        kind: 'conv2d',
        filters: 16,
        kernelSize: 3,
        stride: 2,
        padding: 'same'
      },
      { id: 'relu-encode-1', kind: 'relu' },
      {
        id: 'conv-encode-2',
        kind: 'conv2d',
        filters: 32,
        kernelSize: 3,
        stride: 2,
        padding: 'same'
      },
      { id: 'relu-encode-2', kind: 'relu' },
      { id: 'flatten', kind: 'flatten' },
      { id: 'latent', kind: 'linear', units: latentSize },
      { id: 'latent-output', kind: 'output', shape: [latentSize] }
    ],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
  const decoder: Network = {
    blocks: [
      { id: 'latent-input', kind: 'input', shape: [latentSize] },
      { id: 'expand', kind: 'linear', units: 16 * 16 * 32 },
      { id: 'reshape', kind: 'reshape', shape: [16, 16, 32] },
      { id: 'upsample-1', kind: 'upsampling2d', size: 2 },
      {
        id: 'conv-decode-1',
        kind: 'conv2d',
        filters: 16,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      },
      { id: 'relu-decode-1', kind: 'relu' },
      { id: 'upsample-2', kind: 'upsampling2d', size: 2 },
      {
        id: 'conv-decode-2',
        kind: 'conv2d',
        filters: 3,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      },
      { id: 'sigmoid', kind: 'sigmoid' },
      { id: 'output', kind: 'output', shape: [64, 64, 3] }
    ],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
  return { encoder, decoder, latentSize, training: { ...DEFAULT_TRAINING } };
}

export function syncLatentSize(vae: VaeNetwork, latentSize: number): VaeNetwork {
  const latentIndex = vae.encoder.blocks.reduce(
    (found, block, index) => (block.kind === 'linear' ? index : found),
    -1
  );
  return {
    ...vae,
    latentSize,
    encoder: {
      ...vae.encoder,
      blocks: vae.encoder.blocks.map((block, index) => {
        if (index === latentIndex) return { ...block, units: latentSize } as Block;
        if (block.kind === 'output') return { ...block, shape: [latentSize] };
        return block;
      })
    },
    decoder: {
      ...vae.decoder,
      blocks: vae.decoder.blocks.map((block) =>
        block.kind === 'input' ? { ...block, shape: [latentSize] } : block
      )
    }
  };
}

export function syncTraining(vae: VaeNetwork): VaeNetwork {
  return {
    ...vae,
    encoder: { ...vae.encoder, training: { ...vae.training } },
    decoder: { ...vae.decoder, training: { ...vae.training } }
  };
}
