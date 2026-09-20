import { cloneNetwork } from '../../network/factory';
import type { Block, Network } from '../../network/types';
import type { StorageKeys } from '../../persist/storage';
import { CONV_AUTOENCODER_NETWORK } from './networks/conv';
import { DENSE_AUTOENCODER_NETWORK } from './networks/dense';
import { UPSAMPLE_AUTOENCODER_NETWORK } from './networks/upsample';

export const AUTOENCODER_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:autoencoder:network:v1',
  dataset: 'visnet:autoencoder:dataset:v1'
};

export const AUTOENCODER_WEIGHTS_ID = 'autoencoder';

export const TRAIN_COUNT = 1000;
export const SCATTER_COUNT = 200;
export const RECONSTRUCTION_COUNT = 30;

export type AutoencoderPresetId = 'dense' | 'conv' | 'upsample';

export interface AutoencoderPreset {
  id: AutoencoderPresetId;
  label: string;
  create(): Network;
}

export const PRESETS: AutoencoderPreset[] = [
  {
    id: 'dense',
    label: 'Dense (2-neuron code)',
    create: () => cloneNetwork(DENSE_AUTOENCODER_NETWORK)
  },
  {
    id: 'conv',
    label: 'Convolutional (32-number code)',
    create: () => cloneNetwork(CONV_AUTOENCODER_NETWORK)
  },
  {
    id: 'upsample',
    label: 'Upsampling (32-number code)',
    create: () => cloneNetwork(UPSAMPLE_AUTOENCODER_NETWORK)
  }
];

function blockSignature(block: Block): unknown {
  switch (block.kind) {
    case 'linear':
      return [block.kind, block.units];
    case 'conv2d':
    case 'conv2dtranspose':
      return [block.kind, block.filters, block.kernelSize, block.stride, block.padding];
    case 'maxpool2d':
      return [block.kind, block.poolSize, block.stride, block.padding];
    case 'upsampling2d':
      return [block.kind, block.size];
    case 'input':
    case 'reshape':
    case 'output':
      return [block.kind, ...block.shape];
    default:
      return [block.kind];
  }
}

function networkSignature(network: Network): string {
  return JSON.stringify({
    blocks: network.blocks.map(blockSignature),
    training: {
      loss: network.training.loss,
      optimizer: network.training.optimizer,
      learningRate: network.training.learningRate,
      batchSize: network.training.batchSize
    }
  });
}

const PRESET_SIGNATURES = new Map<AutoencoderPresetId, string>(
  PRESETS.map((preset) => [preset.id, networkSignature(preset.create())])
);

export function presetFor(network: Network): AutoencoderPresetId | null {
  const signature = networkSignature(network);
  for (const [id, presetSignature] of PRESET_SIGNATURES) {
    if (presetSignature === signature) {
      return id;
    }
  }
  return null;
}
