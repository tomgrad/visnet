import { createBlock } from '../../network/factory';
import type {
  BlockKind,
  Conv2dBlock,
  InputBlock,
  LinearBlock,
  Network,
  OutputBlock,
  ReshapeBlock
} from '../../network/types';
import type { StorageKeys } from '../../persist/storage';

export const AUTOENCODER_PALETTE: BlockKind[] = [
  'conv2d',
  'maxpool2d',
  'upsampling2d',
  'flatten',
  'reshape',
  'linear',
  'relu',
  'sigmoid',
  'tanh',
  'softmax'
];

export const AUTOENCODER_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:autoencoder:network:v1',
  dataset: 'visnet:autoencoder:dataset:v1'
};

export const AUTOENCODER_WEIGHTS_ID = 'autoencoder';

export const TRAIN_COUNT = 5000;
export const SCATTER_COUNT = 500;
export const RECONSTRUCTION_COUNT = 40;

export type AutoencoderPresetId = 'dense' | 'conv';

export interface AutoencoderPreset {
  id: AutoencoderPresetId;
  label: string;
  create(): Network;
}

const TRAINING = {
  loss: 'mse' as const,
  optimizer: 'adam' as const,
  learningRate: 0.001,
  batchSize: 32
};

function createDenseNetwork(): Network {
  const input: InputBlock = { ...(createBlock('input') as InputBlock), shape: [28, 28, 1] };
  const flatten = createBlock('flatten');
  const encoder: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 32 };
  const encodeRelu = createBlock('relu');
  const code: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 2 };
  const codeTanh = createBlock('tanh');
  const decoder: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 32 };
  const decodeRelu = createBlock('relu');
  const pixels: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 784 };
  const pixelsSigmoid = createBlock('sigmoid');
  const reshape: ReshapeBlock = { ...(createBlock('reshape') as ReshapeBlock), shape: [28, 28, 1] };
  const output: OutputBlock = { ...(createBlock('output') as OutputBlock), shape: [28, 28, 1] };

  return {
    version: 2,
    blocks: [
      input,
      flatten,
      encoder,
      encodeRelu,
      code,
      codeTanh,
      decoder,
      decodeRelu,
      pixels,
      pixelsSigmoid,
      reshape,
      output
    ],
    training: { ...TRAINING },
    positions: {}
  };
}

function createConvNetwork(): Network {
  const input: InputBlock = { ...(createBlock('input') as InputBlock), shape: [28, 28, 1] };
  const encodeConv = createBlock('conv2d');
  const encodeRelu = createBlock('relu');
  const encodePool = createBlock('maxpool2d');
  const deeperConv = createBlock('conv2d');
  const deeperRelu = createBlock('relu');
  const deeperPool = createBlock('maxpool2d');
  const firstUpsample = createBlock('upsampling2d');
  const decodeConv = createBlock('conv2d');
  const decodeRelu = createBlock('relu');
  const secondUpsample = createBlock('upsampling2d');
  const finalConv: Conv2dBlock = { ...(createBlock('conv2d') as Conv2dBlock), filters: 1 };
  const finalSigmoid = createBlock('sigmoid');
  const output: OutputBlock = { ...(createBlock('output') as OutputBlock), shape: [28, 28, 1] };

  return {
    version: 2,
    blocks: [
      input,
      encodeConv,
      encodeRelu,
      encodePool,
      deeperConv,
      deeperRelu,
      deeperPool,
      firstUpsample,
      decodeConv,
      decodeRelu,
      secondUpsample,
      finalConv,
      finalSigmoid,
      output
    ],
    training: { ...TRAINING },
    positions: {}
  };
}

export const PRESETS: AutoencoderPreset[] = [
  { id: 'dense', label: 'Dense (2-neuron code)', create: createDenseNetwork },
  { id: 'conv', label: 'Convolutional', create: createConvNetwork }
];

const PRESET_KINDS = new Map<AutoencoderPresetId, BlockKind[]>(
  PRESETS.map((preset) => [preset.id, preset.create().blocks.map((block) => block.kind)])
);

export function presetFor(network: Network): AutoencoderPresetId | null {
  const kinds = network.blocks.map((block) => block.kind);
  for (const [id, presetKinds] of PRESET_KINDS) {
    if (presetKinds.length === kinds.length && presetKinds.every((kind, i) => kind === kinds[i])) {
      return id;
    }
  }
  return null;
}
