import type { Network } from '../../../network/types';

export const CONV_AUTOENCODER_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1] },
    { id: 'conv-encode', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'relu-encode', kind: 'relu' },
    { id: 'pool-encode', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
    { id: 'conv-deeper', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'relu-deeper', kind: 'relu' },
    { id: 'pool-deeper', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
    { id: 'flatten', kind: 'flatten' },
    { id: 'code', kind: 'linear', units: 2 },
    { id: 'code-tanh', kind: 'tanh' },
    { id: 'expand', kind: 'linear', units: 392 },
    { id: 'expand-relu', kind: 'relu' },
    { id: 'reshape', kind: 'reshape', shape: [7, 7, 8] },
    { id: 'upsample-1', kind: 'upsampling2d', size: 2 },
    { id: 'conv-decode', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'relu-decode', kind: 'relu' },
    { id: 'upsample-2', kind: 'upsampling2d', size: 2 },
    { id: 'conv-final', kind: 'conv2d', filters: 1, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'sigmoid', kind: 'sigmoid' },
    { id: 'output', kind: 'output', shape: [28, 28, 1] }
  ],
  training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
  positions: {}
} satisfies Network;
