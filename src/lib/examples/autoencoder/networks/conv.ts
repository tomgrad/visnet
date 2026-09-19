import type { Network } from '$lib/network/types';

export const CONV_AUTOENCODER_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1] },
    { id: 'conv-encode-1', kind: 'conv2d', filters: 16, kernelSize: 3, stride: 2, padding: 'same' },
    { id: 'relu-encode-1', kind: 'relu' },
    { id: 'conv-encode-2', kind: 'conv2d', filters: 32, kernelSize: 3, stride: 2, padding: 'same' },
    { id: 'relu-encode-2', kind: 'relu' },
    { id: 'flatten', kind: 'flatten' },
    { id: 'code', kind: 'linear', units: 32 },
    { id: 'expand', kind: 'linear', units: 7 * 7 * 32 },
    { id: 'reshape', kind: 'reshape', shape: [7, 7, 32] },
    {
      id: 'conv-decode-1',
      kind: 'conv2dtranspose',
      filters: 16,
      kernelSize: 3,
      stride: 2,
      padding: 'same'
    },
    { id: 'relu-decode-1', kind: 'relu' },
    {
      id: 'conv-decode-2',
      kind: 'conv2dtranspose',
      filters: 1,
      kernelSize: 3,
      stride: 2,
      padding: 'same'
    },
    { id: 'sigmoid', kind: 'sigmoid' },
    { id: 'output', kind: 'output', shape: [28, 28, 1] }
  ],
  training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
  positions: {}
} satisfies Network;
