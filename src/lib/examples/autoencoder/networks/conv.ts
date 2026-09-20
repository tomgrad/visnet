import type { Network } from '$lib/network/types';

export const CONV_AUTOENCODER_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1], colour: 'amber' },
    { id: 'conv-encode-1', kind: 'conv2d', filters: 16, kernelSize: 3, stride: 2, padding: 'same' },
    { id: 'relu-encode-1', kind: 'relu' },
    { id: 'conv-encode-2', kind: 'conv2d', filters: 32, kernelSize: 3, stride: 2, padding: 'same' },
    { id: 'relu-encode-2', kind: 'relu' },
    { id: 'flatten', kind: 'flatten' },
    { id: 'code', kind: 'linear', units: 32, colour: 'green' },
    { id: 'expand', kind: 'linear', units: 1568 },
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
    { id: 'output', kind: 'output', shape: [28, 28, 1], colour: 'amber' }
  ],
  training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
  positions: {
    input: { x: 0, y: 0 },
    'conv-encode-1': { x: 0, y: 170 },
    'relu-encode-1': { x: 0, y: 340 },
    'conv-encode-2': { x: 0, y: 510 },
    'relu-encode-2': { x: 0, y: 680 },
    flatten: { x: 0, y: 850 },
    code: { x: 0, y: 1020 },
    expand: { x: 364, y: 4.570897102355957 },
    reshape: { x: 362, y: 174 },
    'conv-decode-1': { x: 352, y: 342 },
    'relu-decode-1': { x: 352, y: 510 },
    'conv-decode-2': { x: 346, y: 670 },
    sigmoid: { x: 350, y: 841.2509546279907 },
    output: { x: 344, y: 1012.2690925598145 }
  }
} satisfies Network;
