import type { Network } from '$lib/network/types';

export const UPSAMPLE_AUTOENCODER_NETWORK = {
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
      filters: 1,
      kernelSize: 3,
      stride: 1,
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
    expand: { x: 370, y: 0 },
    reshape: { x: 370, y: 170 },
    'upsample-1': { x: 370, y: 340 },
    'conv-decode-1': { x: 370, y: 510 },
    'relu-decode-1': { x: 370, y: 680 },
    'upsample-2': { x: 370, y: 850 },
    'conv-decode-2': { x: 370, y: 1020 },
    sigmoid: { x: 370, y: 1190 },
    output: { x: 370, y: 1360 }
  }
} satisfies Network;
