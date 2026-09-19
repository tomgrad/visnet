import type { Network } from '$lib/network/types';

export const CNN_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1] },
    { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'relu', kind: 'relu' },
    { id: 'flatten', kind: 'flatten' },
    { id: 'dense', kind: 'linear', units: 10 },
    { id: 'softmax', kind: 'softmax' },
    { id: 'output', kind: 'output', shape: [10] }
  ],
  training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
  positions: {}
} satisfies Network;
