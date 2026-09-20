import type { Network } from '$lib/network/types';

export const MLP_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [2] },
    { id: 'hidden', kind: 'linear', units: 8 },
    { id: 'relu', kind: 'relu' },
    { id: 'output-layer', kind: 'linear', units: 2 },
    { id: 'softmax', kind: 'softmax' },
    { id: 'output', kind: 'output', shape: [2] }
  ],
  training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
  positions: {}
} satisfies Network;
