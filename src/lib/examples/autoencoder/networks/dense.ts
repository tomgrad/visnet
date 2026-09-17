import type { Network } from '../../../network/types';

export const DENSE_AUTOENCODER_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1] },
    { id: 'flatten', kind: 'flatten' },
    { id: 'encoder', kind: 'linear', units: 32 },
    { id: 'encoder-relu', kind: 'relu' },
    { id: 'code', kind: 'linear', units: 2 },
    { id: 'code-tanh', kind: 'tanh' },
    { id: 'decoder', kind: 'linear', units: 32 },
    { id: 'decoder-relu', kind: 'relu' },
    { id: 'pixels', kind: 'linear', units: 784 },
    { id: 'pixels-sigmoid', kind: 'sigmoid' },
    { id: 'reshape', kind: 'reshape', shape: [28, 28, 1] },
    { id: 'output', kind: 'output', shape: [28, 28, 1] }
  ],
  training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
  positions: {}
} satisfies Network;
