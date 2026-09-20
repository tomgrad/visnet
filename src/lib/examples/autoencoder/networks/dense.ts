import type { Network } from '$lib/network/types';

export const DENSE_AUTOENCODER_NETWORK = {
  blocks: [
    { id: 'input', kind: 'input', shape: [28, 28, 1], colour: 'amber' },
    { id: 'flatten', kind: 'flatten', colour: 'blue' },
    { id: 'encoder', kind: 'linear', units: 50 },
    { id: '960a5673-1890-40f3-b112-502f6f32ec64', kind: 'tanh' },
    { id: 'code', kind: 'linear', units: 50 },
    { id: 'code-tanh', kind: 'tanh' },
    { id: 'decoder', kind: 'linear', units: 2, colour: 'green' },
    { id: '7a348aab-a269-489e-bd67-113db720a01c', kind: 'linear', units: 50 },
    { id: '20ec8161-42ce-4cfa-b28e-3622a9666368', kind: 'tanh' },
    { id: '88510fca-77ab-4e96-bfe3-eff97c1a4ab5', kind: 'linear', units: 50 },
    { id: '398a041b-7693-4444-b9d5-ee35197d55e2', kind: 'tanh' },
    { id: 'pixels', kind: 'linear', units: 784 },
    { id: '411c069b-f403-43be-9a31-771b0f77d1d6', kind: 'sigmoid' },
    { id: 'reshape', kind: 'reshape', shape: [28, 28, 1], colour: 'blue' },
    { id: 'output', kind: 'output', shape: [28, 28, 1], colour: 'amber' }
  ],
  training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
  positions: {
    input: { x: 0, y: 0 },
    flatten: { x: 0, y: 170 },
    encoder: { x: 0, y: 340 },
    code: { x: 0, y: 680 },
    'code-tanh': { x: 0, y: 850 },
    decoder: { x: 0, y: 1020 },
    pixels: { x: 357.91099533307, y: 597.9447026137701 },
    reshape: { x: 360.08900466693, y: 872.0418973275398 },
    output: { x: 358.50396728515625, y: 1004.5162286758423 },
    '960a5673-1890-40f3-b112-502f6f32ec64': { x: 6.678150539678768, y: 502.44452818250375 },
    '7a348aab-a269-489e-bd67-113db720a01c': { x: 351.182117824835, y: -0.2882885136632467 },
    '20ec8161-42ce-4cfa-b28e-3622a9666368': { x: 354.915103824045, y: 160.28763681545658 },
    '88510fca-77ab-4e96-bfe3-eff97c1a4ab5': { x: 351.004108490975, y: 299.13057614536643 },
    '398a041b-7693-4444-b9d5-ee35197d55e2': { x: 351.44913182562505, y: 444.32953414299635 },
    '411c069b-f403-43be-9a31-771b0f77d1d6': { x: 357.09424796214785, y: 734.8077508682035 }
  }
} satisfies Network;
