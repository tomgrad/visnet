import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createVaeNetwork } from '../network/vae';
import { buildVaeModels } from '../tf/vae';
import type { TrainStats } from './Trainer';
import { VaeTrainer } from './VaeTrainer';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

let dispose: Array<() => void> = [];
afterEach(() => {
  dispose.forEach((fn) => fn());
  dispose = [];
});

function tinySetup() {
  const models = buildVaeModels(createVaeNetwork(2));
  const xs = tf.randomUniform([8, 64, 64, 3]);
  dispose.push(() => {
    models.encoder.dispose();
    models.decoder.dispose();
    xs.dispose();
  });
  return { models, xs };
}

describe('VaeTrainer', () => {
  it('reduces the combined loss and updates both models', async () => {
    const { models, xs } = tinySetup();
    const decoderBefore = models.decoder.getWeights().map((tensor) => tensor.dataSync()[0]);
    const encoderBefore = models.encoder.getWeights().map((tensor) => tensor.dataSync()[0]);
    const losses: number[] = [];
    const trainer = new VaeTrainer(
      models,
      { xs },
      { loss: 'mse', optimizer: 'adam', learningRate: 0.01, batchSize: 4 },
      2,
      (stats: TrainStats) => {
        if (stats.epochMeanLoss !== null) losses.push(stats.epochMeanLoss);
      },
      async () => {}
    );
    for (let step = 0; step < 6; step++) await trainer.step();
    trainer.dispose();

    expect(losses.length).toBeGreaterThan(0);
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    const decoderAfter = models.decoder.getWeights().map((tensor) => tensor.dataSync()[0]);
    expect(decoderAfter).not.toEqual(decoderBefore);
    const encoderAfter = models.encoder.getWeights().map((tensor) => tensor.dataSync()[0]);
    expect(encoderAfter).not.toEqual(encoderBefore);
  }, 120_000);

  it('reports no accuracy', async () => {
    const { models, xs } = tinySetup();
    const seen: TrainStats[] = [];
    const trainer = new VaeTrainer(
      models,
      { xs },
      { loss: 'mse', optimizer: 'adam', learningRate: 0.01, batchSize: 4 },
      2,
      (stats) => seen.push(stats),
      async () => {}
    );
    for (let step = 0; step < 4; step++) await trainer.step();
    trainer.dispose();
    expect(seen.some((stats) => stats.epochMeanLoss !== null)).toBe(true);
    expect(seen.every((stats) => stats.epochAccuracy === null)).toBe(true);
  }, 120_000);
});
