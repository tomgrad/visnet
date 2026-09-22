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
    const before = models.decoder.getWeights().map((tensor) => tensor.dataSync()[0]);
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
    const after = models.decoder.getWeights().map((tensor) => tensor.dataSync()[0]);
    expect(after).not.toEqual(before);
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
    await trainer.step();
    trainer.dispose();
    expect(seen.every((stats) => stats.epochAccuracy === null)).toBe(true);
  }, 120_000);
});
