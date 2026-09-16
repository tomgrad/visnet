import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { Trainer, type TrainStats } from './Trainer';

let models: tf.Sequential[] = [];
let tensors: tf.Tensor[] = [];

function makeData(count = 8): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < count; i++) {
    const positive = i % 2 === 0;
    xs.push([positive ? 0.5 : -0.5, positive ? 0.5 : -0.5]);
    ys.push(positive ? [0, 1] : [1, 0]);
  }
  const x = tf.tensor2d(xs, [count, 2]);
  const y = tf.tensor2d(ys, [count, 2]);
  tensors.push(x, y);
  return { xs: x, ys: y };
}

function makeTrainer(onStats: (stats: TrainStats) => void, batchSize = 4, yieldFn?: () => Promise<void>) {
  const model = buildModel(createEmptyNetwork());
  models.push(model);
  return new Trainer(model, makeData(), batchSize, onStats, yieldFn);
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  tensors.forEach((tensor) => tensor.dispose());
  models = [];
  tensors = [];
});

describe('Trainer bookkeeping', () => {
  it('computes batches per epoch', () => {
    expect(makeTrainer(() => {}).batchesPerEpoch).toBe(2);
  });

  it('reports a finite loss for every step', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s));

    await trainer.step();
    await trainer.step();

    expect(stats).toHaveLength(2);
    expect(stats.every((s) => Number.isFinite(s.batchLoss))).toBe(true);
  });

  it('reports null epoch statistics mid-epoch and real ones at the end', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s));

    await trainer.step();
    expect(stats[0]).toMatchObject({ epoch: 0, batch: 1, epochMeanLoss: null, epochAccuracy: null });

    await trainer.step();
    expect(stats[1].epoch).toBe(1);
    expect(stats[1].batch).toBe(0);
    expect(stats[1].epochMeanLoss).toBeGreaterThan(0);
    expect(stats[1].epochAccuracy).toBeGreaterThanOrEqual(0);
    expect(stats[1].epochAccuracy).toBeLessThanOrEqual(1);
  });

  it('advances the epoch every batchesPerEpoch steps', async () => {
    const epochs: number[] = [];
    const trainer = makeTrainer((s) => epochs.push(s.epoch));
    expect(trainer.batchesPerEpoch).toBe(2);
    for (let i = 0; i < 6; i++) await trainer.step();
    expect(epochs).toEqual([0, 1, 1, 2, 2, 3]);
  });

  it('never requests a batch larger than the dataset', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s), 100);
    expect(trainer.batchesPerEpoch).toBe(1);
    await trainer.step();
    expect(Number.isFinite(stats[0].batchLoss)).toBe(true);
  });
});

describe('Trainer play loop', () => {
  it('runs steps until paused and resolves', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      yields += 1;
      if (yields >= 3) trainer.pause();
    });

    expect(trainer.isPlaying).toBe(false);
    const loop = trainer.play();
    expect(trainer.isPlaying).toBe(true);
    await loop;

    expect(trainer.isPlaying).toBe(false);
    expect(stats).toHaveLength(3);
  });

  it('is a no-op when already playing', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      trainer.pause();
    });

    const first = trainer.play();
    const second = trainer.play();
    await Promise.all([first, second]);

    expect(stats).toHaveLength(1);
  });

  it('stops immediately after dispose', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      yields += 1;
      if (yields >= 2) trainer.dispose();
    });

    await trainer.play();
    const seen = stats.length;
    await trainer.step();
    expect(stats).toHaveLength(seen);
  });
});
