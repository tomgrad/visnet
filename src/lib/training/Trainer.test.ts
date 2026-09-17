import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
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

function makeTrainer(
  onStats: (stats: TrainStats) => void,
  batchSize = 4,
  yieldFn?: () => Promise<void>
) {
  const model = buildModel(createEmptyNetwork());
  models.push(model);
  return new Trainer(model, makeData(), batchSize, onStats, yieldFn);
}

async function captureBatchSizes(count: number, batchSize: number): Promise<number[]> {
  const model = buildModel(createEmptyNetwork());
  models.push(model);
  const original = model.trainOnBatch.bind(model);
  const sizes: number[] = [];
  vi.spyOn(model, 'trainOnBatch').mockImplementation(async (x, y) => {
    sizes.push((x as tf.Tensor).shape[0]);
    return original(x, y);
  });
  const trainer = new Trainer(model, makeData(count), batchSize, () => {});
  for (let i = 0; i < trainer.batchesPerEpoch; i++) await trainer.step();
  return sizes;
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
    expect(stats[0]).toMatchObject({
      epoch: 0,
      batch: 1,
      epochMeanLoss: null,
      epochAccuracy: null
    });

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

describe('Trainer batch sampling', () => {
  it('covers each example exactly once per epoch', async () => {
    expect(await captureBatchSizes(5, 4)).toEqual([4, 1]);
    expect(await captureBatchSizes(8, 4)).toEqual([4, 4]);
    expect(await captureBatchSizes(5, 2)).toEqual([2, 2, 1]);
  });
});

describe('Trainer play loop', () => {
  it('runs steps until paused and resolves', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer(
      (s) => stats.push(s),
      4,
      async () => {
        yields += 1;
        if (yields >= 3) trainer.pause();
      }
    );

    expect(trainer.isPlaying).toBe(false);
    const loop = trainer.play();
    expect(trainer.isPlaying).toBe(true);
    await loop;

    expect(trainer.isPlaying).toBe(false);
    expect(stats).toHaveLength(3);
  });

  it('is a no-op when already playing', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer(
      (s) => stats.push(s),
      4,
      async () => {
        trainer.pause();
      }
    );

    const first = trainer.play();
    const second = trainer.play();
    await Promise.all([first, second]);

    expect(stats).toHaveLength(1);
  });

  it('stops immediately after dispose', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer(
      (s) => stats.push(s),
      4,
      async () => {
        yields += 1;
        if (yields >= 2) trainer.dispose();
      }
    );

    await trainer.play();
    const seen = stats.length;
    await trainer.step();
    expect(stats).toHaveLength(seen);
  });

  it('reports an error and stays playable when a step rejects', async () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    const original = model.trainOnBatch.bind(model);
    let calls = 0;
    vi.spyOn(model, 'trainOnBatch').mockImplementation(async (x, y) => {
      calls += 1;
      if (calls === 1) throw new Error('boom');
      return original(x, y);
    });

    const errors: unknown[] = [];
    let yields = 0;
    const trainer = new Trainer(
      model,
      makeData(),
      4,
      () => {},
      async () => {
        yields += 1;
        trainer.pause();
      },
      (error) => errors.push(error)
    );

    await expect(trainer.play()).resolves.toBeUndefined();
    expect(trainer.isPlaying).toBe(false);
    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toBe('boom');

    await expect(trainer.play()).resolves.toBeUndefined();
    expect(trainer.isPlaying).toBe(false);
    expect(yields).toBe(1);
  });

  it('reports a manual step failure through onError instead of rejecting', async () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    vi.spyOn(model, 'trainOnBatch').mockRejectedValue(new Error('boom'));

    const errors: unknown[] = [];
    const trainer = new Trainer(
      model,
      makeData(),
      4,
      () => {},
      async () => {},
      (error) => errors.push(error)
    );

    await expect(trainer.step()).resolves.toBeUndefined();
    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toBe('boom');
  });

  it('ignores a training result that arrives after dispose', async () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    let resolveTraining: (loss: number) => void = () => {};
    const pending = new Promise<number>((resolve) => {
      resolveTraining = resolve;
    });
    vi.spyOn(model, 'trainOnBatch').mockImplementation(() => pending);
    const predict = vi.spyOn(model, 'predict');

    const stats: TrainStats[] = [];
    const trainer = new Trainer(model, makeData(), 8, (next) => stats.push(next));

    const step = trainer.step();
    trainer.dispose();
    resolveTraining(0.5);

    await expect(step).resolves.toBeUndefined();
    expect(stats).toHaveLength(0);
    expect(predict).not.toHaveBeenCalled();
  });
});

describe('a four-dimensional feature batch', () => {
  it('trains on image-shaped inputs', async () => {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [2, 2, 1] }));
    model.add(tf.layers.dense({ units: 2 }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    const xs = tf.tensor4d([0, 1, 1, 0, 1, 0, 0, 1], [2, 2, 2, 1]);
    const ys = tf.tensor2d(
      [
        [1, 0],
        [0, 1]
      ],
      [2, 2]
    );
    tensors.push(xs, ys);

    const stats: TrainStats[] = [];
    const trainer = new Trainer(model, { xs, ys }, 2, (s) => stats.push(s));

    await trainer.step();

    expect(stats).toHaveLength(1);
    expect(Number.isFinite(stats[0].batchLoss)).toBe(true);
  });
});
