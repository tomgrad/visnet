import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createVaeNetwork } from '../network/vae';
import {
  buildVaeModels,
  generate,
  klDivergence,
  reconstruct,
  sampleLatent,
  splitLatent
} from './vae';

let models: { encoder: tf.Sequential; decoder: tf.Sequential }[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach(({ encoder, decoder }) => {
    encoder.dispose();
    decoder.dispose();
  });
  models = [];
});

describe('buildVaeModels', () => {
  it('gives the encoder a two-times-latent output and the decoder a full image output', () => {
    const built = buildVaeModels(createVaeNetwork(2));
    models.push(built);
    expect(built.encoder.outputs[0].shape).toEqual([null, 4]);
    expect(built.decoder.inputs[0].shape).toEqual([null, 2]);
    expect(built.decoder.outputs[0].shape).toEqual([null, 64, 64, 3]);
  });

  it('scales the encoder head with the latent size', () => {
    const built = buildVaeModels(createVaeNetwork(8));
    models.push(built);
    expect(built.encoder.outputs[0].shape).toEqual([null, 16]);
    expect(built.decoder.inputs[0].shape).toEqual([null, 8]);
  });
});

describe('splitLatent', () => {
  it('splits the head into a mean and a log variance', () => {
    const output = tf.tensor2d([[1, 2, 3, 4]]);
    const [mean, logVariance] = splitLatent(output, 2);
    expect(mean.shape).toEqual([1, 2]);
    expect(logVariance.shape).toEqual([1, 2]);
    expect(Array.from(mean.dataSync())).toEqual([1, 2]);
    expect(Array.from(logVariance.dataSync())).toEqual([3, 4]);
    output.dispose();
    mean.dispose();
    logVariance.dispose();
  });
});

describe('sampleLatent', () => {
  it('returns the mean when the draw is zero', () => {
    const mean = tf.tensor2d([[1, 2]]);
    const logVariance = tf.tensor2d([[0, 0]]);
    const z = sampleLatent(mean, logVariance, () => tf.zeros([1, 2]));
    expect(Array.from(z.dataSync())).toEqual([1, 2]);
    mean.dispose();
    logVariance.dispose();
    z.dispose();
  });

  it('scales the draw by the standard deviation', () => {
    const mean = tf.tensor2d([[0, 0]]);
    const logVariance = tf.tensor2d([[Math.log(4), Math.log(4)]]);
    const z = sampleLatent(mean, logVariance, () => tf.ones([1, 2]));
    expect(Array.from(z.dataSync())).toEqual([2, 2]);
    mean.dispose();
    logVariance.dispose();
    z.dispose();
  });
});

describe('klDivergence', () => {
  it('is zero for a standard normal', () => {
    const mean = tf.zeros([3, 4]);
    const logVariance = tf.zeros([3, 4]);
    const kl = klDivergence(mean, logVariance);
    expect(kl.dataSync()[0]).toBe(0);
    mean.dispose();
    logVariance.dispose();
    kl.dispose();
  });

  it('matches a hand-computed value', () => {
    const mean = tf.tensor2d([[1, 0]]);
    const logVariance = tf.tensor2d([[0, 0]]);
    const kl = klDivergence(mean, logVariance);
    expect(kl.dataSync()[0]).toBeCloseTo(0.5, 6);
    mean.dispose();
    logVariance.dispose();
    kl.dispose();
  });
});

describe('reconstruct and generate', () => {
  it('reconstructs an image batch through the mean', () => {
    const built = buildVaeModels(createVaeNetwork(2));
    models.push(built);
    const xs = tf.zeros([2, 64, 64, 3]);
    const out = reconstruct(built, xs, 2);
    expect(out.shape).toEqual([2, 64, 64, 3]);
    xs.dispose();
    out.dispose();
  });

  it('generates from a latent sample', () => {
    const built = buildVaeModels(createVaeNetwork(2));
    models.push(built);
    const z = tf.zeros([3, 2]);
    const out = generate(built, z);
    expect(out.shape).toEqual([3, 64, 64, 3]);
    z.dispose();
    out.dispose();
  });
});
