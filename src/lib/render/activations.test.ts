import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { forwardActivations } from './activations';

let models: tf.Sequential[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('forwardActivations', () => {
  it('returns one activation per layer, matching predict', () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    const input = tf.tensor2d([[0.1, -0.2]]);

    const { count, last } = tf.tidy(() => {
      const activations = forwardActivations(model, input);
      return {
        count: activations.length,
        last: Array.from(activations[activations.length - 1].dataSync())
      };
    });
    input.dispose();

    const predicted = tf.tidy(() =>
      Array.from((model.predict(tf.tensor2d([[0.1, -0.2]])) as tf.Tensor).dataSync())
    );
    expect(count).toBe(model.layers.length);
    expect(last).toEqual(predicted);
  });
});
