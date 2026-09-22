import * as tf from '@tensorflow/tfjs';
import type { VaeNetwork } from '../network/vae';
import { buildModel } from './buildModel';

export interface VaeModels {
  encoder: tf.Sequential;
  decoder: tf.Sequential;
}

export function buildVaeModels(vae: VaeNetwork): VaeModels {
  const head = tf.layers.dense({ units: 2 * vae.latentSize });
  const encoder = buildModel(vae.encoder, [head]);
  const decoder = buildModel(vae.decoder);
  return { encoder, decoder };
}

export function splitLatent(output: tf.Tensor, latentSize: number): [tf.Tensor, tf.Tensor] {
  const mean = output.slice([0, 0], [-1, latentSize]);
  const logVariance = output.slice([0, latentSize], [-1, latentSize]);
  return [mean, logVariance];
}

export function sampleLatent(
  mean: tf.Tensor,
  logVariance: tf.Tensor,
  random: () => tf.Tensor = () => tf.randomNormal(mean.shape)
): tf.Tensor {
  return tf.tidy(() => {
    const epsilon = random();
    return tf.add(mean, tf.mul(tf.exp(tf.mul(0.5, logVariance)), epsilon));
  });
}

export function klDivergence(mean: tf.Tensor, logVariance: tf.Tensor): tf.Scalar {
  return tf.tidy(() => {
    const inner = tf.sub(tf.add(tf.exp(logVariance), tf.square(mean)), tf.add(1, logVariance));
    return tf.mean(tf.mul(0.5, tf.sum(inner, 1)));
  });
}

export function reconstruct(models: VaeModels, xs: tf.Tensor, latentSize: number): tf.Tensor {
  return tf.tidy(() => {
    const encoded = models.encoder.apply(xs) as tf.Tensor;
    const [mean] = splitLatent(encoded, latentSize);
    return models.decoder.apply(mean) as tf.Tensor;
  });
}

export function generate(models: VaeModels, z: tf.Tensor): tf.Tensor {
  return tf.tidy(() => models.decoder.apply(z) as tf.Tensor);
}
