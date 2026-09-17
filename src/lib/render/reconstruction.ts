import * as tf from '@tensorflow/tfjs';
import { forwardActivations } from './activations';

export function reconstruct(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  count: number
): Uint8ClampedArray {
  const output = tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const input = tf.tensor4d(normalised, [count, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    return activations[activations.length - 1].dataSync();
  });
  const result = new Uint8ClampedArray(count * rows * cols);
  for (let index = 0; index < result.length; index++) {
    result[index] = Math.round(Math.max(0, Math.min(1, output[index])) * 255);
  }
  return result;
}
