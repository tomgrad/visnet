import * as tf from '@tensorflow/tfjs';
import type { LatentBounds } from './latent';
import { forwardActivations } from './activations';

export interface CodeSample {
  points: Float32Array;
  bounds: LatentBounds;
}

function boundsOf(points: Float32Array): LatentBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < points.length; index += 2) {
    const x = points[index];
    const y = points[index + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (![minX, maxX, minY, maxY].every(Number.isFinite))
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

export function codeScatter(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  count: number,
  source: 'input' | number,
  dimA: number
): CodeSample {
  const points = tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const inputDims = (model.inputs[0].shape ?? []).slice(1);
    const input =
      inputDims.length === 1
        ? tf.tensor2d(normalised, [count, rows * cols])
        : tf.tensor4d(normalised, [count, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    const activation = source === 'input' ? input : activations[source];
    const a = activation.slice([0, dimA], [count, 1]);
    const b = activation.slice([0, dimA + 1], [count, 1]);
    return Float32Array.from(tf.concat([a, b], 1).dataSync());
  });
  return { points, bounds: boundsOf(points) };
}
