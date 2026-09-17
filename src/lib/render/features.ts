import * as tf from '@tensorflow/tfjs';
import { forwardActivations } from './activations';

export interface FeatureMap {
  width: number;
  height: number;
  values: Uint8ClampedArray;
}

function shade(value: number, min: number, max: number): number {
  if (max === min) return 128;
  return Math.round(((value - min) / (max - min)) * 255);
}

function extent(data: ArrayLike<number>, start: number, count: number): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let index = 0; index < count; index++) {
    const value = data[start + index];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return [min, max];
}

export function featureMaps(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  source: 'input' | number
): FeatureMap[] {
  let maps: FeatureMap[] = [];
  tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const input = tf.tensor4d(normalised, [1, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    if (source !== 'input' && (source < 0 || source >= activations.length)) return;
    const activation = source === 'input' ? input : activations[source];
    const dims = activation.shape.slice(1);
    const data = activation.dataSync();

    if (dims.length === 3) {
      const [height, width, channels] = dims;
      const plane = height * width;
      const result: FeatureMap[] = [];
      for (let channel = 0; channel < channels; channel++) {
        const start = channel * plane;
        const [min, max] = extent(data, start, plane);
        const values = new Uint8ClampedArray(plane);
        for (let index = 0; index < plane; index++) {
          values[index] = shade(data[start + index], min, max);
        }
        result.push({ width, height, values });
      }
      maps = result;
      return;
    }

    if (dims.length === 1) {
      const count = dims[0];
      const [min, max] = extent(data, 0, count);
      const result: FeatureMap[] = [];
      for (let index = 0; index < count; index++) {
        result.push({ width: 1, height: 1, values: Uint8ClampedArray.of(shade(data[index], min, max)) });
      }
      maps = result;
      return;
    }

    maps = [];
  });
  return maps;
}
