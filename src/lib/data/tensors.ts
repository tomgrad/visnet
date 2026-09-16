import * as tf from '@tensorflow/tfjs';
import { imageAt, type ImageDataset } from './images';
import type { PointDataset } from './points';

export function toTensors(dataset: PointDataset): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const count = dataset.points.length;
  const xs = tf.tensor2d(
    dataset.points.map((point) => [point.x, point.y]),
    [count, 2],
    'float32'
  );
  const ys = tf.tensor2d(
    dataset.points.map((point) => (point.label === 0 ? [1, 0] : [0, 1])),
    [count, dataset.numClasses],
    'float32'
  );
  return { xs, ys };
}

export function imagesToTensors(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor2D } {
  const selected = indices ?? Array.from({ length: dataset.count }, (_, index) => index);
  const size = dataset.rows * dataset.cols;
  const pixels = new Float32Array(selected.length * size);
  const oneHot = new Float32Array(selected.length * dataset.numClasses);

  selected.forEach((imageIndex, row) => {
    const source = imageAt(dataset, imageIndex);
    for (let i = 0; i < size; i++) pixels[row * size + i] = source[i] / 255;
    oneHot[row * dataset.numClasses + dataset.labels[imageIndex]] = 1;
  });

  return {
    xs: tf.tensor4d(pixels, [selected.length, dataset.rows, dataset.cols, 1]),
    ys: tf.tensor2d(oneHot, [selected.length, dataset.numClasses])
  };
}
