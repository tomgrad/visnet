import * as tf from '@tensorflow/tfjs';
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
