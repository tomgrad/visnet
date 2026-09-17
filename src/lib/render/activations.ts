import * as tf from '@tensorflow/tfjs';

export function forwardActivations(model: tf.LayersModel, input: tf.Tensor): tf.Tensor[] {
  const activations: tf.Tensor[] = [];
  let current: tf.Tensor = input;
  for (const layer of model.layers) {
    current = layer.apply(current) as tf.Tensor;
    activations.push(current);
  }
  return activations;
}
