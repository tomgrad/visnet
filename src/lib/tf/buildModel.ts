import * as tf from '@tensorflow/tfjs';
import type { Block, Network, TrainingConfig } from '../network/types';
import { validate, type Issue } from '../network/validate';

const LOSSES: Record<TrainingConfig['loss'], string> = {
  mse: 'meanSquaredError',
  crossEntropy: 'categoricalCrossentropy'
};

export class NetworkInvalidError extends Error {
  readonly issues: Issue[];

  constructor(issues: Issue[]) {
    super('The network has errors and cannot be built.');
    this.name = 'NetworkInvalidError';
    this.issues = issues;
  }
}

function layerFor(block: Block, inputShape: number[] | undefined): tf.layers.Layer {
  switch (block.kind) {
    case 'linear':
      return tf.layers.dense({ units: block.units, inputShape });
    case 'conv2d':
      return tf.layers.conv2d({
        filters: block.filters,
        kernelSize: block.kernelSize,
        strides: block.stride,
        padding: block.padding,
        activation: 'linear',
        inputShape
      });
    case 'maxpool2d':
      return tf.layers.maxPooling2d({
        poolSize: block.poolSize,
        strides: block.stride,
        padding: block.padding,
        inputShape
      });
    case 'flatten':
      return tf.layers.flatten({ inputShape });
    case 'relu':
    case 'sigmoid':
    case 'softmax':
      return tf.layers.activation({ activation: block.kind, inputShape });
    default:
      throw new Error(`Cannot build a layer for block kind ${block.kind}`);
  }
}

export function compileModel(model: tf.LayersModel, training: TrainingConfig): void {
  const optimizer =
    training.optimizer === 'adam'
      ? tf.train.adam(training.learningRate)
      : tf.train.sgd(training.learningRate);

  model.compile({ optimizer, loss: LOSSES[training.loss] });
}

export function buildModel(net: Network): tf.Sequential {
  const errors = validate(net).filter((issue) => issue.severity === 'error');
  if (errors.length > 0) throw new NetworkInvalidError(errors);

  const inputBlock = net.blocks.find((block) => block.kind === 'input');
  const inputShape = inputBlock && inputBlock.kind === 'input' ? inputBlock.shape : undefined;

  const layers = net.blocks.filter((block) => block.kind !== 'input' && block.kind !== 'output');

  const model = tf.sequential();
  try {
    layers.forEach((block, index) => {
      model.add(layerFor(block, index === 0 ? inputShape : undefined));
    });
    compileModel(model, net.training);
  } catch (error) {
    model.dispose();
    throw error;
  }

  return model;
}
