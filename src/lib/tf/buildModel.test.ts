import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { Network } from '../network/types';
import { NetworkInvalidError, buildModel, compileModel, describeBuildError } from './buildModel';

let models: tf.Sequential[] = [];

function build(net: Network): tf.Sequential {
  const model = buildModel(net);
  models.push(model);
  return model;
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('buildModel', () => {
  it('builds the default MLP with the expected layers and shapes', () => {
    const model = build(createEmptyNetwork());
    expect(model.layers.map((layer) => layer.getClassName())).toEqual([
      'Dense',
      'Activation',
      'Dense',
      'Activation'
    ]);
    expect(model.inputs[0].shape).toEqual([null, 2]);
    expect(model.outputs[0].shape).toEqual([null, 2]);
  });

  it('gives only the first layer an explicit input shape', () => {
    const model = build(createEmptyNetwork());
    expect(model.layers[0].batchInputShape).toEqual([null, 2]);
    expect(model.layers[1].outputShape).toEqual([null, 8]);
    expect(model.layers[1].batchInputShape).toBeUndefined();
  });

  it('builds a convolutional chain with flattening', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', shape: [10] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    expect(model.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  it('builds a pooling chain and halves the spatial dimensions', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', shape: [10] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('MaxPooling2D');
    expect(model.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(model.layers[0].outputShape).toEqual([null, 14, 14, 1]);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  it('runs a forward pass through a pooling layer', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', shape: [10] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    const prediction = model.predict(tf.ones([1, 28, 28, 1])) as tf.Tensor;
    expect(prediction.shape).toEqual([1, 10]);
    prediction.dispose();
  });

  it('refuses to build an invalid network and carries the issues', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        { id: 'out', kind: 'output', shape: [2] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };

    expect(() => buildModel(network)).toThrow(NetworkInvalidError);

    try {
      buildModel(network);
    } catch (error) {
      const invalid = error as NetworkInvalidError;
      expect(invalid.name).toBe('NetworkInvalidError');
      expect(invalid.issues.length).toBeGreaterThan(0);
      expect(invalid.issues.every((issue) => issue.severity === 'error')).toBe(true);
      expect(invalid.issues[0].fix.length).toBeGreaterThan(0);
    }
  });

  it('disposes the partially built model when a layer cannot be created', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [2] },
        { id: 'good', kind: 'linear', units: 8 },
        { id: 'bad', kind: 'linear', units: -1 },
        { id: 'out', kind: 'output', shape: [-1] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };

    const before = tf.memory().numTensors;
    expect(() => buildModel(network)).toThrow();
    expect(tf.memory().numTensors).toBe(before);
  });

  it('turns an unexpected TF.js build failure into a plain-language problem', () => {
    const problem = describeBuildError(new Error('Shape mismatch: expected 2, got 3'));
    expect(problem.severity).toBe('error');
    expect(problem.title.length).toBeGreaterThan(0);
    expect(problem.message).toContain('Shape mismatch');
    expect(problem.fix.length).toBeGreaterThan(0);
  });

  it('runs a forward pass with the shape the network describes', () => {
    const model = build(createEmptyNetwork());
    const prediction = model.predict(tf.tensor2d([[0.1, -0.2]])) as tf.Tensor;
    expect(prediction.shape).toEqual([1, 2]);
    const values = Array.from(prediction.dataSync());
    expect(values[0] + values[1]).toBeCloseTo(1, 5);
    prediction.dispose();
  });

  it('builds a tanh activation layer', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [2] },
        { id: 'dense', kind: 'linear', units: 4 },
        { id: 'tanh', kind: 'tanh' },
        { id: 'out', kind: 'output', shape: [4] }
      ],
      training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers.map((layer) => layer.getClassName())).toEqual(['Dense', 'Activation']);
    expect(model.outputs[0].shape).toEqual([null, 4]);
  });

  it('builds a reshape layer that changes the tensor shape', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [4, 4, 1] },
        { id: 'r', kind: 'reshape', shape: [16] },
        { id: 'out', kind: 'output', shape: [16] }
      ],
      training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('Reshape');
    expect(model.outputs[0].shape).toEqual([null, 16]);
  });

  it('builds a transposed convolution that enlarges the spatial dimensions', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [7, 7, 2] },
        {
          id: 'ct',
          kind: 'conv2dtranspose',
          filters: 3,
          kernelSize: 3,
          stride: 2,
          padding: 'same'
        },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 2 },
        { id: 'out', kind: 'output', shape: [2] }
      ],
      training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('Conv2DTranspose');
    expect(model.inputs[0].shape).toEqual([null, 7, 7, 2]);
    expect(model.layers[0].outputShape).toEqual([null, 14, 14, 3]);
  });

  it('builds an upsampling layer that enlarges the spatial dimensions', () => {
    const network: Network = {
      blocks: [
        { id: 'in', kind: 'input', shape: [4, 4, 2] },
        { id: 'up', kind: 'upsampling2d', size: 2 },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 2 },
        { id: 'out', kind: 'output', shape: [2] }
      ],
      training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('UpSampling2D');
    expect((model.layers[0].getConfig() as { interpolation: string }).interpolation).toBe(
      'nearest'
    );
    expect(model.inputs[0].shape).toEqual([null, 4, 4, 2]);
    expect(model.layers[0].outputShape).toEqual([null, 8, 8, 2]);
    expect(model.outputs[0].shape).toEqual([null, 2]);
  });
});

describe('compileModel', () => {
  it('preserves weights when the training configuration changes', () => {
    const net = createEmptyNetwork();
    const model = build(net);
    const before = model.getWeights().map((weight) => Array.from(weight.dataSync()));

    compileModel(model, { ...net.training, optimizer: 'sgd', learningRate: 0.5, loss: 'mse' });

    const after = model.getWeights().map((weight) => Array.from(weight.dataSync()));
    expect(after).toEqual(before);
  });
});
