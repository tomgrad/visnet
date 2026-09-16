import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { Network } from '../network/types';
import { NetworkInvalidError, buildModel, compileModel } from './buildModel';

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
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', units: 10 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    expect(model.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  it('refuses to build an invalid network and carries the issues', () => {
    const network: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        { id: 'out', kind: 'output', units: 2 }
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

  it('runs a forward pass with the shape the network describes', () => {
    const model = build(createEmptyNetwork());
    const prediction = model.predict(tf.tensor2d([[0.1, -0.2]])) as tf.Tensor;
    expect(prediction.shape).toEqual([1, 2]);
    const values = Array.from(prediction.dataSync());
    expect(values[0] + values[1]).toBeCloseTo(1, 5);
    prediction.dispose();
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
