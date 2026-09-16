import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { shapesMatch, weightShapes, weightsUrl } from './weights';

let models: tf.Sequential[] = [];

function model(units = 8): tf.Sequential {
  const net = createEmptyNetwork();
  const hidden = net.blocks[1];
  if (hidden.kind === 'linear') hidden.units = units;
  const built = buildModel(net);
  models.push(built);
  return built;
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((built) => built.dispose());
  models = [];
});

describe('weightShapes', () => {
  it('reports every weight tensor shape', () => {
    expect(weightShapes(model())).toEqual([[2, 8], [8], [8, 2], [2]]);
  });

  it('changes when the architecture changes', () => {
    expect(weightShapes(model(16))[0]).toEqual([2, 16]);
  });
});

describe('shapesMatch', () => {
  it('accepts identical shapes', () => {
    expect(shapesMatch(weightShapes(model()), weightShapes(model()))).toBe(true);
  });

  it('rejects different shapes', () => {
    expect(shapesMatch(weightShapes(model()), weightShapes(model(16)))).toBe(false);
  });

  it('rejects different lengths', () => {
    expect(shapesMatch([[2, 8]], [])).toBe(false);
  });

  it('rejects a mismatched dimension inside a tensor', () => {
    expect(shapesMatch([[2, 8]], [[2, 9]])).toBe(false);
  });
});

describe('weightsUrl', () => {
  it('namespaces by example id', () => {
    expect(weightsUrl('mlp')).toBe('indexeddb://visnet/weights/mlp');
    expect(weightsUrl('cnn')).toBe('indexeddb://visnet/weights/cnn');
    expect(weightsUrl('mlp')).not.toBe(weightsUrl('cnn'));
  });
});
