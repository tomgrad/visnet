import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import type { Block, Network } from './types';
import { validate, type Issue } from './validate';

function net(blocks: Network['blocks'], training?: Partial<Network['training']>): Network {
  return {
    version: 1,
    blocks,
    training: {
      loss: 'crossEntropy',
      optimizer: 'adam',
      learningRate: 0.01,
      batchSize: 32,
      ...training
    }
  };
}

const errors = (network: Network): Issue[] =>
  validate(network).filter((issue) => issue.severity === 'error');

const titles = (network: Network): string[] => errors(network).map((issue) => issue.title);

const INPUT: Block = { id: 'in', kind: 'input', shape: [2] };
const OUTPUT: Block = { id: 'out', kind: 'output', units: 2 };

describe('validate errors', () => {
  it('accepts the default network', () => {
    expect(errors(createEmptyNetwork())).toEqual([]);
  });

  it('reports a missing input block', () => {
    expect(titles(net([{ id: 'a', kind: 'relu' }, OUTPUT]))).toContain('Missing Input block');
  });

  it('reports a missing output block', () => {
    expect(titles(net([INPUT, { id: 'a', kind: 'relu' }]))).toContain('Missing Output block');
  });

  it('reports more than one input block', () => {
    const second: Block = { id: 'in2', kind: 'input', shape: [2] };
    expect(titles(net([INPUT, second, { id: 'a', kind: 'relu' }, OUTPUT]))).toContain(
      'More than one Input block'
    );
  });

  it('reports more than one output block', () => {
    const second: Block = { id: 'out2', kind: 'output', units: 2 };
    expect(titles(net([INPUT, { id: 'a', kind: 'relu' }, OUTPUT, second]))).toContain(
      'More than one Output block'
    );
  });

  it('reports a network with nothing to learn', () => {
    expect(titles(net([INPUT, OUTPUT]))).toContain('Nothing to learn');
  });

  it('reports a linear layer receiving image-shaped data and names the shape', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'dense', kind: 'linear', units: 8 },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Linear layer needs a flat list');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('dense');
    expect(issue?.message).toContain('[28, 28, 1]');
  });

  it('reports a convolution layer receiving flat data', () => {
    const network = net([
      INPUT,
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Convolution layer needs image data');
    expect(issue?.blockId).toBe('conv');
  });

  it('reports a kernel larger than the image', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [2, 2, 1] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Kernel is larger than the image');
    expect(issue?.blockId).toBe('conv');
    expect(issue?.message).toContain('3');
  });

  it('reports a flatten layer receiving an already flat list', () => {
    const network = net([INPUT, { id: 'flat', kind: 'flatten' }, OUTPUT]);
    const issue = errors(network).find((i) => i.title === 'Nothing to flatten');
    expect(issue?.blockId).toBe('flat');
  });

  it('reports an unrecognised block kind', () => {
    const bogus = { id: 'x', kind: 'dropout' } as unknown as Block;
    const issue = errors(net([INPUT, bogus, OUTPUT])).find(
      (i) => i.title === 'Unrecognised block'
    );
    expect(issue?.blockId).toBe('x');
  });

  it('gives every error a title, message, and fix', () => {
    const networks = [
      net([{ id: 'a', kind: 'relu' }, OUTPUT]),
      net([INPUT, OUTPUT]),
      net([
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        OUTPUT
      ])
    ];
    for (const network of networks) {
      for (const issue of errors(network)) {
        expect(issue.title.length).toBeGreaterThan(0);
        expect(issue.message.length).toBeGreaterThan(0);
        expect(issue.fix.length).toBeGreaterThan(0);
      }
    }
  });
});

const warnings = (network: Network, options?: { expectedClasses?: number }): Issue[] =>
  validate(network, options).filter((issue) => issue.severity === 'warning');

const warningTitles = (network: Network, options?: { expectedClasses?: number }): string[] =>
  warnings(network, options).map((issue) => issue.title);

describe('validate warnings', () => {
  it('says nothing about the default network', () => {
    expect(warnings(createEmptyNetwork())).toEqual([]);
  });

  it('suggests a softmax for cross-entropy without one', () => {
    const network = net([
      INPUT,
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ]);
    expect(warningTitles(network)).toContain('Add a Softmax for probabilities');
  });

  it('flags softmax with mean squared error', () => {
    const network = net(
      [INPUT, { id: 'dense', kind: 'linear', units: 2 }, { id: 'sm', kind: 'softmax' }, OUTPUT],
      { loss: 'mse' }
    );
    expect(warningTitles(network)).toContain('Softmax is unusual with mean squared error');
  });

  it('flags a softmax that is not the last layer', () => {
    const network = net([
      INPUT,
      { id: 'sm', kind: 'softmax' },
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ]);
    const issue = warnings(network).find((i) => i.title === 'Softmax is not the last layer');
    expect(issue?.blockId).toBe('sm');
  });

  it('does not flag a softmax immediately before the output', () => {
    const network = net([
      INPUT,
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(warningTitles(network)).not.toContain('Softmax is not the last layer');
  });

  it('flags a mismatch between output units and the dataset', () => {
    const issue = warnings(createEmptyNetwork(), { expectedClasses: 3 }).find(
      (i) => i.title === 'Output size does not match the data'
    );
    expect(issue).toBeDefined();
    expect(issue?.message).toContain('2');
    expect(issue?.message).toContain('3');
  });

  it('does not flag matching output units', () => {
    expect(warningTitles(createEmptyNetwork(), { expectedClasses: 2 })).toEqual([]);
  });

  it('flags image input without a convolution layer', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network).find(
      (i) => i.title === 'Image input without a Convolution layer'
    );
    expect(issue?.blockId).toBe('in');
  });

  it('flags a convolution layer with flat input', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [784] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network).find(
      (i) => i.title === 'Convolution layer without image input'
    );
    expect(issue?.blockId).toBe('in');
  });
});
