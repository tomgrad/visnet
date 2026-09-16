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
