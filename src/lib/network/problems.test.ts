import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import type { Block, Network } from './types';
import { findProblems, type Problem, type ProblemOptions, type Severity } from './problems';

function net(blocks: Network['blocks'], training?: Partial<Network['training']>): Network {
  return {
    version: 2,
    blocks,
    training: {
      loss: 'crossEntropy',
      optimizer: 'adam',
      learningRate: 0.01,
      batchSize: 32,
      ...training
    },
    positions: {}
  };
}

const errors = (network: Network): Problem[] =>
  findProblems(network).filter((problem) => problem.severity === 'error');

const titles = (network: Network): string[] => errors(network).map((issue) => issue.title);

const INPUT: Block = { id: 'in', kind: 'input', shape: [2] };
const OUTPUT: Block = { id: 'out', kind: 'output', units: 2 };
const IMAGE_INPUT: Block = { id: 'img', kind: 'input', shape: [28, 28, 1] };
const CONV: Block = {
  id: 'conv',
  kind: 'conv2d',
  filters: 4,
  kernelSize: 3,
  stride: 1,
  padding: 'same'
};

function defaultWithLastLayerUnits(units: number): Network {
  const base = createEmptyNetwork();
  return {
    ...base,
    blocks: base.blocks.map((block) =>
      block.kind === 'linear' && block.units === 2 ? { ...block, units } : block
    )
  };
}

describe('findProblems errors', () => {
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

  it('reports nothing to learn when two inputs sit directly against the output', () => {
    const second: Block = { id: 'in2', kind: 'input', shape: [2] };
    expect(titles(net([INPUT, second, OUTPUT]))).toContain('Nothing to learn');
  });

  it('does not claim the input connects straight to the output when a layer is present', () => {
    expect(titles(net([INPUT, { id: 'a', kind: 'relu' }]))).not.toContain('Nothing to learn');
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

  it('names a rank-2 Linear input without calling it image-shaped', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [4, 4] },
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Linear layer needs a flat list');
    expect(issue).toBeDefined();
    expect(issue?.message).toContain('[4, 4]');
    expect(issue?.message).not.toContain('image-shaped');
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

  it('reports a pooling layer receiving flat data', () => {
    const network = net([
      INPUT,
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Pooling layer needs image data');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('pool');
    expect(issue?.message).toContain('[2]');
  });

  it('reports a pool window larger than the image', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [2, 2, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Pool window is larger than the image');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('pool');
    expect(issue?.message).toContain('3');
  });

  it('reports an upsampling layer receiving flat data', () => {
    const network = net([INPUT, { id: 'up', kind: 'upsampling2d', size: 2 }, OUTPUT]);
    const issue = errors(network).find((i) => i.title === 'Upsampling layer needs image data');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('up');
    expect(issue?.message).toContain('[2]');
  });

  it('reports a flatten layer receiving an already flat list', () => {
    const network = net([INPUT, { id: 'flat', kind: 'flatten' }, OUTPUT]);
    const issue = errors(network).find((i) => i.title === 'Nothing to flatten');
    expect(issue?.blockId).toBe('flat');
  });

  it('reports a reshape whose element count does not match', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'r', kind: 'reshape', shape: [10] },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Reshape size does not match');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('r');
    expect(issue?.message).toContain('16');
    expect(issue?.message).toContain('10');
  });

  it('reports an unrecognised block kind', () => {
    const bogus = { id: 'x', kind: 'dropout' } as unknown as Block;
    const issue = errors(net([INPUT, bogus, OUTPUT])).find((i) => i.title === 'Unrecognised block');
    expect(issue?.blockId).toBe('x');
  });

  it('reports when the last real layer does not match the Output block', () => {
    const network = defaultWithLastLayerUnits(3);
    const issue = errors(network).find(
      (i) => i.title === 'Last layer size does not match the Output block'
    );
    const lastReal = network.blocks.filter((block) => block.kind !== 'output');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe(lastReal[lastReal.length - 1].id);
    expect(issue?.message).toContain('3');
    expect(issue?.message).toContain('2');
  });
});

const warnings = (network: Network, options?: ProblemOptions): Problem[] =>
  findProblems(network, options).filter((problem) => problem.severity === 'warning');

const warningTitles = (network: Network, options?: { expectedClasses?: number }): string[] =>
  warnings(network, options).map((issue) => issue.title);

describe('findProblems warnings', () => {
  it('says nothing about the default network', () => {
    expect(warnings(createEmptyNetwork())).toEqual([]);
  });

  it('suggests a softmax for cross-entropy without one', () => {
    const network = net([INPUT, { id: 'dense', kind: 'linear', units: 2 }, OUTPUT]);
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

const ERROR_RULE_TITLES = [
  'Missing Input block',
  'Missing Output block',
  'More than one Input block',
  'More than one Output block',
  'Nothing to learn',
  'Unrecognised block',
  'Linear layer needs a flat list',
  'Convolution layer needs image data',
  'Kernel is larger than the image',
  'Pooling layer needs image data',
  'Pool window is larger than the image',
  'Upsampling layer needs image data',
  'Nothing to flatten',
  'Reshape size does not match',
  'Output must be a list of scores',
  'Last layer size does not match the Output block'
];

const WARNING_RULE_TITLES = [
  'Add a Softmax for probabilities',
  'Softmax is unusual with mean squared error',
  'Softmax is not the last layer',
  'Output size does not match the data',
  'Image input without a Convolution layer',
  'Convolution layer without image input',
  'Input shape does not match the data'
];

interface RuleCase {
  title: string;
  severity: Severity;
  network: Network;
  options?: ProblemOptions;
}

const RULE_CASES: RuleCase[] = [
  {
    title: 'Missing Input block',
    severity: 'error',
    network: net([{ id: 'a', kind: 'relu' }, OUTPUT])
  },
  {
    title: 'Missing Output block',
    severity: 'error',
    network: net([INPUT, { id: 'a', kind: 'relu' }])
  },
  {
    title: 'More than one Input block',
    severity: 'error',
    network: net([
      INPUT,
      { id: 'in2', kind: 'input', shape: [2] },
      { id: 'a', kind: 'relu' },
      OUTPUT
    ])
  },
  {
    title: 'More than one Output block',
    severity: 'error',
    network: net([
      INPUT,
      { id: 'a', kind: 'relu' },
      OUTPUT,
      { id: 'out2', kind: 'output', units: 2 }
    ])
  },
  { title: 'Nothing to learn', severity: 'error', network: net([INPUT, OUTPUT]) },
  {
    title: 'Unrecognised block',
    severity: 'error',
    network: net([INPUT, { id: 'x', kind: 'dropout' } as unknown as Block, OUTPUT])
  },
  {
    title: 'Linear layer needs a flat list',
    severity: 'error',
    network: net([IMAGE_INPUT, { id: 'dense', kind: 'linear', units: 2 }, OUTPUT])
  },
  {
    title: 'Convolution layer needs image data',
    severity: 'error',
    network: net([INPUT, CONV, OUTPUT])
  },
  {
    title: 'Kernel is larger than the image',
    severity: 'error',
    network: net([
      { id: 'small', kind: 'input', shape: [2, 2, 1] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ])
  },
  {
    title: 'Pooling layer needs image data',
    severity: 'error',
    network: net([
      INPUT,
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      OUTPUT
    ])
  },
  {
    title: 'Pool window is larger than the image',
    severity: 'error',
    network: net([
      { id: 'small', kind: 'input', shape: [2, 2, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ])
  },
  {
    title: 'Upsampling layer needs image data',
    severity: 'error',
    network: net([INPUT, { id: 'up', kind: 'upsampling2d', size: 2 }, OUTPUT])
  },
  {
    title: 'Nothing to flatten',
    severity: 'error',
    network: net([INPUT, { id: 'flat', kind: 'flatten' }, OUTPUT])
  },
  {
    title: 'Reshape size does not match',
    severity: 'error',
    network: net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'r', kind: 'reshape', shape: [10] },
      OUTPUT
    ])
  },
  {
    title: 'Output must be a list of scores',
    severity: 'error',
    network: net(
      [{ id: 'grid', kind: 'input', shape: [2, 2] }, { id: 'act', kind: 'relu' }, OUTPUT],
      { loss: 'mse' }
    )
  },
  {
    title: 'Last layer size does not match the Output block',
    severity: 'error',
    network: defaultWithLastLayerUnits(3)
  },
  {
    title: 'Add a Softmax for probabilities',
    severity: 'warning',
    network: net([INPUT, { id: 'dense', kind: 'linear', units: 2 }, OUTPUT])
  },
  {
    title: 'Softmax is unusual with mean squared error',
    severity: 'warning',
    network: net(
      [INPUT, { id: 'dense', kind: 'linear', units: 2 }, { id: 'sm', kind: 'softmax' }, OUTPUT],
      { loss: 'mse' }
    )
  },
  {
    title: 'Softmax is not the last layer',
    severity: 'warning',
    network: net([
      INPUT,
      { id: 'sm', kind: 'softmax' },
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ])
  },
  {
    title: 'Output size does not match the data',
    severity: 'warning',
    network: createEmptyNetwork(),
    options: { expectedClasses: 3 }
  },
  {
    title: 'Image input without a Convolution layer',
    severity: 'warning',
    network: net([
      IMAGE_INPUT,
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ])
  },
  {
    title: 'Convolution layer without image input',
    severity: 'warning',
    network: net([
      { id: 'flat', kind: 'input', shape: [784] },
      CONV,
      { id: 'flat2', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ])
  },
  {
    title: 'Input shape does not match the data',
    severity: 'warning',
    options: { expectedInputShape: [28, 28, 1] },
    network: net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ])
  }
];

describe('issue message contract', () => {
  it.each(RULE_CASES)(
    '$title produces issues with a title, message, and fix',
    ({ network, options, title }) => {
      const issues = findProblems(network, options);
      expect(issues.map((issue) => issue.title)).toContain(title);
      for (const issue of issues) {
        expect(issue.title.length).toBeGreaterThan(0);
        expect(issue.message.length).toBeGreaterThan(0);
        expect(issue.fix.length).toBeGreaterThan(0);
      }
    }
  );

  it('has a case for every rule and no case without a rule', () => {
    const errorCases = RULE_CASES.filter((entry) => entry.severity === 'error').map(
      (entry) => entry.title
    );
    const warningCases = RULE_CASES.filter((entry) => entry.severity === 'warning').map(
      (entry) => entry.title
    );
    expect(new Set(errorCases)).toEqual(new Set(ERROR_RULE_TITLES));
    expect(new Set(warningCases)).toEqual(new Set(WARNING_RULE_TITLES));

    const produced = new Set(
      RULE_CASES.flatMap((entry) =>
        findProblems(entry.network, entry.options).map((issue) => issue.title)
      )
    );
    expect(produced).toEqual(new Set([...ERROR_RULE_TITLES, ...WARNING_RULE_TITLES]));
  });
});

describe('expectedInputShape', () => {
  it('says nothing when the input matches', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(
      warnings(network, { expectedInputShape: [28, 28, 1] }).map((i) => i.title)
    ).not.toContain('Input shape does not match the data');
  });

  it('warns when the input shape differs', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network, { expectedInputShape: [28, 28, 1] }).find(
      (i) => i.title === 'Input shape does not match the data'
    );
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('in');
    expect(issue?.message).toContain('[4, 4, 1]');
    expect(issue?.message).toContain('[28, 28, 1]');
    expect(issue?.fix).toContain('[28, 28, 1]');
  });

  it('warns when the ranks differ', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [784] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(warnings(network, { expectedInputShape: [28, 28, 1] }).map((i) => i.title)).toContain(
      'Input shape does not match the data'
    );
  });

  it('says nothing when no shape is expected', () => {
    expect(warnings(createEmptyNetwork())).toEqual([]);
  });
});
