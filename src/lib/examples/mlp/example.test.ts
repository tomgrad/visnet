import { describe, expect, it } from 'vitest';
import { findProblems } from '../../network/problems';
import { createMlpNetwork } from './example';

describe('createMlpNetwork', () => {
  it('builds the default MLP architecture', () => {
    const network = createMlpNetwork();
    expect(network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
    expect(network.blocks[1]).toMatchObject({ kind: 'linear', units: 8 });
    expect(network.blocks[3]).toMatchObject({ kind: 'linear', units: 2 });
    expect(network.blocks[0]).toMatchObject({ kind: 'input', shape: [2] });
    expect(network.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [2] });
  });

  it('reports no problems for the 2D points task', () => {
    expect(findProblems(createMlpNetwork(), { expectedClasses: 2 })).toEqual([]);
  });

  it('returns a fresh copy on every call', () => {
    const first = createMlpNetwork();
    const second = createMlpNetwork();
    first.blocks[1] = { id: first.blocks[1].id, kind: 'linear', units: 99 };
    expect(second.blocks[1]).toMatchObject({ kind: 'linear', units: 8 });
  });
});
