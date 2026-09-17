import { describe, expect, it } from 'vitest';
import { BLOCK_DESCRIPTIONS, PARAM_DESCRIPTIONS, classifyInputShape } from './descriptions';
import { BLOCK_KINDS } from './types';

describe('BLOCK_DESCRIPTIONS', () => {
  it.each([...BLOCK_KINDS])('describes %s', (kind) => {
    expect(BLOCK_DESCRIPTIONS[kind]).toBeTruthy();
    expect(BLOCK_DESCRIPTIONS[kind].length).toBeGreaterThan(15);
  });

  it('has no extra keys', () => {
    expect(Object.keys(BLOCK_DESCRIPTIONS).sort()).toEqual([...BLOCK_KINDS].sort());
  });
});

describe('PARAM_DESCRIPTIONS', () => {
  it.each(Object.keys(PARAM_DESCRIPTIONS))('describes %s', (key) => {
    expect(PARAM_DESCRIPTIONS[key as keyof typeof PARAM_DESCRIPTIONS]).toBeTruthy();
  });

  it('covers every tunable parameter', () => {
    expect(Object.keys(PARAM_DESCRIPTIONS).sort()).toEqual(
      [
        'batchSize',
        'filters',
        'inputShape',
        'kernelSize',
        'learningRate',
        'loss',
        'optimizer',
        'padding',
        'poolSize',
        'stride',
        'units'
      ].sort()
    );
  });
});

describe('classifyInputShape', () => {
  it('classifies a flat list', () => {
    expect(classifyInputShape([2])).toBe('flat');
  });

  it('classifies an image', () => {
    expect(classifyInputShape([28, 28, 1])).toBe('image');
  });

  it('classifies anything else as other', () => {
    expect(classifyInputShape([4, 4])).toBe('other');
    expect(classifyInputShape([])).toBe('other');
  });
});
