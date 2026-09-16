import { describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { dropIndexFor, insertionIndexFor } from './placement';

function net(): Network {
  return {
    version: 1,
    blocks: [
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'a', kind: 'linear', units: 8 },
      { id: 'b', kind: 'relu' },
      { id: 'c', kind: 'linear', units: 2 },
      { id: 'out', kind: 'output', units: 2 }
    ],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

describe('insertionIndexFor', () => {
  it('inserts after the selected block', () => {
    expect(insertionIndexFor(net(), 'a')).toBe(2);
    expect(insertionIndexFor(net(), 'b')).toBe(3);
  });

  it('inserts just before the output when the selection is the last interior block', () => {
    expect(insertionIndexFor(net(), 'c')).toBe(4);
  });

  it('inserts just before the output when nothing is selected', () => {
    expect(insertionIndexFor(net(), null)).toBe(4);
  });

  it('falls back to the output slot for an unknown selection', () => {
    expect(insertionIndexFor(net(), 'missing')).toBe(4);
  });

  it('never returns an index outside the interior range', () => {
    const network = net();
    for (const id of ['in', 'a', 'b', 'c', 'out', null]) {
      const index = insertionIndexFor(network, id);
      expect(index).toBeGreaterThanOrEqual(1);
      expect(index).toBeLessThanOrEqual(network.blocks.length - 1);
    }
  });
});

describe('dropIndexFor', () => {
  const width = 200;
  const gap = 80;

  it('drops before the first interior block when left of it', () => {
    expect(dropIndexFor(0, 5, width, gap)).toBe(1);
    expect(dropIndexFor(200, 5, width, gap)).toBe(1);
  });

  it('advances one slot per interior block centre passed', () => {
    expect(dropIndexFor(379, 5, width, gap)).toBe(1);
    expect(dropIndexFor(380, 5, width, gap)).toBe(2);
    expect(dropIndexFor(659, 5, width, gap)).toBe(2);
    expect(dropIndexFor(660, 5, width, gap)).toBe(3);
    expect(dropIndexFor(940, 5, width, gap)).toBe(4);
  });

  it('clamps to the last interior slot when dropped past the end', () => {
    expect(dropIndexFor(5000, 5, width, gap)).toBe(4);
  });

  it('uses both interior slots for a three-block network', () => {
    expect(dropIndexFor(0, 3, width, gap)).toBe(1);
    expect(dropIndexFor(9999, 3, width, gap)).toBe(2);
  });

  it('stays inside the interior range for any coordinate', () => {
    for (let x = -500; x <= 3000; x += 37) {
      const index = dropIndexFor(x, 6, width, gap);
      expect(index).toBeGreaterThanOrEqual(1);
      expect(index).toBeLessThanOrEqual(5);
    }
  });
});
