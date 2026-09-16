import { describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { NODE_GAP, NODE_HEIGHT, NODE_WIDTH } from './flow';
import { dropIndexFor, insertionIndexFor } from './placement';

function net(): Network {
  return {
    version: 2,
    blocks: [
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'a', kind: 'linear', units: 8 },
      { id: 'b', kind: 'relu' },
      { id: 'c', kind: 'linear', units: 2 },
      { id: 'out', kind: 'output', units: 2 }
    ],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
    positions: {}
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
  const STEP = NODE_HEIGHT + NODE_GAP;
  const MID_X = NODE_WIDTH / 2;

  function column(count: number): { x: number; y: number }[] {
    return Array.from({ length: count }, (_, index) => ({
      x: MID_X,
      y: index * STEP + NODE_HEIGHT / 2
    }));
  }

  function at(x: number, y: number): { x: number; y: number } {
    return { x, y };
  }

  it('picks the first slot when the drop is above the first wire', () => {
    expect(dropIndexFor(at(MID_X, -100), column(5))).toBe(1);
    expect(dropIndexFor(at(MID_X, STEP / 2 - 1), column(5))).toBe(1);
  });

  it('picks the nearest wire as the drop moves down the column', () => {
    const centres = column(5);
    const midpoints = [0, 1, 2, 3].map((edge) => edge * STEP + STEP / 2 + NODE_HEIGHT / 2);

    midpoints.forEach((y, edge) => {
      expect(dropIndexFor(at(MID_X, y - 1), centres)).toBe(edge + 1);
      expect(dropIndexFor(at(MID_X, y + 1), centres)).toBe(edge + 1);
    });
  });

  it('clamps to the last interior slot when dropped past the end', () => {
    expect(dropIndexFor(at(MID_X, 5000), column(5))).toBe(4);
  });

  it('uses both interior slots for a three-block network', () => {
    expect(dropIndexFor(at(MID_X, 0), column(3))).toBe(1);
    expect(dropIndexFor(at(MID_X, 9999), column(3))).toBe(2);
  });

  it('follows the wires when the nodes are scattered rather than in a column', () => {
    const row = [at(100, 45), at(500, 45), at(900, 45)];
    expect(dropIndexFor(at(300, 45), row)).toBe(1);
    expect(dropIndexFor(at(700, 45), row)).toBe(2);
    expect(dropIndexFor(at(1500, 45), row)).toBe(2);

    const corner = [at(100, 45), at(500, 45), at(100, 445)];
    expect(dropIndexFor(at(300, 45), corner)).toBe(1);
    expect(dropIndexFor(at(300, 245), corner)).toBe(2);
  });

  it('never moves backwards as the drop point moves down a column', () => {
    const centres = column(6);
    let previous = 0;
    for (let y = -500; y <= 3000; y += 37) {
      const index = dropIndexFor(at(MID_X, y), centres);
      expect(index).toBeGreaterThanOrEqual(previous);
      previous = index;
    }
  });

  it('stays inside the interior range for any coordinate', () => {
    const centres = column(6);
    for (let y = -500; y <= 3000; y += 37) {
      const index = dropIndexFor(at(MID_X, y), centres);
      expect(index).toBeGreaterThanOrEqual(1);
      expect(index).toBeLessThanOrEqual(5);
    }
  });
});
