import { describe, expect, it } from 'vitest';
import { BLOCK_CATEGORIES, PALETTE_KINDS } from './categories';
import { BLOCK_KINDS } from './types';

describe('BLOCK_CATEGORIES', () => {
  it('has no empty category', () => {
    for (const category of BLOCK_CATEGORIES) {
      expect(category.kinds.length).toBeGreaterThan(0);
    }
  });

  it('places each palette kind in exactly one category', () => {
    expect(new Set(PALETTE_KINDS).size).toBe(PALETTE_KINDS.length);
  });

  it('covers every non-pinned block kind', () => {
    const pinned = new Set(['input', 'output']);
    const expected = BLOCK_KINDS.filter((kind) => !pinned.has(kind));
    expect(new Set(PALETTE_KINDS)).toEqual(new Set(expected));
  });

  it('excludes the pinned input and output markers', () => {
    expect(PALETTE_KINDS).not.toContain('input');
    expect(PALETTE_KINDS).not.toContain('output');
  });
});
