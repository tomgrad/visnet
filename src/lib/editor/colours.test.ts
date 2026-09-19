import { describe, expect, it } from 'vitest';
import { BLOCK_COLOURS } from '../network/types';
import { BLOCK_COLOUR_OPTIONS, DEFAULT_BLOCK_COLOUR, blockColourHex } from './colours';

describe('BLOCK_COLOUR_OPTIONS', () => {
  it('covers the block colours exactly, in order', () => {
    expect(BLOCK_COLOUR_OPTIONS.map((option) => option.id)).toEqual([...BLOCK_COLOURS]);
  });

  it('uses six-digit hex colours', () => {
    for (const option of BLOCK_COLOUR_OPTIONS) {
      expect(option.hex).toMatch(/^#[0-9a-f]{6}$/);
    }
  });
});

describe('blockColourHex', () => {
  it('returns the option hex for a colour', () => {
    expect(blockColourHex('blue')).toBe('#dbeafe');
  });

  it('returns the default for an unset colour', () => {
    expect(blockColourHex(undefined)).toBe(DEFAULT_BLOCK_COLOUR);
  });
});
