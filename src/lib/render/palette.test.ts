import { describe, expect, it } from 'vitest';
import { CLASS_COLOURS, classColour } from './palette';

describe('CLASS_COLOURS', () => {
  it('has at least ten distinct colours', () => {
    const hexes = CLASS_COLOURS.map((colour) => colour.hex);
    expect(hexes.length).toBeGreaterThanOrEqual(10);
    expect(new Set(hexes).size).toBe(hexes.length);
  });

  it('keeps the first two class colours unchanged', () => {
    expect(classColour(0).hex).toBe('#38bdf8');
    expect(classColour(1).hex).toBe('#fb7185');
  });

  it('falls back to a neutral colour beyond the palette', () => {
    expect(classColour(99).hex).toBe('#94a3b8');
  });
});
