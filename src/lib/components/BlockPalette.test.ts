import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
import type { BlockKind } from '../network/types';
import BlockPalette from './BlockPalette.svelte';

const PALETTE: BlockKind[] = ['linear', 'relu', 'sigmoid', 'softmax'];

describe('BlockPalette', () => {
  it('renders one button per permitted block', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`)).toBeTruthy();
    }
    expect(screen.queryByTestId('palette-conv2d')).toBeNull();
  });

  it('shows the plain-language description for each block', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`).textContent).toContain(BLOCK_DESCRIPTIONS[kind]);
    }
  });

  it('reports the chosen kind on click', async () => {
    const onadd = vi.fn();
    render(BlockPalette, { props: { palette: PALETTE, onadd } });
    await userEvent.click(screen.getByTestId('palette-relu'));
    expect(onadd).toHaveBeenCalledWith('relu');
  });

  it('makes every entry draggable', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`).getAttribute('draggable')).toBe('true');
    }
  });
});
