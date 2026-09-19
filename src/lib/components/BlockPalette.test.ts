import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { BLOCK_CATEGORIES } from '../network/categories';
import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
import BlockPalette from './BlockPalette.svelte';

describe('BlockPalette', () => {
  it('renders a section per category with its entries', () => {
    render(BlockPalette);
    for (const category of BLOCK_CATEGORIES) {
      expect(screen.getByTestId(`palette-category-${category.id}`)).toBeTruthy();
      for (const kind of category.kinds) {
        expect(screen.getByTestId(`palette-${kind}`)).toBeTruthy();
      }
    }
  });

  it('shows the plain-language description for each block', () => {
    render(BlockPalette);
    for (const category of BLOCK_CATEGORIES) {
      for (const kind of category.kinds) {
        expect(screen.getByTestId(`palette-${kind}`).textContent).toContain(
          BLOCK_DESCRIPTIONS[kind]
        );
      }
    }
  });

  it('makes every entry draggable', () => {
    render(BlockPalette);
    for (const category of BLOCK_CATEGORIES) {
      for (const kind of category.kinds) {
        expect(screen.getByTestId(`palette-${kind}`).getAttribute('draggable')).toBe('true');
      }
    }
  });

  it('collapses and expands a category', async () => {
    render(BlockPalette);
    const header = screen.getByTestId('palette-category-dense');
    expect(header.getAttribute('aria-expanded')).toBe('true');

    await userEvent.click(header);

    expect(header.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByTestId('palette-linear')).toBeNull();

    await userEvent.click(header);
    expect(screen.getByTestId('palette-linear')).toBeTruthy();
  });
});
