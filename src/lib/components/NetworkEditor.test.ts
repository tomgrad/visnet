import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import NetworkEditor from './NetworkEditor.svelte';

vi.mock('./BlockCanvas.svelte', () => ({
  default: () => null
}));

function editor() {
  const store = new NetworkStore();
  render(NetworkEditor, {
    props: { store, palette: ['linear', 'relu', 'sigmoid', 'softmax'] }
  });
  return store;
}

describe('NetworkEditor', () => {
  it('shows the palette, the shape table, and the issues panel', () => {
    editor();
    expect(screen.getByTestId('palette-linear')).toBeTruthy();
    expect(screen.getByTestId('shape-table')).toBeTruthy();
    expect(screen.getByTestId('issues-panel')).toBeTruthy();
  });

  it('adds a block when a palette entry is clicked', async () => {
    const store = editor();
    const before = store.network.blocks.length;
    await userEvent.click(screen.getByTestId('palette-relu'));
    expect(store.network.blocks.length).toBe(before + 1);
  });

  it('undoes and redoes through the toolbar', async () => {
    const store = editor();
    const before = store.network.blocks.length;
    await userEvent.click(screen.getByTestId('palette-relu'));
    await userEvent.click(screen.getByTestId('undo'));
    expect(store.network.blocks.length).toBe(before);
    await userEvent.click(screen.getByTestId('redo'));
    expect(store.network.blocks.length).toBe(before + 1);
  });

  it('resets the network', async () => {
    const store = editor();
    await userEvent.click(screen.getByTestId('palette-relu'));
    await userEvent.click(screen.getByTestId('reset-network'));
    expect(store.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
  });

  it('surfaces an automatic correction as a dismissible announcement', async () => {
    const store = editor();
    store.select(store.network.blocks[1].id);
    store.updateBlock(store.network.blocks[1].id, { units: 0 });
    await Promise.resolve();
    expect(screen.getByTestId('announcements').textContent).toContain('Units changed from 0 to 1');
  });
});
