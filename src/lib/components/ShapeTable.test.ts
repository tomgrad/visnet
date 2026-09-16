import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import ShapeTable from './ShapeTable.svelte';

describe('ShapeTable', () => {
  it('lists one row per block with its shapes', () => {
    const store = new NetworkStore();
    render(ShapeTable, { props: { store } });

    const rows = screen.getAllByTestId('shape-row');
    expect(rows).toHaveLength(store.network.blocks.length);
    expect(rows[0].textContent).toContain('input');
    expect(rows[1].textContent).toContain('[2]');
    expect(rows[1].textContent).toContain('[8]');
  });

  it('reports the network total parameter count', () => {
    const store = new NetworkStore();
    render(ShapeTable, { props: { store } });
    expect(screen.getByTestId('shape-table').textContent).toContain('42');
  });

  it('shows a dash for shapes it cannot compute', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    render(ShapeTable, { props: { store } });
    const rows = screen.getAllByTestId('shape-row');
    expect(rows[1].textContent).toContain('—');
  });
});
