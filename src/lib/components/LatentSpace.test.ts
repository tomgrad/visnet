import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { PointDataset } from '../data/points';
import { NetworkStore } from '../editor/networkStore.svelte';
import LatentSpace from './LatentSpace.svelte';

vi.mock('../render/latent', () => ({
  LATENT_GRID_SIZE: 32,
  gridInputs: (size: number) => new Float32Array(size * size * 2),
  projectLatent: () => ({
    grid: new Float32Array(32 * 32 * 2),
    gridClasses: new Int32Array(32 * 32),
    points: new Float32Array(4),
    bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 }
  })
}));

const DATASET: PointDataset = {
  points: [
    { x: 0, y: 0, label: 0 },
    { x: 0.5, y: 0.5, label: 1 }
  ],
  numClasses: 2
};

function selectedStore(index = 1): NetworkStore {
  const store = new NetworkStore();
  store.select(store.network.blocks[index].id);
  return store;
}

function show(store: NetworkStore, model: unknown = {}): void {
  render(LatentSpace, {
    props: { model: model as never, store, dataset: DATASET, redrawKey: 0 }
  });
}

describe('LatentSpace', () => {
  it('shows the dimension label for the selected layer', async () => {
    show(selectedStore(1));
    await waitFor(() =>
      expect(screen.getByTestId('latent-label').textContent).toContain('dimensions 1 & 2 of 8')
    );
  });

  it('cycles the dimension pair and wraps', async () => {
    show(selectedStore(1));
    await waitFor(() => screen.getByTestId('latent-next'));
    for (let i = 0; i < 7; i++) {
      await userEvent.click(screen.getByTestId('latent-next'));
    }
    expect(screen.getByTestId('latent-label').textContent).toContain('dimensions 1 & 2 of 8');
  });

  it('prompts to select a block', () => {
    show(new NetworkStore());
    expect(screen.getByTestId('latent-message').textContent).toContain('Select a block');
  });

  it('explains when there is no model', () => {
    show(selectedStore(1), null);
    expect(screen.getByTestId('latent-message').textContent).toContain('Fix the problems');
  });

  it('explains when the layer has fewer than two dimensions', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[1].id, { units: 1 });
    store.select(store.network.blocks[1].id);
    show(store);
    await waitFor(() =>
      expect(screen.getByTestId('latent-message').textContent).toContain('fewer than two')
    );
  });
});
