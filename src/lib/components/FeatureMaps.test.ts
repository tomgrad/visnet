import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import { NetworkStore } from '../editor/networkStore.svelte';
import { createCnnNetwork } from '../examples/cnn/example';
import { featureMaps } from '../render/features';
import FeatureMaps from './FeatureMaps.svelte';

vi.mock('../render/features', () => ({
  featureMaps: vi.fn(() => [
    { width: 2, height: 2, values: new Uint8ClampedArray([0, 128, 255, 64]) }
  ])
}));

const DATASET: ImageDataset = {
  count: 3,
  rows: 2,
  cols: 2,
  numClasses: 10,
  pixels: new Uint8Array(3 * 2 * 2).fill(100),
  labels: Uint8Array.of(3, 7, 1)
};

class FakeImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;

  constructor(data: Uint8ClampedArray, width: number, height: number) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

let originalGetContext: typeof HTMLCanvasElement.prototype.getContext;

beforeEach(() => {
  vi.mocked(featureMaps).mockClear();
  vi.stubGlobal('ImageData', FakeImageData);
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    scale: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    set fillStyle(_value: string) {},
    set imageSmoothingEnabled(_value: boolean) {}
  })) as never;
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = originalGetContext;
  vi.unstubAllGlobals();
});

function storeSelecting(index: number): NetworkStore {
  const store = new NetworkStore(createCnnNetwork());
  store.select(store.network.blocks[index].id);
  return store;
}

function show(store: NetworkStore, model: unknown = {}, dataset: ImageDataset | null = DATASET) {
  render(FeatureMaps, {
    props: { model: model as never, store, dataset, indices: [0, 1, 2], redrawKey: 0 }
  });
}

describe('FeatureMaps', () => {
  it('draws the maps for a convolution block', async () => {
    show(storeSelecting(1));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.queryByTestId('feature-error')).toBeNull();
  });

  it('draws the maps for a dense block', async () => {
    vi.mocked(featureMaps).mockReturnValueOnce([
      { width: 1, height: 1, values: Uint8ClampedArray.of(200) }
    ]);
    show(storeSelecting(4));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.queryByTestId('feature-error')).toBeNull();
  });

  it('cycles the digit and wraps', async () => {
    show(storeSelecting(1));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 1 of 3');
    await userEvent.click(screen.getByTestId('feature-next'));
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 2 of 3');
    await userEvent.click(screen.getByTestId('feature-next'));
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 3 of 3');
    await userEvent.click(screen.getByTestId('feature-next'));
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 1 of 3');
  });

  it('prompts to select a block', () => {
    show(new NetworkStore(createCnnNetwork()));
    expect(screen.getByTestId('feature-message').textContent).toContain('Select a block');
  });

  it('explains when there is no model', () => {
    show(storeSelecting(1), null);
    expect(screen.getByTestId('feature-message').textContent).toContain('Fix the problems');
  });
});
