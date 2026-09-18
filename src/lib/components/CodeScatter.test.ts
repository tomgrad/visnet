import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import { NetworkStore } from '../editor/networkStore.svelte';
import { codeScatter } from '../render/codes';
import CodeScatter from './CodeScatter.svelte';

vi.mock('../render/codes', () => ({ codeScatter: vi.fn() }));

const DATASET: ImageDataset = {
  count: 3,
  rows: 2,
  cols: 2,
  numClasses: 10,
  pixels: new Uint8Array(3 * 2 * 2).fill(100),
  labels: Uint8Array.of(3, 7, 1)
};

function sample() {
  return {
    points: new Float32Array([0, 0, 0.5, 0.5, 1, 1]),
    bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 }
  };
}

function fakeContext() {
  return {
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    createImageData: vi.fn((width: number, height: number) => ({
      data: new Uint8ClampedArray(width * height * 4),
      width,
      height
    })),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    imageSmoothingEnabled: false
  };
}

let context: ReturnType<typeof fakeContext>;
let originalGetContext: typeof HTMLCanvasElement.prototype.getContext;

beforeEach(() => {
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  context = fakeContext();
  vi.mocked(codeScatter).mockReturnValue(sample());
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => context as unknown as CanvasRenderingContext2D
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = originalGetContext;
  vi.restoreAllMocks();
});

function selectedStore(index = 1): NetworkStore {
  const store = new NetworkStore();
  store.select(store.network.blocks[index].id);
  return store;
}

function show(store: NetworkStore, model: unknown = {}): void {
  render(CodeScatter, {
    props: { model: model as never, store, dataset: DATASET, indices: [0, 1, 2], redrawKey: 0 }
  });
}

describe('CodeScatter', () => {
  it('shows the dimension label for the selected layer', async () => {
    show(selectedStore(1));
    await waitFor(() =>
      expect(screen.getByTestId('code-label').textContent).toContain('dimensions 1 & 2 of 8')
    );
  });

  it('cycles the dimension pair and wraps', async () => {
    show(selectedStore(1));
    await waitFor(() => screen.getByTestId('code-next'));
    await userEvent.click(screen.getByTestId('code-next'));
    expect(screen.getByTestId('code-label').textContent).toContain('dimensions 2 & 3 of 8');
    for (let i = 0; i < 6; i++) {
      await userEvent.click(screen.getByTestId('code-next'));
    }
    expect(screen.getByTestId('code-label').textContent).toContain('dimensions 1 & 2 of 8');
  });

  it('prompts to select a block', () => {
    show(new NetworkStore());
    expect(screen.getByTestId('code-message').textContent).toContain('Select a block');
  });

  it('explains when there is no model', () => {
    show(selectedStore(1), null);
    expect(screen.getByTestId('code-message').textContent).toContain('Fix the problems');
  });

  it('explains when the layer has fewer than two dimensions', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[1].id, { units: 1 });
    store.select(store.network.blocks[1].id);
    show(store);
    await waitFor(() =>
      expect(screen.getByTestId('code-message').textContent).toContain('fewer than two')
    );
  });

  it('draws the sample digits on the canvas by default', async () => {
    show(selectedStore(1));
    await waitFor(() => expect(codeScatter).toHaveBeenCalled());
    await waitFor(() => expect(context.drawImage).toHaveBeenCalled());
    expect((screen.getByTestId('code-show-digits') as HTMLInputElement).checked).toBe(true);
    expect(codeScatter).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(Uint8Array),
      2,
      2,
      3,
      0,
      0
    );
    expect(screen.queryByTestId('code-error')).toBeNull();
  });

  it('draws dots instead when the digits toggle is off', async () => {
    show(selectedStore(1));
    await waitFor(() => expect(codeScatter).toHaveBeenCalled());
    context.arc.mockClear();
    await userEvent.click(screen.getByTestId('code-show-digits'));
    await waitFor(() => expect(context.arc).toHaveBeenCalled());
  });

  it('reports a render failure', async () => {
    vi.mocked(codeScatter).mockImplementationOnce(() => {
      throw new Error('boom');
    });
    show(selectedStore(1));
    await waitFor(() =>
      expect(screen.getByTestId('code-error').textContent).toContain('could not be drawn')
    );
  });
});
