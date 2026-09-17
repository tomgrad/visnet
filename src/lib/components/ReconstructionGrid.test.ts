import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import { reconstruct } from '../render/reconstruction';
import ReconstructionGrid from './ReconstructionGrid.svelte';

vi.mock('../render/reconstruction', () => ({
  reconstruct: vi.fn(() => new Uint8ClampedArray(3 * 2 * 2).fill(128))
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
  vi.mocked(reconstruct).mockClear();
  vi.stubGlobal('ImageData', FakeImageData);
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    scale: vi.fn(),
    clearRect: vi.fn(),
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

function show(model: unknown = {}, dataset: ImageDataset | null = DATASET): void {
  render(ReconstructionGrid, {
    props: { model: model as never, dataset, indices: [0, 1, 2], redrawKey: 0 }
  });
}

describe('ReconstructionGrid', () => {
  it('draws the reconstructions for the sample digits', async () => {
    show();
    await waitFor(() => expect(reconstruct).toHaveBeenCalled());
    expect(reconstruct).toHaveBeenCalledWith(expect.anything(), expect.any(Uint8Array), 2, 2, 3);
    expect(screen.queryByTestId('reconstruction-error')).toBeNull();
  });

  it('explains when there is no model', () => {
    show(null);
    expect(screen.getByTestId('reconstruction-message').textContent).toContain('Fix the problems');
  });

  it('explains when the digit images are not loaded', () => {
    show({}, null);
    expect(screen.getByTestId('reconstruction-message').textContent).toContain('not loaded');
  });

  it('reports a render failure', async () => {
    vi.mocked(reconstruct).mockImplementationOnce(() => {
      throw new Error('boom');
    });
    show();
    await waitFor(() =>
      expect(screen.getByTestId('reconstruction-error').textContent).toContain('could not be drawn')
    );
  });
});
