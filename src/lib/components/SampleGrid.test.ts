import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import SampleGridHarness from './__stubs__/SampleGridHarness.svelte';

const SIZE = 40;

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

function dataset(count = SIZE): ImageDataset {
  const rows = 2;
  const cols = 2;
  return {
    count,
    rows,
    cols,
    numClasses: 3,
    pixels: Uint8Array.from({ length: count * rows * cols }, (_, i) => i % 256),
    labels: Uint8Array.from({ length: count }, (_, i) => i % 3)
  };
}

function fakeContext() {
  return {
    scale: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    font: '',
    textAlign: '',
    textBaseline: '',
    imageSmoothingEnabled: false
  };
}

let context: ReturnType<typeof fakeContext>;

beforeEach(() => {
  context = fakeContext();
  vi.stubGlobal('ImageData', FakeImageData);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as never);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('SampleGrid', () => {
  it('draws one border and no digit for every cell when there is no model', async () => {
    render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    expect(context.strokeRect).toHaveBeenCalledTimes(SIZE);
    expect(context.fillText).not.toHaveBeenCalled();
    expect(screen.getByTestId('sample-grid-caption').textContent).toContain('not started');
  });

  it('draws the predicted digit for every cell once there is a model', async () => {
    const model = {
      predict: () => ({
        arraySync: () => Array.from({ length: SIZE }, () => [0.1, 0.8, 0.1]),
        dispose: () => {}
      })
    };
    render(SampleGridHarness, {
      props: {
        model: model as never,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: {} as never
      }
    });
    await tick();

    expect(context.fillText).toHaveBeenCalledTimes(SIZE);
  });

  it('redraws when the redraw key changes', async () => {
    const view = render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    const before = context.strokeRect.mock.calls.length;
    expect(before).toBeGreaterThan(0);

    const harness = view.component as unknown as { bump: () => void };
    harness.bump();
    await tick();

    expect(context.strokeRect.mock.calls.length).toBeGreaterThan(before);
  });

  it('reports a failed prediction instead of throwing', async () => {
    const onerror = vi.fn();
    const model = {
      predict: () => {
        throw new Error('disposed');
      }
    };
    render(SampleGridHarness, {
      props: {
        model: model as never,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: {} as never,
        onerror
      }
    });
    await tick();

    expect(onerror).toHaveBeenCalledTimes(1);
    expect(context.fillText).not.toHaveBeenCalled();
  });

  it('keeps the grid the expected size', async () => {
    render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    const canvases = screen.getByTestId('sample-grid').querySelectorAll('canvas');
    expect(canvases).toHaveLength(2);
    expect(Number(canvases[0].getAttribute('width'))).toBeGreaterThan(0);
  });
});
