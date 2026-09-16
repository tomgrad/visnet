import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PointDataset } from '../data/points';
import DecisionBoundary from './DecisionBoundary.svelte';
import DecisionBoundaryHarness from './DecisionBoundaryHarness.svelte';

const DATASET: PointDataset = {
  points: [
    { x: 0, y: 0, label: 0 },
    { x: 0.5, y: 0.5, label: 1 }
  ],
  numClasses: 2
};

const RECT = { left: 0, top: 0, width: 200, height: 200 };

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    ...RECT,
    right: RECT.width,
    bottom: RECT.height,
    x: 0,
    y: 0,
    toJSON: () => ({})
  } as DOMRect);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

describe('DecisionBoundary', () => {
  async function waitForBoundary(): Promise<void> {
    await vi.waitFor(() => {
      expect(screen.getByTestId('decision-boundary').getAttribute('data-ready')).toBe('true');
    });
  }

  it('renders a canvas and a caption', () => {
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {}, redrawKey: 0 }
    });
    expect(screen.getByTestId('boundary-canvas')).toBeTruthy();
    expect(screen.getByTestId('boundary-caption').textContent).toBeTruthy();
  });

  it('says the network must be fixed before a boundary can be drawn', () => {
    render(DecisionBoundary, {
      props: {
        model: null,
        dataset: DATASET,
        selectedLabel: 0,
        onaddpoint: () => {},
        redrawKey: 0,
        caption: 'Fix the network to see the boundary.'
      }
    });
    expect(screen.getByTestId('boundary-caption').textContent).toContain('Fix the network');
  });

  it('turns a click into domain coordinates', async () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 1, onaddpoint, redrawKey: 0 }
    });
    await waitForBoundary();

    const canvas = screen.getByTestId('boundary-canvas');
    fireEvent.click(canvas, { clientX: 100, clientY: 100 });
    expect(onaddpoint).toHaveBeenCalledTimes(1);
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(0, 5);
    expect(y).toBeCloseTo(0, 5);
  });

  it('maps the top-right corner to the positive x, positive y corner', async () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint, redrawKey: 0 }
    });
    await waitForBoundary();

    fireEvent.click(screen.getByTestId('boundary-canvas'), { clientX: 200, clientY: 0 });
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(1, 5);
    expect(y).toBeCloseTo(1, 5);
  });

  it('does not crash when the canvas has no 2d context', () => {
    expect(() =>
      render(DecisionBoundary, {
        props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {}, redrawKey: 0 }
      })
    ).not.toThrow();
  });

  it('redraws when the redraw key changes', async () => {
    const fillRect = vi.fn();
    const context = {
      fillRect,
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      putImageData: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 0,
      imageSmoothingEnabled: false
    };
    HTMLCanvasElement.prototype.getContext = vi.fn(
      () => context as unknown as CanvasRenderingContext2D
    ) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    const { component } = render(DecisionBoundaryHarness, {
      props: {
        model: null,
        dataset: DATASET,
        selectedLabel: 0,
        onaddpoint: () => {}
      }
    });
    await waitForBoundary();

    const before = fillRect.mock.calls.length;
    expect(before).toBeGreaterThan(0);

    (component as unknown as { bump: () => void }).bump();

    await vi.waitFor(() => {
      expect(fillRect.mock.calls.length).toBeGreaterThan(before);
    });
  });
});
