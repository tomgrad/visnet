import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PointDataset } from '../data/points';
import DecisionBoundary from './DecisionBoundary.svelte';

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
});

describe('DecisionBoundary', () => {
  it('renders a canvas and a caption', () => {
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {} }
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
        caption: 'Fix the network to see the boundary.'
      }
    });
    expect(screen.getByTestId('boundary-caption').textContent).toContain('Fix the network');
  });

  it('turns a click into domain coordinates', async () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 1, onaddpoint }
    });

    const canvas = screen.getByTestId('boundary-canvas');
    fireEvent.click(canvas, { clientX: 100, clientY: 100 });
    expect(onaddpoint).toHaveBeenCalledTimes(1);
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(0, 5);
    expect(y).toBeCloseTo(0, 5);
  });

  it('maps the top-right corner to the positive x, positive y corner', () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint }
    });

    fireEvent.click(screen.getByTestId('boundary-canvas'), { clientX: 200, clientY: 0 });
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(1, 5);
    expect(y).toBeCloseTo(1, 5);
  });

  it('does not crash when the canvas has no 2d context', () => {
    expect(() =>
      render(DecisionBoundary, {
        props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {} }
      })
    ).not.toThrow();
  });
});
