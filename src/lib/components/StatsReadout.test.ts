import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import StatsReadout from './StatsReadout.svelte';

describe('StatsReadout', () => {
  it('prompts the user before training starts', () => {
    render(StatsReadout, { props: { stats: null } });
    expect(screen.getByTestId('stats-empty')).toBeTruthy();
  });

  it('shows the epoch and loss mid-epoch', () => {
    render(StatsReadout, {
      props: {
        stats: { epoch: 3, batch: 2, batchLoss: 0.42, epochMeanLoss: null, epochAccuracy: null }
      }
    });
    expect(screen.getByTestId('stats-epoch').textContent).toContain('3');
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.420');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('—');
  });

  it('shows accuracy once an epoch has completed', () => {
    render(StatsReadout, {
      props: {
        stats: {
          epoch: 4,
          batch: 0,
          batchLoss: 0.2,
          epochMeanLoss: 0.25,
          epochAccuracy: 0.875
        }
      }
    });
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.250');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('87.5');
  });

  it('keeps the same cells whether or not an epoch has completed', () => {
    const mid = render(StatsReadout, {
      props: {
        stats: { epoch: 3, batch: 2, batchLoss: 0.42, epochMeanLoss: null, epochAccuracy: null }
      }
    });
    const cells = (root: HTMLElement) => root.querySelectorAll('dl > div').length;

    expect(cells(mid.container)).toBe(3);

    const end = render(StatsReadout, {
      props: {
        stats: {
          epoch: 4,
          batch: 0,
          batchLoss: 0.2,
          epochMeanLoss: 0.25,
          epochAccuracy: 0.875
        }
      }
    });
    expect(cells(end.container)).toBe(3);
  });

  it('names which loss is shown, in a label short enough to stay on one line', () => {
    const label = (root: HTMLElement) =>
      root.querySelector('[data-testid="stats-loss-label"]')?.textContent;

    const mid = render(StatsReadout, {
      props: {
        stats: { epoch: 3, batch: 2, batchLoss: 0.42, epochMeanLoss: null, epochAccuracy: null }
      }
    });
    expect(label(mid.container)).toBe('Last batch');

    const end = render(StatsReadout, {
      props: {
        stats: {
          epoch: 4,
          batch: 0,
          batchLoss: 0.2,
          epochMeanLoss: 0.25,
          epochAccuracy: 0.875
        }
      }
    });
    expect(label(end.container)).toBe('Epoch average');
  });
});
