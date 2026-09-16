import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import StatsReadout from './StatsReadout.svelte';

const MID_EPOCH = {
  epoch: 3,
  batch: 2,
  batchLoss: 0.42,
  epochMeanLoss: null,
  epochAccuracy: null
};

const EPOCH_END = {
  epoch: 4,
  batch: 0,
  batchLoss: 0.2,
  epochMeanLoss: 0.25,
  epochAccuracy: 0.875
};

describe('StatsReadout', () => {
  it('prompts the user before training starts', () => {
    render(StatsReadout, { props: { stats: null } });
    expect(screen.getByTestId('stats-empty')).toBeTruthy();
  });

  it('shows the epoch and loss mid-epoch', () => {
    render(StatsReadout, {
      props: {
        stats: MID_EPOCH
      }
    });
    expect(screen.getByTestId('stats-epoch').textContent).toContain('3');
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.420');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('—');
  });

  it('shows accuracy once an epoch has completed', () => {
    render(StatsReadout, {
      props: {
        stats: EPOCH_END
      }
    });
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.250');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('87.5');
  });

  it('keeps the same cells whether or not an epoch has completed', () => {
    const mid = render(StatsReadout, {
      props: {
        stats: MID_EPOCH
      }
    });
    const cells = (root: HTMLElement) => root.querySelectorAll('dl > div').length;

    expect(cells(mid.container)).toBe(3);

    const end = render(StatsReadout, {
      props: {
        stats: EPOCH_END
      }
    });
    expect(cells(end.container)).toBe(3);
  });

  it('names which loss is shown, in a label short enough to stay on one line', () => {
    const label = (root: HTMLElement) =>
      root.querySelector('[data-testid="stats-loss-label"]')?.textContent;

    const mid = render(StatsReadout, {
      props: {
        stats: MID_EPOCH
      }
    });
    expect(label(mid.container)).toBe('Last batch');

    const end = render(StatsReadout, {
      props: {
        stats: EPOCH_END
      }
    });
    expect(label(end.container)).toBe('Epoch average');
  });

  it('keeps the last accuracy visible between epochs instead of blanking it', async () => {
    const view = render(StatsReadout, { props: { stats: MID_EPOCH } });
    const shown = () => view.getByTestId('stats-accuracy').textContent?.trim();

    expect(shown()).toBe('—');

    await view.rerender({ stats: EPOCH_END });
    await tick();
    expect(shown()).toBe('87.5%');

    await view.rerender({ stats: MID_EPOCH });
    await tick();
    expect(shown()).toBe('87.5%');
  });

  it('forgets the accuracy when training resets', async () => {
    const view = render(StatsReadout, { props: { stats: EPOCH_END } });
    expect(view.getByTestId('stats-accuracy').textContent?.trim()).toBe('87.5%');

    await view.rerender({ stats: null });
    await tick();
    expect(view.getByTestId('stats-empty')).toBeTruthy();

    await view.rerender({ stats: MID_EPOCH });
    await tick();
    expect(view.getByTestId('stats-accuracy').textContent?.trim()).toBe('—');
  });
});
