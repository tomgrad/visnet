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
    expect(screen.queryByTestId('stats-accuracy')).toBeNull();
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
});
