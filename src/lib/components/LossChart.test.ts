import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import LossChart from './LossChart.svelte';

describe('LossChart', () => {
  it('explains itself when there is nothing to plot yet', () => {
    render(LossChart, { props: { points: [] } });
    expect(screen.getByTestId('loss-chart-empty')).toBeTruthy();
    expect(screen.queryByTestId('loss-chart-path')).toBeNull();
  });

  it('does not plot a single point', () => {
    render(LossChart, { props: { points: [0.5] } });
    expect(screen.queryByTestId('loss-chart-path')).toBeNull();
    expect(screen.getByTestId('loss-chart-latest').textContent).toContain('0.500');
  });

  it('plots one segment per consecutive pair', () => {
    render(LossChart, { props: { points: [1, 0.5, 0.25] } });
    const path = screen.getByTestId('loss-chart-path').getAttribute('d') ?? '';
    expect(path.split('L')).toHaveLength(3);
    expect(path.startsWith('M')).toBe(true);
  });

  it('shows the latest value', () => {
    render(LossChart, { props: { points: [1, 0.5, 0.25] } });
    expect(screen.getByTestId('loss-chart-latest').textContent).toContain('0.250');
  });

  it('plots only the most recent window', () => {
    const points = Array.from({ length: 300 }, (_, index) => index);
    render(LossChart, { props: { points } });
    const path = screen.getByTestId('loss-chart-path').getAttribute('d') ?? '';
    expect(path.split('L')).toHaveLength(200);
  });
});
