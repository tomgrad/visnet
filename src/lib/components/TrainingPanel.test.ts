import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import TrainingPanel from './TrainingPanel.svelte';

function panel(overrides: Partial<Record<string, unknown>> = {}) {
  const store = new NetworkStore();
  const handlers = {
    onplay: vi.fn(),
    onpause: vi.fn(),
    onstep: vi.fn(),
    onreset: vi.fn()
  };
  render(TrainingPanel, {
    props: { store, playing: false, disabled: false, ...handlers, ...overrides }
  });
  return { store, ...handlers };
}

describe('TrainingPanel', () => {
  it('writes a loss change through to the network', async () => {
    const { store } = panel();
    await userEvent.selectOptions(screen.getByTestId('training-loss'), 'mse');
    expect(store.network.training.loss).toBe('mse');
  });

  it('writes an optimizer change through to the network', async () => {
    const { store } = panel();
    await userEvent.selectOptions(screen.getByTestId('training-optimizer'), 'sgd');
    expect(store.network.training.optimizer).toBe('sgd');
  });

  it('writes a learning rate change through to the network', async () => {
    const { store } = panel();
    const input = screen.getByTestId('training-learning-rate');
    await userEvent.clear(input);
    await userEvent.type(input, '0.25');
    await userEvent.tab();
    expect(store.network.training.learningRate).toBe(0.25);
  });

  it('writes a batch size change through to the network', async () => {
    const { store } = panel();
    const input = screen.getByTestId('training-batch-size');
    await userEvent.clear(input);
    await userEvent.type(input, '64');
    await userEvent.tab();
    expect(store.network.training.batchSize).toBe(64);
  });

  it('describes every setting in plain language', () => {
    panel();
    const text = screen.getByTestId('training-panel').textContent ?? '';
    expect(text).toContain('The number the network tries to make smaller while training.');
    expect(text).toContain('The rule used to update the weights after each batch.');
    expect(text).toContain('How big each learning step is');
    expect(text).toContain('How many examples are used for one weight update.');
  });

  it('calls the control handlers', async () => {
    const { onplay, onpause, onstep, onreset } = panel({ playing: true });
    await userEvent.click(screen.getByTestId('training-play'));
    await userEvent.click(screen.getByTestId('training-pause'));
    await userEvent.click(screen.getByTestId('training-step'));
    await userEvent.click(screen.getByTestId('training-reset'));
    expect(onplay).not.toHaveBeenCalled();
    expect(onpause).toHaveBeenCalledTimes(1);
    expect(onstep).toHaveBeenCalledTimes(1);
    expect(onreset).toHaveBeenCalledTimes(1);
  });

  it('disables training controls and explains why when the network is invalid', () => {
    panel({ disabled: true });
    expect((screen.getByTestId('training-play') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-step') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-loss') as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByTestId('training-learning-rate') as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByTestId('training-blocked').textContent).toContain('fix');
  });

  it('disables play while already playing', () => {
    panel({ playing: true });
    expect((screen.getByTestId('training-play') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-pause') as HTMLButtonElement).disabled).toBe(false);
  });
});
