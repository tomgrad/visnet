import { fireEvent, render, screen } from '@testing-library/svelte';
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
  const view = render(TrainingPanel, {
    props: { store, playing: false, disabled: false, lossPoints: [], ...handlers, ...overrides }
  });
  return { store, view, ...handlers };
}

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

describe('TrainingPanel', () => {
  it('labels the four controls Train, Pause, Step, and Reset', () => {
    panel();
    for (const [testId, label] of [
      ['training-play', 'Train'],
      ['training-pause', 'Pause'],
      ['training-step', 'Step'],
      ['training-reset', 'Reset']
    ]) {
      expect(screen.getByTestId(testId).textContent?.trim()).toBe(label);
    }
  });

  it('puts the controls above the training settings', () => {
    panel();
    const controls = screen.getByTestId('training-play').closest('.controls');
    const settings = screen.getByTestId('training-loss');
    expect(controls).not.toBeNull();
    expect(
      controls!.compareDocumentPosition(settings) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
  });

  it('shows the loss chart at the bottom of the panel', () => {
    panel();
    const chart = screen.getByTestId('loss-chart');
    const stats = screen.getByTestId('stats-readout');
    expect(screen.getByTestId('training-panel').contains(chart)).toBe(true);
    expect(stats.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

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
    await fireEvent.change(screen.getByTestId('training-learning-rate'), {
      target: { value: '0.25' }
    });
    expect(store.network.training.learningRate).toBe(0.25);
  });

  it('writes a batch size change through to the network', async () => {
    const { store } = panel();
    await fireEvent.change(screen.getByTestId('training-batch-size'), { target: { value: '64' } });
    expect(store.network.training.batchSize).toBe(64);
  });

  it('leaves the network alone when a numeric field is cleared', async () => {
    const { store } = panel();
    await fireEvent.change(screen.getByTestId('training-batch-size'), { target: { value: '' } });
    expect(store.network.training.batchSize).toBe(32);
  });

  it('restores the field and explains a rejected batch size', async () => {
    const { store } = panel();
    const field = screen.getByTestId('training-batch-size') as HTMLInputElement;
    await fireEvent.change(field, { target: { value: '2.5' } });
    expect(store.network.training.batchSize).toBe(32);
    expect(field.value).toBe('32');
    expect(screen.getByTestId('training-error').textContent).toContain('whole number');
  });

  it('restores the field and explains a rejected learning rate', async () => {
    const { store } = panel();
    const field = screen.getByTestId('training-learning-rate') as HTMLInputElement;
    await fireEvent.change(field, { target: { value: '-1' } });
    expect(store.network.training.learningRate).toBe(0.01);
    expect(field.value).toBe('0.01');
    expect(screen.getByTestId('training-error').textContent).toContain('positive');
  });

  it('clears the error once a valid value is entered', async () => {
    const { store } = panel();
    const field = screen.getByTestId('training-batch-size') as HTMLInputElement;
    await fireEvent.change(field, { target: { value: '2.5' } });
    await fireEvent.change(field, { target: { value: '64' } });
    expect(store.network.training.batchSize).toBe(64);
    expect(screen.queryByTestId('training-error')).toBeNull();
  });

  it('describes every setting in plain language', () => {
    panel();
    const text = screen.getByTestId('training-panel').textContent ?? '';
    expect(text).toContain('The number the network tries to make smaller while training.');
    expect(text).toContain('The rule used to update the weights after each batch.');
    expect(text).toContain('How big each learning step is');
    expect(text).toContain('How many examples are used for one weight update.');
  });

  it('starts training when play is pressed on a valid network', async () => {
    const { onplay } = panel();
    await userEvent.click(screen.getByTestId('training-play'));
    expect(onplay).toHaveBeenCalledTimes(1);
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
    const blocked = screen.getByTestId('training-blocked').textContent ?? '';
    expect(blocked).toContain('problems');
    expect(blocked).toContain('training');
  });

  it('disables play while already playing', () => {
    panel({ playing: true });
    expect((screen.getByTestId('training-play') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-pause') as HTMLButtonElement).disabled).toBe(false);
  });

  it('shows the epoch, loss and accuracy of the latest stats', () => {
    panel({
      stats: { epoch: 3, batch: 0, batchLoss: 0.12, epochMeanLoss: 0.2, epochAccuracy: 0.9 }
    });
    expect(screen.getByTestId('stats-epoch').textContent).toContain('3');
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.200');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('90');
  });

  it('hides the accuracy cell when showAccuracy is false', () => {
    panel({
      showAccuracy: false,
      stats: { epoch: 1, batch: 0, batchLoss: 0.1, epochMeanLoss: 0.2, epochAccuracy: 0.9 }
    });
    expect(screen.queryByTestId('stats-accuracy')).toBeNull();
  });

  it('prompts the user before training starts', () => {
    panel({ stats: null });
    expect(screen.getByTestId('stats-empty')).toBeTruthy();
  });

  it('labels the mid-epoch loss as the last batch and leaves accuracy blank', () => {
    panel({ stats: MID_EPOCH });
    expect(screen.getByTestId('stats-epoch').textContent).toContain('3');
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.420');
    expect(screen.getByTestId('stats-loss-label').textContent).toBe('Last batch');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('—');
  });

  it('labels the completed-epoch loss as the epoch average', () => {
    panel({ stats: EPOCH_END });
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.250');
    expect(screen.getByTestId('stats-loss-label').textContent).toBe('Epoch average');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('87.5');
  });

  it('keeps the last accuracy visible between epochs instead of blanking it', async () => {
    const { view } = panel({ stats: MID_EPOCH });
    const shown = () => screen.getByTestId('stats-accuracy').textContent?.trim();

    expect(shown()).toBe('—');

    await view.rerender({ stats: EPOCH_END });
    expect(shown()).toBe('87.5%');

    await view.rerender({ stats: MID_EPOCH });
    expect(shown()).toBe('87.5%');
  });

  it('forgets the accuracy when training resets', async () => {
    const { view } = panel({ stats: EPOCH_END });
    expect(screen.getByTestId('stats-accuracy').textContent?.trim()).toBe('87.5%');

    await view.rerender({ stats: null });
    expect(screen.getByTestId('stats-empty')).toBeTruthy();

    await view.rerender({ stats: MID_EPOCH });
    expect(screen.getByTestId('stats-accuracy').textContent?.trim()).toBe('—');
  });
});
