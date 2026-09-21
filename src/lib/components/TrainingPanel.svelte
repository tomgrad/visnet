<script lang="ts">
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { PARAM_DESCRIPTIONS } from '../network/descriptions';
  import type { TrainStats } from '../training/Trainer';
  import LossChart from './LossChart.svelte';

  let {
    store,
    playing,
    disabled,
    lossPoints,
    stats = null,
    showAccuracy = true,
    onplay,
    onpause,
    onstep,
    onreset
  }: {
    store: NetworkStore;
    playing: boolean;
    disabled: boolean;
    lossPoints: number[];
    stats?: TrainStats | null;
    showAccuracy?: boolean;
    onplay: () => void;
    onpause: () => void;
    onstep: () => void;
    onreset: () => void;
  } = $props();

  let error = $state<string | null>(null);

  const loss = $derived(stats ? (stats.epochMeanLoss ?? stats.batchLoss) : null);
  const lossLabel = $derived(stats?.epochMeanLoss == null ? 'Last batch' : 'Epoch average');
  const lossExplanation = $derived(
    stats?.epochMeanLoss == null
      ? 'Loss on the most recent batch. Lower is better.'
      : 'Average loss over the epoch that just finished. Lower is better.'
  );

  let lastAccuracy = $state<number | null>(null);

  $effect(() => {
    if (!stats) {
      lastAccuracy = null;
      return;
    }
    if (stats.epochAccuracy !== null) lastAccuracy = stats.epochAccuracy;
  });

  const accuracy = $derived(stats ? (stats.epochAccuracy ?? lastAccuracy) : null);

  function setNumber(event: Event, key: 'learningRate' | 'batchSize'): void {
    const input = event.currentTarget as HTMLInputElement;
    const value = Number(input.value);
    const valid =
      input.value.trim() !== '' &&
      Number.isFinite(value) &&
      value > 0 &&
      (key !== 'batchSize' || Number.isInteger(value));

    if (!valid) {
      input.value = String(store.network.training[key]);
      error =
        key === 'batchSize'
          ? 'Batch size must be a positive whole number.'
          : 'Learning rate must be a positive number.';
      return;
    }

    error = null;
    store.updateTraining({ [key]: value });
  }
</script>

<div class="training" data-testid="training-panel">
  <h2>Training</h2>

  <div class="controls">
    <button
      type="button"
      data-testid="training-play"
      disabled={disabled || playing}
      onclick={onplay}
    >
      Train
    </button>
    <button type="button" data-testid="training-pause" disabled={!playing} onclick={onpause}>
      Pause
    </button>
    <button type="button" data-testid="training-step" {disabled} onclick={onstep}>Step</button>
    <button type="button" data-testid="training-reset" onclick={onreset}>Reset</button>
  </div>

  <label>
    <span>Loss</span>
    <select
      data-testid="training-loss"
      title={PARAM_DESCRIPTIONS.loss}
      {disabled}
      value={store.network.training.loss}
      onchange={(event) =>
        store.updateTraining({ loss: event.currentTarget.value as 'mse' | 'crossEntropy' })}
    >
      <option value="crossEntropy">cross entropy</option>
      <option value="mse">mean squared error</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.loss}</small>
  </label>

  <label>
    <span>Optimizer</span>
    <select
      data-testid="training-optimizer"
      title={PARAM_DESCRIPTIONS.optimizer}
      {disabled}
      value={store.network.training.optimizer}
      onchange={(event) =>
        store.updateTraining({ optimizer: event.currentTarget.value as 'sgd' | 'adam' })}
    >
      <option value="adam">Adam</option>
      <option value="sgd">SGD</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.optimizer}</small>
  </label>

  <label>
    <span>Learning rate</span>
    <input
      type="number"
      step="0.001"
      min="0.0001"
      data-testid="training-learning-rate"
      title={PARAM_DESCRIPTIONS.learningRate}
      {disabled}
      value={store.network.training.learningRate}
      onchange={(event) => setNumber(event, 'learningRate')}
    />
    <small>{PARAM_DESCRIPTIONS.learningRate}</small>
  </label>

  <label>
    <span>Batch size</span>
    <input
      type="number"
      min="1"
      data-testid="training-batch-size"
      title={PARAM_DESCRIPTIONS.batchSize}
      {disabled}
      value={store.network.training.batchSize}
      onchange={(event) => setNumber(event, 'batchSize')}
    />
    <small>{PARAM_DESCRIPTIONS.batchSize}</small>
  </label>

  {#if error}
    <p class="error" data-testid="training-error">{error}</p>
  {/if}

  {#if disabled}
    <p class="blocked" data-testid="training-blocked">
      There are problems to fix below before training. The network cannot be built until they are
      resolved.
    </p>
  {/if}

  <div class="stats" data-testid="stats-readout">
    {#if !stats}
      <p class="empty" data-testid="stats-empty">
        Not training yet. Press play to start, or step to train one batch.
      </p>
    {:else}
      <dl style="grid-template-columns: repeat({showAccuracy ? 3 : 2}, minmax(0, 1fr))">
        <div>
          <dt>Epoch</dt>
          <dd data-testid="stats-epoch">{stats.epoch}</dd>
          <small>One full pass over all the points.</small>
        </div>
        <div title={lossExplanation}>
          <dt>Loss</dt>
          <dd data-testid="stats-loss">{loss?.toFixed(3) ?? '—'}</dd>
          <small><span data-testid="stats-loss-label">{lossLabel}</span>. Lower is better.</small>
        </div>
        {#if showAccuracy}
          <div>
            <dt>Accuracy</dt>
            <dd data-testid="stats-accuracy">
              {accuracy === null ? '—' : `${(accuracy * 100).toFixed(1)}%`}
            </dd>
            <small>Share of points classified correctly, over the last completed epoch.</small>
          </div>
        {/if}
      </dl>
    {/if}
  </div>

  <LossChart points={lossPoints} />
</div>

<style>
  .training {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  select,
  input {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  .error {
    margin: 0;
    color: var(--color-error);
    font-size: var(--text-xs);
  }

  .controls {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-2);
  }

  .controls button {
    padding: var(--space-1) var(--space-2);
    text-align: center;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }

  .controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .blocked {
    margin: 0;
    color: var(--color-error);
    font-size: var(--text-sm);
  }

  .stats {
    font-size: var(--text-sm);
  }

  .stats .empty {
    margin: 0;
    color: var(--color-text-muted);
  }

  .stats dl {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
    margin: 0;
  }

  .stats dt {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
    min-height: 1.6em;
  }

  .stats dd {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-lg);
    font-variant-numeric: tabular-nums;
  }

  .stats small {
    display: block;
    color: var(--color-text-muted);
    font-size: var(--text-xs);
    min-height: 3.2em;
  }
</style>
