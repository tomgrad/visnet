<script lang="ts">
  import type { TrainStats } from '../training/Trainer';

  let { stats }: { stats: TrainStats | null } = $props();

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
</script>

<div class="stats" data-testid="stats-readout">
  {#if !stats}
    <p class="empty" data-testid="stats-empty">
      Not training yet. Press play to start, or step to train one batch.
    </p>
  {:else}
    <dl>
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
      <div>
        <dt>Accuracy</dt>
        <dd data-testid="stats-accuracy">
          {accuracy === null ? '—' : `${(accuracy * 100).toFixed(1)}%`}
        </dd>
        <small>Share of points classified correctly, over the last completed epoch.</small>
      </div>
    </dl>
  {/if}
</div>

<style>
  .stats {
    font-size: var(--text-sm);
  }

  .empty {
    margin: 0;
    color: var(--color-text-muted);
  }

  dl {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
    margin: 0;
  }

  dt {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
    min-height: 1.6em;
  }

  dd {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-lg);
    font-variant-numeric: tabular-nums;
  }

  small {
    display: block;
    color: var(--color-text-muted);
    font-size: var(--text-xs);
    min-height: 3.2em;
  }
</style>
