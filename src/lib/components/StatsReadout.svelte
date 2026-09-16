<script lang="ts">
  import type { TrainStats } from '../training/Trainer';

  let { stats }: { stats: TrainStats | null } = $props();

  const loss = $derived(stats ? (stats.epochMeanLoss ?? stats.batchLoss) : null);
  const lossLabel = $derived(
    stats?.epochMeanLoss !== null && stats ? 'Average loss this epoch' : 'Loss on the last batch'
  );
  const accuracy = $derived(stats?.epochAccuracy ?? null);
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
      <div>
        <dt>Loss</dt>
        <dd data-testid="stats-loss">{loss?.toFixed(3) ?? '—'}</dd>
        <small>{lossLabel}. Lower is better.</small>
      </div>
      {#if accuracy !== null}
        <div>
          <dt>Accuracy</dt>
          <dd data-testid="stats-accuracy">{(accuracy * 100).toFixed(1)}%</dd>
          <small>Share of points the network classifies correctly.</small>
        </div>
      {/if}
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
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: var(--space-3);
    margin: 0;
  }

  dt {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  dd {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-lg);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }
</style>
