<script lang="ts">
  let { points, width = 320, height = 120 }: { points: number[]; width?: number; height?: number } =
    $props();

  const WINDOW = 200;
  const view = $derived(points.slice(-WINDOW));
  const lowest = $derived(view.length ? Math.min(...view) : 0);
  const highest = $derived(view.length ? Math.max(...view) : 1);
  const latest = $derived(view.length ? view[view.length - 1] : null);

  const path = $derived.by(() => {
    if (view.length < 2) return '';
    const span = highest - lowest || 1;
    return view
      .map((value, index) => {
        const x = (index / (view.length - 1)) * width;
        const y = height - ((value - lowest) / span) * height;
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });
</script>

<figure class="chart" data-testid="loss-chart">
  <figcaption>
    Loss per epoch
    {#if latest !== null}
      <span data-testid="loss-chart-latest">latest {latest.toFixed(3)}</span>
    {/if}
  </figcaption>

  {#if view.length < 2}
    <p class="empty" data-testid="loss-chart-empty">
      Press play to start training. The loss curve appears after the first epoch.
    </p>
  {:else}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss per epoch">
      <path data-testid="loss-chart-path" d={path} fill="none" stroke="var(--color-accent)" />
    </svg>
    <div class="axis">
      <span>lowest {lowest.toFixed(3)}</span>
      <span>highest {highest.toFixed(3)}</span>
    </div>
  {/if}
</figure>

<style>
  .chart {
    margin: 0;
    display: grid;
    gap: var(--space-1);
  }

  figcaption {
    display: flex;
    justify-content: space-between;
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .empty {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  svg {
    width: 100%;
    height: auto;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
  }

  .axis {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
