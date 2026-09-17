<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import type { PointDataset } from '../data/points';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { probeTargetFor } from '../network/probe';
  import { BACKGROUND_RGB, CLASS_COLOURS } from '../render/palette';
  import type { LatentBounds, LatentSample } from '../render/latent';

  let {
    model,
    store,
    dataset,
    redrawKey
  }: {
    model: tf.LayersModel | null;
    store: NetworkStore;
    dataset: PointDataset;
    redrawKey: number;
  } = $props();

  const SIZE = 320;
  const GRID = 32;
  const POINT_RADIUS = 4;

  let canvas = $state<HTMLCanvasElement | null>(null);
  let latent = $state<typeof import('../render/latent') | null>(null);
  let pair = $state(0);
  let error = $state<string | null>(null);

  const target = $derived(probeTargetFor(store.network, store.selectedBlockId));
  const dims = $derived(target ? target.dims.reduce((total, size) => total * size, 1) : 0);
  const pairCount = $derived(Math.max(1, dims - 1));
  const message = $derived(
    !model
      ? 'Fix the problems listed in the editor before the latent space can be drawn.'
      : !target
        ? 'Select a block to see its latent space.'
        : dims < 2
          ? 'This layer has fewer than two dimensions, so there is nothing to plot.'
          : null
  );

  $effect(() => {
    void store.selectedBlockId;
    pair = 0;
  });

  $effect(() => {
    if (pair > pairCount - 1) pair = pairCount - 1;
  });

  onMount(async () => {
    latent = await import('../render/latent');
  });

  function toCanvas(x: number, y: number, bounds: LatentBounds): [number, number] {
    const spanX = bounds.maxX - bounds.minX || 1;
    const spanY = bounds.maxY - bounds.minY || 1;
    const minX = bounds.minX - spanX * 0.05;
    const maxX = bounds.maxX + spanX * 0.05;
    const minY = bounds.minY - spanY * 0.05;
    const maxY = bounds.maxY + spanY * 0.05;
    return [
      ((x - minX) / (maxX - minX)) * SIZE,
      SIZE - ((y - minY) / (maxY - minY)) * SIZE
    ];
  }

  function draw(context: CanvasRenderingContext2D, sample: LatentSample): void {
    const cell = SIZE / GRID;
    for (let index = 0; index < GRID * GRID; index++) {
      const [px, py] = toCanvas(sample.grid[index * 2], sample.grid[index * 2 + 1], sample.bounds);
      context.fillStyle = CLASS_COLOURS[sample.gridClasses[index]].hex;
      context.fillRect(px - cell / 2, py - cell / 2, cell, cell);
    }

    context.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    context.lineWidth = 0.5;
    context.beginPath();
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const index = gy * GRID + gx;
        const [x0, y0] = toCanvas(sample.grid[index * 2], sample.grid[index * 2 + 1], sample.bounds);
        if (gx + 1 < GRID) {
          const right = index + 1;
          const [x1, y1] = toCanvas(
            sample.grid[right * 2],
            sample.grid[right * 2 + 1],
            sample.bounds
          );
          context.moveTo(x0, y0);
          context.lineTo(x1, y1);
        }
        if (gy + 1 < GRID) {
          const below = index + GRID;
          const [x1, y1] = toCanvas(
            sample.grid[below * 2],
            sample.grid[below * 2 + 1],
            sample.bounds
          );
          context.moveTo(x0, y0);
          context.lineTo(x1, y1);
        }
      }
    }
    context.stroke();

    for (let index = 0; index < dataset.points.length; index++) {
      const [px, py] = toCanvas(
        sample.points[index * 2],
        sample.points[index * 2 + 1],
        sample.bounds
      );
      context.beginPath();
      context.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      context.fillStyle = CLASS_COLOURS[dataset.points[index].label].hex;
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = '#ffffff';
      context.stroke();
    }
  }

  $effect(() => {
    void redrawKey;
    const module = latent;
    const element = canvas;
    const currentModel = model;
    const currentTarget = target;
    const currentPair = pair;
    const currentDims = dims;
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
    context.fillRect(0, 0, SIZE, SIZE);
    if (!currentModel || !currentTarget || currentDims < 2) return;

    try {
      const cells = module.gridInputs(GRID);
      const points = new Float32Array(dataset.points.flatMap((point) => [point.x, point.y]));
      const sample = module.projectLatent(
        currentModel,
        cells,
        points,
        currentTarget.source,
        currentPair
      );
      draw(context, sample);
      error = null;
    } catch (cause) {
      console.error(cause);
      error = 'The latent space could not be drawn.';
    }
  });
</script>

<figure class="latent" data-testid="latent-space" data-ready={latent ? 'true' : 'false'}>
  <div class="controls">
    <button
      type="button"
      data-testid="latent-next"
      disabled={!!message || pairCount <= 1}
      onclick={() => (pair = (pair + 1) % pairCount)}
    >
      Next dimensions
    </button>
    {#if target && dims >= 2}
      <span data-testid="latent-label">dimensions {pair + 1} & {pair + 2} of {dims}</span>
    {/if}
  </div>

  <canvas bind:this={canvas} width={SIZE} height={SIZE} data-testid="latent-canvas"></canvas>

  {#if message}
    <figcaption data-testid="latent-message">{message}</figcaption>
  {/if}
  {#if error}
    <figcaption class="error" data-testid="latent-error">{error}</figcaption>
  {/if}
</figure>

<style>
  .latent {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .controls button {
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }

  .controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  canvas {
    width: 100%;
    max-width: 360px;
    aspect-ratio: 1;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg);
  }

  figcaption {
    max-width: 360px;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .error {
    color: var(--color-error);
  }
</style>
