<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import { imageAt, type ImageDataset } from '../data/images';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { probeTargetFor } from '../network/probe';
  import { BACKGROUND_RGB, classColour } from '../render/palette';
  import type { LatentBounds } from '../render/latent';
  import type { CodeSample } from '../render/codes';

  let {
    model,
    store,
    dataset,
    indices,
    redrawKey
  }: {
    model: tf.LayersModel | null;
    store: NetworkStore;
    dataset: ImageDataset | null;
    indices: number[];
    redrawKey: number;
  } = $props();

  const SIZE = 320;
  const POINT_RADIUS = 4;

  let canvas = $state<HTMLCanvasElement | null>(null);
  let codes = $state<typeof import('../render/codes') | null>(null);
  let pair = $state(0);
  let error = $state<string | null>(null);

  const target = $derived(probeTargetFor(store.network, store.selectedBlockId));
  const featureCount = $derived(target ? target.dims.reduce((total, size) => total * size, 1) : 0);
  const plottable = $derived(target !== null && target.dims.length === 1 && featureCount >= 2);
  const pairCount = $derived(Math.max(1, featureCount - 1));
  const message = $derived(
    !model
      ? 'Fix the problems listed in the editor before the code scatter can be drawn.'
      : !target
        ? 'Select a block to see its code scatter.'
        : !plottable
          ? 'This layer has fewer than two dimensions, so there is nothing to plot.'
          : !dataset || indices.length === 0
            ? 'The digit images are not loaded yet.'
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
    codes = await import('../render/codes');
  });

  function toCanvas(x: number, y: number, bounds: LatentBounds): [number, number] {
    const spanX = bounds.maxX - bounds.minX || 1;
    const spanY = bounds.maxY - bounds.minY || 1;
    const minX = bounds.minX - spanX * 0.1;
    const maxX = bounds.maxX + spanX * 0.1;
    const minY = bounds.minY - spanY * 0.1;
    const maxY = bounds.maxY + spanY * 0.1;
    return [((x - minX) / (maxX - minX)) * SIZE, SIZE - ((y - minY) / (maxY - minY)) * SIZE];
  }

  function draw(context: CanvasRenderingContext2D, sample: CodeSample): void {
    context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
    context.fillRect(0, 0, SIZE, SIZE);
    for (let index = 0; index < indices.length; index++) {
      const [px, py] = toCanvas(
        sample.points[index * 2],
        sample.points[index * 2 + 1],
        sample.bounds
      );
      context.beginPath();
      context.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      context.fillStyle = classColour(dataset ? dataset.labels[indices[index]] : 0).hex;
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = '#ffffff';
      context.stroke();
    }
  }

  $effect(() => {
    void redrawKey;
    const module = codes;
    const element = canvas;
    const currentModel = model;
    const currentTarget = target;
    const currentPlottable = plottable;
    const currentPair = pair;
    const data = dataset;
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
    context.fillRect(0, 0, SIZE, SIZE);
    if (!currentModel || !currentTarget || !currentPlottable || !data || indices.length === 0) {
      error = null;
      return;
    }

    try {
      const size = data.rows * data.cols;
      const pixels = new Uint8Array(indices.length * size);
      indices.forEach((imageIndex, row) => {
        pixels.set(imageAt(data, imageIndex), row * size);
      });
      const sample = module.codeScatter(
        currentModel,
        pixels,
        data.rows,
        data.cols,
        indices.length,
        currentTarget.source as number,
        currentPair
      );
      draw(context, sample);
      error = null;
    } catch (cause) {
      console.error(cause);
      error = 'The code scatter could not be drawn.';
    }
  });
</script>

<figure class="codes" data-testid="code-scatter" data-ready={codes ? 'true' : 'false'}>
  <div class="controls">
    <button
      type="button"
      data-testid="code-next"
      disabled={!!message || pairCount <= 1}
      onclick={() => (pair = (pair + 1) % pairCount)}
    >
      Next dimensions
    </button>
    {#if plottable}
      <span data-testid="code-label">dimensions {pair + 1} & {pair + 2} of {featureCount}</span>
    {/if}
  </div>

  <canvas bind:this={canvas} width={SIZE} height={SIZE}></canvas>

  {#if message}
    <figcaption data-testid="code-message">{message}</figcaption>
  {/if}
  {#if error}
    <figcaption class="error" data-testid="code-error">{error}</figcaption>
  {/if}
</figure>

<style>
  .codes {
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
