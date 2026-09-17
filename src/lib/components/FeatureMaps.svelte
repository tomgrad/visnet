<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import { imageAt, type ImageDataset } from '../data/images';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { probeTargetFor } from '../network/probe';
  import type { FeatureMap } from '../render/features';

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

  const THUMBNAIL = 48;
  const MAP_SIZE = 40;
  const MAP_COLUMNS = 4;
  const SQUARE = 4;
  const GAP = 1;
  const PANEL_WIDTH = 320;

  let canvas = $state<HTMLCanvasElement | null>(null);
  let features = $state<typeof import('../render/features') | null>(null);
  let digit = $state(0);
  let error = $state<string | null>(null);

  const target = $derived(probeTargetFor(store.network, store.selectedBlockId));
  const dims = $derived(target?.dims ?? []);
  const supported = $derived(dims.length === 1 || dims.length === 3);
  const message = $derived(
    !model
      ? 'Fix the problems listed in the editor before the feature maps can be drawn.'
      : !target
        ? 'Select a block to see its feature maps.'
        : !supported
          ? "This layer's output cannot be shown as feature maps."
          : !dataset || indices.length === 0
            ? 'The digit images are not loaded yet.'
            : null
  );
  const label = $derived(
    dataset && indices.length > 0 ? dataset.labels[indices[digit]] : null
  );

  $effect(() => {
    if (digit > indices.length - 1) digit = 0;
  });

  onMount(async () => {
    features = await import('../render/features');
  });

  function drawImage(
    context: CanvasRenderingContext2D,
    values: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    x: number,
    y: number,
    size: number
  ): void {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let index = 0; index < width * height; index++) {
      const value = values[index];
      rgba[index * 4] = value;
      rgba[index * 4 + 1] = value;
      rgba[index * 4 + 2] = value;
      rgba[index * 4 + 3] = 255;
    }
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offscreenContext = offscreen.getContext('2d');
    if (!offscreenContext) return;
    offscreenContext.putImageData(new ImageData(rgba, width, height), 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(offscreen, x, y, size, size);
  }

  $effect(() => {
    void redrawKey;
    const module = features;
    const element = canvas;
    const currentModel = model;
    const currentTarget = target;
    const data = dataset;
    const imageIndex = indices[digit];
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    if (
      !currentModel ||
      !currentTarget ||
      !data ||
      imageIndex === undefined ||
      !supported
    ) {
      context.clearRect(0, 0, element.width, element.height);
      error = null;
      return;
    }

    try {
      const pixels = imageAt(data, imageIndex);
      const maps = module.featureMaps(currentModel, pixels, data.rows, data.cols, currentTarget.source);
      const images = maps.length > 0 && maps[0].width > 1;
      const columns = images
        ? Math.min(MAP_COLUMNS, maps.length)
        : Math.max(1, Math.floor(PANEL_WIDTH / (SQUARE + GAP)));
      const rows = Math.max(1, Math.ceil(maps.length / columns));
      const cell = images ? MAP_SIZE + GAP : SQUARE + GAP;
      const width = columns * cell;
      const height = THUMBNAIL + GAP + rows * cell;

      const ratio = globalThis.devicePixelRatio ?? 1;
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      element.style.width = `${width}px`;
      element.style.height = `${height}px`;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);

      drawImage(context, pixels, data.cols, data.rows, 0, 0, THUMBNAIL);

      maps.forEach((map: FeatureMap, index) => {
        const x = (index % columns) * cell;
        const y = THUMBNAIL + GAP + Math.floor(index / columns) * cell;
        if (images) {
          drawImage(context, map.values, map.width, map.height, x, y, MAP_SIZE);
        } else {
          const value = map.values[0];
          context.fillStyle = `rgb(${value}, ${value}, ${value})`;
          context.fillRect(x, y, SQUARE, SQUARE);
        }
      });
      error = null;
    } catch (cause) {
      console.error(cause);
      error = 'The feature maps could not be drawn.';
    }
  });
</script>

<figure class="features" data-testid="feature-maps">
  <div class="controls">
    <button
      type="button"
      data-testid="feature-next"
      disabled={!!message || indices.length <= 1}
      onclick={() => (digit = (digit + 1) % indices.length)}
    >
      Next digit
    </button>
  </div>

  <canvas bind:this={canvas} data-testid="feature-canvas"></canvas>

  {#if message}
    <figcaption data-testid="feature-message">{message}</figcaption>
  {:else if label !== null}
    <figcaption data-testid="feature-caption">
      Digit {label}, sample {digit + 1} of {indices.length}.
    </figcaption>
  {/if}
  {#if error}
    <figcaption class="error" data-testid="feature-error">{error}</figcaption>
  {/if}
</figure>

<style>
  .features {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
  }

  .controls {
    display: flex;
    gap: var(--space-2);
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
    max-width: 100%;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg);
  }

  figcaption {
    max-width: 340px;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .error {
    color: var(--color-error);
  }
</style>
