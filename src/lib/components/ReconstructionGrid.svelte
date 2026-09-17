<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import { imageAt, type ImageDataset } from '../data/images';

  let {
    model,
    dataset,
    indices,
    redrawKey
  }: {
    model: tf.LayersModel | null;
    dataset: ImageDataset | null;
    indices: number[];
    redrawKey: number;
  } = $props();

  const CELL = 48;
  const GAP = 4;
  const PADDING = 6;

  let canvas = $state<HTMLCanvasElement | null>(null);
  let reconstruction = $state<typeof import('../render/reconstruction') | null>(null);
  let error = $state<string | null>(null);

  const message = $derived(
    !model
      ? 'Fix the problems listed in the editor before the reconstructions can be drawn.'
      : !dataset || indices.length === 0
        ? 'The digit images are not loaded yet.'
        : null
  );

  onMount(async () => {
    reconstruction = await import('../render/reconstruction');
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
    const module = reconstruction;
    const element = canvas;
    const currentModel = model;
    const data = dataset;
    const samples = indices;
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    const columns = Math.max(1, samples.length);
    const width = columns * CELL + (columns - 1) * GAP;
    const height = 2 * CELL + GAP;
    const ratio = globalThis.devicePixelRatio ?? 1;
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    element.style.width = `${width}px`;
    element.style.height = `${height}px`;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    if (!currentModel || !data || samples.length === 0) {
      error = null;
      return;
    }

    try {
      const size = data.rows * data.cols;
      const pixels = new Uint8Array(samples.length * size);
      samples.forEach((imageIndex, column) => {
        pixels.set(imageAt(data, imageIndex), column * size);
      });
      const reconstructions = module.reconstruct(
        currentModel,
        pixels,
        data.rows,
        data.cols,
        samples.length
      );
      const imageSize = CELL - 2 * PADDING;
      samples.forEach((imageIndex, column) => {
        const x = column * (CELL + GAP) + PADDING;
        drawImage(context, imageAt(data, imageIndex), data.cols, data.rows, x, PADDING, imageSize);
        drawImage(
          context,
          reconstructions.subarray(column * size, (column + 1) * size),
          data.cols,
          data.rows,
          x,
          CELL + GAP + PADDING,
          imageSize
        );
      });
      error = null;
    } catch (cause) {
      console.error(cause);
      error = 'The reconstructions could not be drawn.';
    }
  });
</script>

<figure class="reconstructions" data-testid="reconstruction-grid">
  <canvas bind:this={canvas}></canvas>

  {#if message}
    <figcaption data-testid="reconstruction-message">{message}</figcaption>
  {:else}
    <figcaption>Originals on top, the network's reconstructions below.</figcaption>
  {/if}
  {#if error}
    <figcaption class="error" data-testid="reconstruction-error">{error}</figcaption>
  {/if}
</figure>

<style>
  .reconstructions {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
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
