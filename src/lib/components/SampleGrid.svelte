<script lang="ts">
  import { onDestroy } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import { imageAt, type ImageDataset } from '../data/images';
  import {
    SAMPLE_GRID_CELL_HEIGHT,
    SAMPLE_GRID_CELL_WIDTH,
    SAMPLE_GRID_COLUMNS,
    SAMPLE_GRID_GAP
  } from '../examples/cnn/example';
  import { gridCellRect, predictionsFromLogits, type GridPrediction } from '../examples/cnn/grid';

  let {
    model,
    dataset,
    indices,
    sampleXs,
    redrawKey,
    onerror
  }: {
    model: tf.LayersModel | null;
    dataset: ImageDataset | null;
    indices: number[];
    sampleXs: tf.Tensor | null;
    redrawKey: number;
    onerror?: (message: string) => void;
  } = $props();

  const IMAGE_MARGIN = 4;
  const NEUTRAL = '#cbd5e1';
  const CORRECT_COLOUR = '#10b981';
  const WRONG_COLOUR = '#dc2626';

  const rows = $derived(Math.max(1, Math.ceil(indices.length / SAMPLE_GRID_COLUMNS)));
  const width = $derived(
    SAMPLE_GRID_COLUMNS * SAMPLE_GRID_CELL_WIDTH + (SAMPLE_GRID_COLUMNS - 1) * SAMPLE_GRID_GAP
  );
  const height = $derived(
    rows * SAMPLE_GRID_CELL_HEIGHT + (rows - 1) * SAMPLE_GRID_GAP
  );

  let base: HTMLCanvasElement | null = $state(null);
  let overlay: HTMLCanvasElement | null = $state(null);
  let baseLayer: HTMLCanvasElement | null = null;
  let marks: GridPrediction[] | null = $state(null);
  let failed = $state(false);

  function prepare(element: HTMLCanvasElement | null): CanvasRenderingContext2D | null {
    if (!element) return null;
    const ratio = globalThis.devicePixelRatio ?? 1;
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    element.style.width = `${width}px`;
    element.style.height = `${height}px`;
    const context = element.getContext('2d');
    context?.scale(ratio, ratio);
    return context;
  }

  $effect(() => {
    const element = base;
    const data = dataset;
    if (!element || !data || indices.length === 0) return;

    const context = prepare(element);
    if (!context) return;

    if (!baseLayer) baseLayer = document.createElement('canvas');
    baseLayer.width = data.cols;
    baseLayer.height = data.rows;
    const layerContext = baseLayer.getContext('2d');
    if (!layerContext) return;

    const rgba = new Uint8ClampedArray(data.rows * data.cols * 4);
    const image = new ImageData(rgba, data.cols, data.rows);

    context.clearRect(0, 0, width, height);
    indices.forEach((imageIndex, cell) => {
      const rect = gridCellRect(
        cell,
        SAMPLE_GRID_COLUMNS,
        SAMPLE_GRID_CELL_WIDTH,
        SAMPLE_GRID_CELL_HEIGHT,
        SAMPLE_GRID_GAP
      );
      const pixels = imageAt(data, imageIndex);
      for (let i = 0; i < pixels.length; i++) {
        const value = pixels[i];
        rgba[i * 4] = value;
        rgba[i * 4 + 1] = value;
        rgba[i * 4 + 2] = value;
        rgba[i * 4 + 3] = 255;
      }
      layerContext.putImageData(image, 0, 0);
      context.drawImage(
        baseLayer as HTMLCanvasElement,
        rect.x + IMAGE_MARGIN,
        rect.y + IMAGE_MARGIN,
        data.cols,
        data.rows
      );
    });
  });

  $effect(() => {
    void redrawKey;
    const element = overlay;
    const data = dataset;
    if (!element || !data) return;

    const context = prepare(element);
    if (!context) return;

    let predictions: GridPrediction[] | null = null;
    let problem = false;
    const xs = sampleXs;
    if (model && xs) {
      try {
        const logits = model.predict(xs) as tf.Tensor;
        try {
          const values = logits.arraySync() as number[][];
          predictions = predictionsFromLogits(values, data.labels, indices);
        } finally {
          logits.dispose();
        }
      } catch (error) {
        console.error(error);
        onerror?.('The sample predictions could not be updated.');
        predictions = null;
        problem = true;
      }
    }

    marks = predictions;
    failed = problem;
    context.clearRect(0, 0, width, height);
    context.lineWidth = 2;
    context.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    indices.forEach((_imageIndex, cell) => {
      const rect = gridCellRect(
        cell,
        SAMPLE_GRID_COLUMNS,
        SAMPLE_GRID_CELL_WIDTH,
        SAMPLE_GRID_CELL_HEIGHT,
        SAMPLE_GRID_GAP
      );
      const prediction = predictions?.[cell];
      context.strokeStyle = prediction
        ? prediction.correct
          ? CORRECT_COLOUR
          : WRONG_COLOUR
        : NEUTRAL;
      context.strokeRect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2);

      if (prediction) {
        context.fillStyle = prediction.correct ? CORRECT_COLOUR : WRONG_COLOUR;
        context.fillText(
          String(prediction.predicted),
          rect.x + rect.width / 2,
          rect.y + rect.height - 8
        );
      }
    });
  });

  onDestroy(() => {
    baseLayer = null;
  });
</script>

<figure class="grid" data-testid="sample-grid">
  <div class="stack" style="width: {width}px; height: {height}px">
    <canvas bind:this={base}></canvas>
    <canvas bind:this={overlay}></canvas>
  </div>
  <figcaption data-testid="sample-grid-caption">
    {#if failed}
      The predictions could not be updated. They will come back once training continues.
    {:else if marks}
      Each digit shows what the network predicts; a green outline means it is right, a red one
      that it is wrong.
    {:else}
      Training has not started. These are test digits the network has not trained on; the
      predicted labels appear once it does.
    {/if}
  </figcaption>
</figure>

<style>
  .grid {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
  }

  .stack {
    position: relative;
    background: var(--color-surface);
    border-radius: var(--radius-sm);
  }

  canvas {
    position: absolute;
    inset: 0;
  }

  figcaption {
    max-width: 340px;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
