<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import type { PointDataset } from '../data/points';
  import { BACKGROUND_RGB, CLASS_COLOURS } from '../render/palette';

  let {
    model,
    dataset,
    selectedLabel,
    onaddpoint,
    caption = 'Each coloured area is the class the network predicts at that spot. Click to add a point.'
  }: {
    model: tf.LayersModel | null;
    dataset: PointDataset;
    selectedLabel: 0 | 1;
    onaddpoint: (x: number, y: number) => void;
    caption?: string;
  } = $props();

  const SIZE = 320;
  const POINT_RADIUS = 4;
  const COLOURS = CLASS_COLOURS.map((colour) => colour.rgb);

  let canvas: HTMLCanvasElement | null = $state(null);
  let boundary = $state<typeof import('../render/boundary') | null>(null);

  onMount(async () => {
    boundary = await import('../render/boundary');
  });

  $effect(() => {
    const module = boundary;
    const element = canvas;
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    const { GRID_SIZE, classesToRgba, sampleGrid } = module;

    if (model) {
      const classes = sampleGrid(model, GRID_SIZE);
      const rgba = classesToRgba(classes, COLOURS, GRID_SIZE);
      const offscreen = document.createElement('canvas');
      offscreen.width = GRID_SIZE;
      offscreen.height = GRID_SIZE;
      const offscreenContext = offscreen.getContext('2d');
      if (offscreenContext) {
        offscreenContext.putImageData(
          new ImageData(new Uint8ClampedArray(rgba), GRID_SIZE, GRID_SIZE),
          0,
          0
        );
        context.imageSmoothingEnabled = true;
        context.clearRect(0, 0, SIZE, SIZE);
        context.drawImage(offscreen, 0, 0, SIZE, SIZE);
      }
    } else {
      context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
      context.fillRect(0, 0, SIZE, SIZE);
    }

    for (const point of dataset.points) {
      const px = ((point.x + 1) / 2) * SIZE;
      const py = (1 - (point.y + 1) / 2) * SIZE;
      context.beginPath();
      context.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      context.fillStyle = CLASS_COLOURS[point.label].hex;
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = '#ffffff';
      context.stroke();
    }
  });

  function handleClick(event: MouseEvent): void {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const { x, y } = boundary
      ? boundary.clientToDomain(event.clientX, event.clientY, rect)
      : {
          x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
          y: 1 - ((event.clientY - rect.top) / rect.height) * 2
        };
    onaddpoint(x, y);
  }
</script>

<figure class="boundary" data-testid="decision-boundary">
  <canvas
    bind:this={canvas}
    width={SIZE}
    height={SIZE}
    data-testid="boundary-canvas"
    onclick={handleClick}
  ></canvas>
  <figcaption data-testid="boundary-caption">
    {caption}
    <span class="hint">Adding class {selectedLabel} points.</span>
  </figcaption>
</figure>

<style>
  .boundary {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
  }

  canvas {
    width: 100%;
    max-width: 360px;
    aspect-ratio: 1;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    cursor: crosshair;
    background: var(--color-bg);
  }

  figcaption {
    max-width: 360px;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .hint {
    display: block;
    font-weight: 600;
    color: var(--color-text);
  }
</style>
