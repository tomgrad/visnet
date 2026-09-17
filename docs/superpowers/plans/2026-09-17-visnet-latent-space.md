# Latent Space View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live 2D latent-space view of the block selected on the canvas to the MLP example.

**Architecture:** A pure helper maps a selected block to an activation source and dimensions. A TF.js render module runs a manual forward pass over the model's own layers (`layer.apply`) to collect activations — no second model, so the training weights are never touched. A Svelte component projects the selected layer onto a cycled dimension pair and draws a warped mesh, class-coloured cells, and the dataset points, auto-fitting every frame.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@tensorflow/tfjs`, Vitest (`engine` in Node, `ui` in jsdom).

## Global Constraints

- `src/lib/network/**` is pure: no Svelte, no TF.js, no DOM, runs in plain Node.
- `@tensorflow/tfjs` runtime imports are confined to `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, `src/lib/render/latent.ts`, `src/lib/persist/weights.ts`, and colocated tests. Other modules may use `import type`.
- TF.js and `@xyflow/svelte` are only reached from the browser (dynamic import or `onMount`/`$effect`); every route must prerender.
- No block stores its input dimension.
- Do not add code comments unless a non-obvious constraint requires one.
- Every validation problem carries a non-empty `title`, `message`, and `fix`.
- Verify with `npm test`, `npm run check`, `npm run lint`, `npm run build` after every task.

## Design reference

`docs/superpowers/specs/2026-09-17-visnet-latent-space-design.md`

---

### Task 1: Pure probe target mapping

**Files:**
- Create: `src/lib/network/probe.ts`
- Create: `src/lib/network/probe.test.ts`

**Interfaces:**
- Consumes: `inferShapes` from `./inferShapes`, `Network` from `./types`.
- Produces: `ProbeTarget`, `probeTargetFor(net, blockId)`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/network/probe.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import { probeTargetFor } from './probe';
import type { Network } from './types';

const net = createEmptyNetwork();
const TRAINING = net.training;

describe('probeTargetFor', () => {
  it('maps the input block to the raw input', () => {
    expect(probeTargetFor(net, net.blocks[0].id)).toEqual({ source: 'input', dims: [2] });
  });

  it('maps a real layer to its model layer index and output shape', () => {
    expect(probeTargetFor(net, net.blocks[1].id)).toEqual({ source: 0, dims: [8] });
    expect(probeTargetFor(net, net.blocks[2].id)).toEqual({ source: 1, dims: [8] });
    expect(probeTargetFor(net, net.blocks[3].id)).toEqual({ source: 2, dims: [2] });
  });

  it('maps the output marker to the last real layer', () => {
    expect(probeTargetFor(net, net.blocks[5].id)).toEqual({ source: 3, dims: [2] });
  });

  it('returns null without a selection or for an unknown id', () => {
    expect(probeTargetFor(net, null)).toBeNull();
    expect(probeTargetFor(net, 'nope')).toBeNull();
  });

  it('maps the input but returns null for the output when there are no real layers', () => {
    const bare: Network = {
      version: 2,
      blocks: [net.blocks[0], net.blocks[5]],
      training: TRAINING,
      positions: {}
    };
    expect(probeTargetFor(bare, bare.blocks[0].id)).toEqual({ source: 'input', dims: [2] });
    expect(probeTargetFor(bare, bare.blocks[1].id)).toBeNull();
  });

  it('reports unknown dimensions when an earlier block is invalid', () => {
    const broken: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'l', kind: 'linear', units: 8 },
        { id: 'out', kind: 'output', units: 2 }
      ],
      training: TRAINING,
      positions: {}
    };
    expect(probeTargetFor(broken, 'l')).toEqual({ source: 0, dims: [] });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/network/probe.test.ts`
Expected: FAIL with "Cannot find module './probe'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/network/probe.ts`:

```ts
import { inferShapes } from './inferShapes';
import type { Network } from './types';

export interface ProbeTarget {
  source: 'input' | number;
  dims: number[];
}

export function probeTargetFor(net: Network, blockId: string | null): ProbeTarget | null {
  if (blockId === null) return null;
  const index = net.blocks.findIndex((block) => block.id === blockId);
  if (index === -1) return null;

  const { perBlock } = inferShapes(net);
  const block = net.blocks[index];

  if (block.kind === 'input') {
    return { source: 'input', dims: perBlock[index].outShape ?? [] };
  }

  const realBlocks = net.blocks.filter(
    (candidate) => candidate.kind !== 'input' && candidate.kind !== 'output'
  );
  if (realBlocks.length === 0) return null;

  const realIndex =
    block.kind === 'output'
      ? realBlocks.length - 1
      : realBlocks.findIndex((candidate) => candidate.id === block.id);
  if (realIndex === -1) return null;

  const realBlockIndex = net.blocks.findIndex(
    (candidate) => candidate.id === realBlocks[realIndex].id
  );
  return { source: realIndex, dims: perBlock[realBlockIndex].outShape ?? [] };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/network/probe.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/probe.ts src/lib/network/probe.test.ts
git commit -m "feat: map a selected block to a latent-space probe target"
```

---

### Task 2: Latent render module

**Files:**
- Create: `src/lib/render/latent.ts`
- Create: `src/lib/render/latent.test.ts`

**Interfaces:**
- Consumes: `@tensorflow/tfjs`; `buildModel` from `../tf/buildModel` and `createEmptyNetwork` from `../network/factory` (tests only).
- Produces: `LATENT_GRID_SIZE`, `LatentBounds`, `LatentSample`, `gridInputs(size?)`, `projectLatent(model, cells, points, source, dimA)`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/render/latent.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { LATENT_GRID_SIZE, gridInputs, projectLatent } from './latent';

let models: tf.Sequential[] = [];

function model(): tf.Sequential {
  const built = buildModel(createEmptyNetwork());
  models.push(built);
  return built;
}

const POINTS = new Float32Array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6]);

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((entry) => entry.dispose());
  models = [];
});

describe('gridInputs', () => {
  it('builds a size-by-size grid of coordinate pairs', () => {
    const cells = gridInputs(4);
    expect(cells).toHaveLength(4 * 4 * 2);
    expect(cells[0]).toBeCloseTo(-0.75, 5);
    expect(cells[1]).toBeCloseTo(0.75, 5);
  });
});

describe('projectLatent', () => {
  it('projects the grid and points onto the chosen dimensions', () => {
    const sample = projectLatent(model(), gridInputs(LATENT_GRID_SIZE), POINTS, 0, 0);
    expect(sample.grid).toHaveLength(LATENT_GRID_SIZE * LATENT_GRID_SIZE * 2);
    expect(sample.gridClasses).toHaveLength(LATENT_GRID_SIZE * LATENT_GRID_SIZE);
    expect(sample.points).toHaveLength(POINTS.length);
    expect(sample.bounds.maxX).toBeGreaterThanOrEqual(sample.bounds.minX);
  });

  it('returns the raw input coordinates for the input source', () => {
    const cells = gridInputs(4);
    const sample = projectLatent(model(), cells, POINTS, 'input', 0);
    expect(Array.from(sample.grid.slice(0, 2))).toEqual([cells[0], cells[1]]);
  });

  it('colours each grid cell by the model argmax', () => {
    const built = model();
    const cells = gridInputs(4);
    const sample = projectLatent(built, cells, POINTS, 0, 0);
    const expected = tf.tidy(() => {
      const input = tf.tensor2d(cells, [cells.length / 2, 2]);
      const logits = built.predict(input) as tf.Tensor;
      return Array.from(tf.argMax(logits, 1).dataSync());
    });
    expect(Array.from(sample.gridClasses)).toEqual(expected);
  });

  it('leaves the training model usable and does not leak tensors', () => {
    const built = model();
    const before = tf.memory().numTensors;
    projectLatent(built, gridInputs(8), POINTS, 1, 0);
    expect(tf.memory().numTensors).toBe(before);
    const prediction = tf.tidy(() => built.predict(tf.tensor2d([[0.1, 0.2]])) as tf.Tensor);
    expect(prediction.shape).toEqual([1, 2]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/render/latent.test.ts`
Expected: FAIL with "Cannot find module './latent'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/render/latent.ts`:

```ts
import * as tf from '@tensorflow/tfjs';

export const LATENT_GRID_SIZE = 32;

export interface LatentBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface LatentSample {
  grid: Float32Array;
  gridClasses: Int32Array;
  points: Float32Array;
  bounds: LatentBounds;
}

export function gridInputs(size: number = LATENT_GRID_SIZE): Float32Array {
  const cells = new Float32Array(size * size * 2);
  for (let gy = 0; gy < size; gy++) {
    for (let gx = 0; gx < size; gx++) {
      const offset = (gy * size + gx) * 2;
      cells[offset] = ((gx + 0.5) / size) * 2 - 1;
      cells[offset + 1] = 1 - ((gy + 0.5) / size) * 2;
    }
  }
  return cells;
}

function boundsOf(arrays: Float32Array[]): LatentBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const array of arrays) {
    for (let index = 0; index < array.length; index += 2) {
      const x = array[index];
      const y = array[index + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

export function projectLatent(
  model: tf.LayersModel,
  cells: Float32Array,
  points: Float32Array,
  source: 'input' | number,
  dimA: number
): LatentSample {
  const cellCount = cells.length / 2;
  const pointCount = points.length / 2;
  const all = new Float32Array(cells.length + points.length);
  all.set(cells, 0);
  all.set(points, cells.length);

  const result = tf.tidy(() => {
    const input = tf.tensor2d(all, [cellCount + pointCount, 2]);
    const activations: tf.Tensor[] = [];
    let current: tf.Tensor = input;
    for (const layer of model.layers) {
      current = layer.apply(current) as tf.Tensor;
      activations.push(current);
    }

    const final = activations[activations.length - 1];
    const gridClasses = Int32Array.from(
      tf.argMax(final, 1).dataSync().slice(0, cellCount)
    );

    let coords: Float32Array;
    if (source === 'input') {
      coords = all.slice();
    } else {
      const activation = activations[source];
      const a = activation.slice([0, dimA], [cellCount + pointCount, 1]);
      const b = activation.slice([0, dimA + 1], [cellCount + pointCount, 1]);
      coords = Float32Array.from(tf.concat([a, b], 1).dataSync());
    }

    return {
      grid: coords.slice(0, cellCount * 2),
      gridClasses,
      points: coords.slice(cellCount * 2)
    };
  });

  return { ...result, bounds: boundsOf([result.grid, result.points]) };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/render/latent.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/render/latent.ts src/lib/render/latent.test.ts
git commit -m "feat: project a layer's activations onto a 2D dimension pair"
```

---

### Task 3: Latent space component

**Files:**
- Create: `src/lib/components/LatentSpace.svelte`
- Create: `src/lib/components/LatentSpace.test.ts`

**Interfaces:**
- Consumes: `probeTargetFor` (Task 1), `gridInputs`/`projectLatent`/`LatentSample` (Task 2), `CLASS_COLOURS`/`BACKGROUND_RGB` from `../render/palette`, `PointDataset` from `../data/points`, `NetworkStore` from `../editor/networkStore.svelte`.
- Produces: a `LatentSpace` component with props `{ model, store, dataset, redrawKey }`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/components/LatentSpace.test.ts`:

```ts
import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { PointDataset } from '../data/points';
import { NetworkStore } from '../editor/networkStore.svelte';
import LatentSpace from './LatentSpace.svelte';

vi.mock('../render/latent', () => ({
  LATENT_GRID_SIZE: 32,
  gridInputs: (size: number) => new Float32Array(size * size * 2),
  projectLatent: () => ({
    grid: new Float32Array(32 * 32 * 2),
    gridClasses: new Int32Array(32 * 32),
    points: new Float32Array(4),
    bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 }
  })
}));

const DATASET: PointDataset = {
  points: [
    { x: 0, y: 0, label: 0 },
    { x: 0.5, y: 0.5, label: 1 }
  ],
  numClasses: 2
};

function selectedStore(index = 1): NetworkStore {
  const store = new NetworkStore();
  store.select(store.network.blocks[index].id);
  return store;
}

function show(store: NetworkStore, model: unknown = {}): void {
  render(LatentSpace, {
    props: { model: model as never, store, dataset: DATASET, redrawKey: 0 }
  });
}

describe('LatentSpace', () => {
  it('shows the dimension label for the selected layer', async () => {
    show(selectedStore(1));
    await waitFor(() =>
      expect(screen.getByTestId('latent-label').textContent).toContain('dimensions 1 & 2 of 8')
    );
  });

  it('cycles the dimension pair and wraps', async () => {
    show(selectedStore(1));
    await waitFor(() => screen.getByTestId('latent-next'));
    for (let i = 0; i < 7; i++) {
      await userEvent.click(screen.getByTestId('latent-next'));
    }
    expect(screen.getByTestId('latent-label').textContent).toContain('dimensions 1 & 2 of 8');
  });

  it('prompts to select a block', () => {
    show(new NetworkStore());
    expect(screen.getByTestId('latent-message').textContent).toContain('Select a block');
  });

  it('explains when there is no model', () => {
    show(selectedStore(1), null);
    expect(screen.getByTestId('latent-message').textContent).toContain('Fix the problems');
  });

  it('explains when the layer has fewer than two dimensions', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[1].id, { units: 1 });
    store.select(store.network.blocks[1].id);
    show(store);
    await waitFor(() =>
      expect(screen.getByTestId('latent-message').textContent).toContain('fewer than two')
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/components/LatentSpace.test.ts`
Expected: FAIL with "Cannot find module './LatentSpace.svelte'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/components/LatentSpace.svelte`:

```svelte
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
  const dims = $derived(target?.dims ?? []);
  const pairCount = $derived(Math.max(1, dims.length - 1));
  const message = $derived(
    !model
      ? 'Fix the problems listed in the editor before the latent space can be drawn.'
      : !target
        ? 'Select a block to see its latent space.'
        : dims.length < 2
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
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
    context.fillRect(0, 0, SIZE, SIZE);
    if (!currentModel || !currentTarget || currentTarget.dims.length < 2) return;

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
    {#if target && dims.length >= 2}
      <span data-testid="latent-label">dimensions {pair + 1} & {pair + 2} of {dims.length}</span>
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/components/LatentSpace.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/LatentSpace.svelte src/lib/components/LatentSpace.test.ts
git commit -m "feat: add the latent space panel"
```

---

### Task 4: Wire the panel in and update the docs

**Files:**
- Modify: `src/routes/examples/mlp/+page.svelte`
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `LatentSpace` (Task 3), the page's `session` controller and `datasetStore`.

- [ ] **Step 1: Add the panel to the MLP page**

In `src/routes/examples/mlp/+page.svelte`, add the import:

```ts
import LatentSpace from '$lib/components/LatentSpace.svelte';
```

and render it in the `experiment` snippet immediately after the `<DecisionBoundary ... />` element:

```svelte
<LatentSpace
  model={session.model}
  {store}
  dataset={datasetStore.dataset}
  redrawKey={session.redrawKey}
/>
```

- [ ] **Step 2: Update `AGENTS.md`**

Add `src/lib/render/latent.ts` to the TF.js runtime-import allow-list, and to the shipped-modules list if `render/` is named there. The rule must read (order the list as in the file):

```
- `@tensorflow/tfjs` may only be imported at runtime by `src/lib/tf/**`,
  `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`,
  `src/lib/render/latent.ts`, `src/lib/persist/weights.ts`, and the test files
  colocated with those modules. Other modules may import its types with
  `import type`; no other production module may reach it at runtime.
```

- [ ] **Step 3: Update `README.md`**

In the "What works today" list, add a bullet after the decision-boundary bullet:

```
- See the latent space of the block selected on the canvas: the input grid warped
  through that layer, with a button to cycle which pair of dimensions is shown.
```

- [ ] **Step 4: Verify**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/routes/examples/mlp/+page.svelte AGENTS.md README.md
git commit -m "feat: show the latent space panel in the 2D example"
```

---

## Self-review

- **Spec coverage:** Task 1 covers the probe target mapping (spec §6); Task 2 covers the render module (spec §7); Task 3 covers the component, cycler, drawing, auto-fit, and error states (spec §8, §9); Task 4 covers wiring, the AGENTS allow-list, and the README (spec §5, §12). Performance (spec §10) is satisfied by reusing `redrawKey` and a 32×32 grid. Testing (spec §11) maps to the three test files.
- **Placeholder scan:** No "TBD"/"handle edge cases" steps; every new module and test is given in full.
- **Type consistency:** `ProbeTarget`/`probeTargetFor` defined in Task 1, consumed in Task 3. `LatentSample`/`LatentBounds`/`gridInputs`/`projectLatent` defined in Task 2, consumed in Task 3. Props `{ model, store, dataset, redrawKey }` defined in Task 3, passed in Task 4.
- **Known risk:** `layer.apply` is called on layers already belonging to the training model. This was verified empirically to be safe: it does not leak tensors, does not grow `inboundNodes`, and leaves the model usable. Task 2's test pins this.
