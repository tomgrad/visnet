# CNN Feature Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live feature-map panel to the CNN example showing the selected block's output for one test digit.

**Architecture:** Extract the `layer.apply` traversal into a shared `render/activations.ts`; add `render/features.ts` that runs it and normalises activations into grayscale `FeatureMap`s; add a `FeatureMaps.svelte` panel that cycles test digits and draws the maps.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@tensorflow/tfjs`, Vitest (`engine` in Node, `ui` in jsdom).

## Global Constraints

- `src/lib/network/**` is pure: no Svelte, no TF.js, no DOM.
- `@tensorflow/tfjs` runtime imports are confined to `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, `src/lib/render/latent.ts`, `src/lib/render/activations.ts`, `src/lib/render/features.ts`, `src/lib/persist/weights.ts`, and colocated tests. Other modules may use `import type`.
- TF.js and `@xyflow/svelte` are only reached from the browser (dynamic import or `onMount`/`$effect`); every route must prerender.
- No block stores its input dimension.
- Do not add code comments unless a non-obvious constraint requires one.
- Every validation problem carries a non-empty `title`, `message`, and `fix`.
- Verify with `npm test`, `npm run check`, `npm run lint`, `npm run build` after every task.

## Design reference

`docs/superpowers/specs/2026-09-17-visnet-cnn-feature-maps-design.md`

> **Correction (applied during execution):** Task 2's rank-3 code below indexes
> channels as contiguous `height*width` slabs (`data[channel * plane + index]`).
> That is wrong: TF.js convolution output is channels-last, so pixel `p` of
> channel `ch` is `data[p * channels + ch]`. The implemented module uses the
> strided index, with a deterministic synthetic-conv regression test. The spec's
> §7 carries the corrected contract.

---

### Task 1: Shared forward pass

**Files:**
- Create: `src/lib/render/activations.ts`
- Create: `src/lib/render/activations.test.ts`
- Modify: `src/lib/render/latent.ts` (use `forwardActivations`)

**Interfaces:**
- Consumes: `@tensorflow/tfjs`.
- Produces: `forwardActivations(model, input): tf.Tensor[]`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/render/activations.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { forwardActivations } from './activations';

let models: tf.Sequential[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('forwardActivations', () => {
  it('returns one activation per layer, matching predict', () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    const input = tf.tensor2d([[0.1, -0.2]]);

    const { count, last } = tf.tidy(() => {
      const activations = forwardActivations(model, input);
      return {
        count: activations.length,
        last: Array.from(activations[activations.length - 1].dataSync())
      };
    });
    input.dispose();

    const predicted = tf.tidy(() =>
      Array.from((model.predict(tf.tensor2d([[0.1, -0.2]])) as tf.Tensor).dataSync())
    );
    expect(count).toBe(model.layers.length);
    expect(last).toEqual(predicted);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/render/activations.test.ts`
Expected: FAIL with "Cannot find module './activations'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/render/activations.ts`:

```ts
import * as tf from '@tensorflow/tfjs';

export function forwardActivations(model: tf.LayersModel, input: tf.Tensor): tf.Tensor[] {
  const activations: tf.Tensor[] = [];
  let current: tf.Tensor = input;
  for (const layer of model.layers) {
    current = layer.apply(current) as tf.Tensor;
    activations.push(current);
  }
  return activations;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/render/activations.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Refactor `latent.ts` to use it**

In `src/lib/render/latent.ts`, add the import:

```ts
import { forwardActivations } from './activations';
```

and replace the manual loop inside `projectLatent`:

```ts
    const input = tf.tensor2d(all, [cellCount + pointCount, 2]);
    const activations: tf.Tensor[] = [];
    let current: tf.Tensor = input;
    for (const layer of model.layers) {
      current = layer.apply(current) as tf.Tensor;
      activations.push(current);
    }
```

with:

```ts
    const input = tf.tensor2d(all, [cellCount + pointCount, 2]);
    const activations = forwardActivations(model, input);
```

- [ ] **Step 6: Run the latent and render tests**

Run: `npx vitest run src/lib/render`
Expected: PASS (latent tests still green).

- [ ] **Step 7: Commit**

```bash
git add src/lib/render/activations.ts src/lib/render/activations.test.ts src/lib/render/latent.ts
git commit -m "refactor: extract the shared layer forward pass"
```

---

### Task 2: Feature render module

**Files:**
- Create: `src/lib/render/features.ts`
- Create: `src/lib/render/features.test.ts`

**Interfaces:**
- Consumes: `forwardActivations` from `./activations`; `@tensorflow/tfjs`.
- Produces: `FeatureMap`, `featureMaps(model, pixels, rows, cols, source)`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/render/features.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createCnnNetwork } from '../examples/cnn/example';
import { buildModel } from '../tf/buildModel';
import { featureMaps } from './features';

let models: tf.Sequential[] = [];

function cnn(): tf.Sequential {
  const model = buildModel(createCnnNetwork());
  models.push(model);
  return model;
}

const FLAT = new Uint8Array(28 * 28).fill(128);
const VARIED = Uint8Array.from({ length: 28 * 28 }, (_, index) => (index * 7) % 256);

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('featureMaps', () => {
  it('returns one map per convolution channel', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 0);
    expect(maps).toHaveLength(8);
    expect(maps[0].width).toBe(28);
    expect(maps[0].height).toBe(28);
    expect(maps[0].values).toHaveLength(28 * 28);
  });

  it('returns one square per dense unit', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 3);
    expect(maps).toHaveLength(10);
    expect(maps[0].width).toBe(1);
    expect(maps[0].height).toBe(1);
  });

  it('normalises a map between its own minimum and maximum', () => {
    const maps = featureMaps(cnn(), VARIED, 28, 28, 0);
    const values = Array.from(maps[0].values);
    expect(Math.min(...values)).toBe(0);
    expect(Math.max(...values)).toBe(255);
  });

  it('shows a constant map as mid-grey', () => {
    const maps = featureMaps(cnn(), FLAT, 28, 28, 'input');
    expect(Array.from(maps[0].values)).toEqual(new Array(28 * 28).fill(128));
  });

  it('does not leak tensors', () => {
    const model = cnn();
    const before = tf.memory().numTensors;
    featureMaps(model, VARIED, 28, 28, 1);
    expect(tf.memory().numTensors).toBe(before);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/render/features.test.ts`
Expected: FAIL with "Cannot find module './features'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/render/features.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { forwardActivations } from './activations';

export interface FeatureMap {
  width: number;
  height: number;
  values: Uint8ClampedArray;
}

function shade(value: number, min: number, max: number): number {
  if (max === min) return 128;
  return Math.round(((value - min) / (max - min)) * 255);
}

function extent(data: Float32Array, start: number, count: number): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let index = 0; index < count; index++) {
    const value = data[start + index];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return [min, max];
}

export function featureMaps(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  source: 'input' | number
): FeatureMap[] {
  return tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const input = tf.tensor4d(normalised, [1, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    if (source !== 'input' && (source < 0 || source >= activations.length)) return [];
    const activation = source === 'input' ? input : activations[source];
    const dims = activation.shape.slice(1);
    const data = activation.dataSync();

    if (dims.length === 3) {
      const [height, width, channels] = dims;
      const plane = height * width;
      const maps: FeatureMap[] = [];
      for (let channel = 0; channel < channels; channel++) {
        const start = channel * plane;
        const [min, max] = extent(data, start, plane);
        const values = new Uint8ClampedArray(plane);
        for (let index = 0; index < plane; index++) {
          values[index] = shade(data[start + index], min, max);
        }
        maps.push({ width, height, values });
      }
      return maps;
    }

    if (dims.length === 1) {
      const count = dims[0];
      const [min, max] = extent(data, 0, count);
      const maps: FeatureMap[] = [];
      for (let index = 0; index < count; index++) {
        maps.push({ width: 1, height: 1, values: Uint8ClampedArray.of(shade(data[index], min, max)) });
      }
      return maps;
    }

    return [];
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/render/features.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/render/features.ts src/lib/render/features.test.ts
git commit -m "feat: render a layer's activation as normalised feature maps"
```

---

### Task 3: Feature maps component

**Files:**
- Create: `src/lib/components/FeatureMaps.svelte`
- Create: `src/lib/components/FeatureMaps.test.ts`

**Interfaces:**
- Consumes: `probeTargetFor` from `../network/probe`; `imageAt`/`ImageDataset` from `../data/images`; `featureMaps`/`FeatureMap` from `../render/features` (dynamic); `NetworkStore`.
- Produces: a `FeatureMaps` component with props `{ model, store, dataset, indices, redrawKey }`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/components/FeatureMaps.test.ts`:

```ts
import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import { NetworkStore } from '../editor/networkStore.svelte';
import { createCnnNetwork } from '../examples/cnn/example';
import { featureMaps } from '../render/features';
import FeatureMaps from './FeatureMaps.svelte';

vi.mock('../render/features', () => ({
  featureMaps: vi.fn(() => [
    { width: 2, height: 2, values: new Uint8ClampedArray([0, 128, 255, 64]) }
  ])
}));

const DATASET: ImageDataset = {
  count: 3,
  rows: 2,
  cols: 2,
  numClasses: 10,
  pixels: new Uint8Array(3 * 2 * 2).fill(100),
  labels: Uint8Array.of(3, 7, 1)
};

beforeEach(() => {
  vi.mocked(featureMaps).mockClear();
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    scale: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    set fillStyle(_value: string) {},
    set imageSmoothingEnabled(_value: boolean) {}
  })) as never;
});

function storeSelecting(index: number): NetworkStore {
  const store = new NetworkStore(createCnnNetwork());
  store.select(store.network.blocks[index].id);
  return store;
}

function show(store: NetworkStore, model: unknown = {}, dataset: ImageDataset | null = DATASET) {
  render(FeatureMaps, {
    props: { model: model as never, store, dataset, indices: [0, 1, 2], redrawKey: 0 }
  });
}

describe('FeatureMaps', () => {
  it('draws the maps for a convolution block', async () => {
    show(storeSelecting(1));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.queryByTestId('feature-error')).toBeNull();
  });

  it('draws the maps for a dense block', async () => {
    vi.mocked(featureMaps).mockReturnValueOnce([
      { width: 1, height: 1, values: Uint8ClampedArray.of(200) }
    ]);
    show(storeSelecting(4));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.queryByTestId('feature-error')).toBeNull();
  });

  it('cycles the digit and wraps', async () => {
    show(storeSelecting(1));
    await waitFor(() => expect(featureMaps).toHaveBeenCalled());
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 1 of 3');
    for (let i = 0; i < 3; i++) {
      await userEvent.click(screen.getByTestId('feature-next'));
    }
    expect(screen.getByTestId('feature-caption').textContent).toContain('sample 1 of 3');
  });

  it('prompts to select a block', () => {
    show(new NetworkStore(createCnnNetwork()));
    expect(screen.getByTestId('feature-message').textContent).toContain('Select a block');
  });

  it('explains when there is no model', () => {
    show(storeSelecting(1), null);
    expect(screen.getByTestId('feature-message').textContent).toContain('Fix the problems');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/components/FeatureMaps.test.ts`
Expected: FAIL with "Cannot find module './FeatureMaps.svelte'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/components/FeatureMaps.svelte`:

```svelte
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
    values: Uint8ClampedArray,
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
    if (
      !module ||
      !element ||
      !currentModel ||
      !currentTarget ||
      !data ||
      imageIndex === undefined ||
      !supported
    ) {
      return;
    }
    const context = element.getContext('2d');
    if (!context) return;

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

<figure class="features" data-testid="feature-maps" data-ready={features ? 'true' : 'false'}>
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/components/FeatureMaps.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/FeatureMaps.svelte src/lib/components/FeatureMaps.test.ts
git commit -m "feat: add the CNN feature maps panel"
```

---

### Task 4: Wire the panel in and update the docs

**Files:**
- Modify: `src/routes/examples/cnn/+page.svelte`
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Add the panel to the CNN page**

In `src/routes/examples/cnn/+page.svelte`, add the import:

```ts
import FeatureMaps from '$lib/components/FeatureMaps.svelte';
```

and render it in the `experiment` snippet immediately after the `<SampleGrid ... />` element:

```svelte
<FeatureMaps
  model={session.model}
  {store}
  dataset={testData}
  indices={sampleIndices}
  redrawKey={session.redrawKey}
/>
```

- [ ] **Step 2: Update `AGENTS.md`**

Add `src/lib/render/activations.ts` and `src/lib/render/features.ts` to the TF.js runtime-import allow-list, keeping the existing wording and wrapping:

```
- `@tensorflow/tfjs` may only be imported at runtime by `src/lib/tf/**`,
  `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`,
  `src/lib/render/latent.ts`, `src/lib/render/activations.ts`,
  `src/lib/render/features.ts`, `src/lib/persist/weights.ts`, and the test files
  colocated with those modules. Other modules may import its types with
  `import type`; no other production module may reach it at runtime.
```

- [ ] **Step 3: Update `README.md`**

In the "What works today" list, add a bullet after the digit-grid bullet:

```
- See the feature maps of the block selected on the canvas for a test digit, with
  a button to step through the sample digits.
```

- [ ] **Step 4: Verify**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/routes/examples/cnn/+page.svelte AGENTS.md README.md
git commit -m "feat: show the feature maps panel in the CNN example"
```

---

## Self-review

- **Spec coverage:** Task 1 is spec §6; Task 2 is spec §7; Task 3 is spec §8 and §9; Task 4 is spec §5, §11, and §12. Performance (spec §8) is satisfied by reusing `redrawKey`. Testing (spec §10) maps to the three test files.
- **Placeholder scan:** No "TBD"/"handle edge cases"; every new module and test is given in full.
- **Type consistency:** `forwardActivations` defined in Task 1, used by `latent.ts` and Task 2. `FeatureMap`/`featureMaps` defined in Task 2, consumed in Task 3. Props `{ model, store, dataset, indices, redrawKey }` defined in Task 3, passed in Task 4.
- **Known risks:** (a) `forwardActivations` returns tensors the caller must tidy — Task 1's test and Task 2's tidy wrap enforce this. (b) The CNN page variable is `session` and the test dataset is `testData`, with `sampleIndices` already derived — Task 4 uses those names.
