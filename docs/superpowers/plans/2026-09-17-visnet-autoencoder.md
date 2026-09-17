# Autoencoder Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third example, `/examples/autoencoder`, that trains an autoencoder on MNIST with dense and convolutional presets and visualizes the reconstruction and the selected layer's code.

**Architecture:** Introduce a `task: 'classification' | 'reconstruction'` distinction (store, validation, trainer, training panel), add a reconstruction tensor builder and two TF.js render modules, add two components, and wire a new page that reuses the shared experiment controller.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@tensorflow/tfjs`, Vitest (`engine` in Node, `ui` in jsdom).

## Global Constraints

- `src/lib/network/**` is pure: no Svelte, no TF.js, no DOM.
- `@tensorflow/tfjs` runtime imports confined to `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, `src/lib/render/latent.ts`, `src/lib/render/activations.ts`, `src/lib/render/features.ts`, `src/lib/render/reconstruction.ts`, `src/lib/render/codes.ts`, `src/lib/persist/weights.ts`, and colocated tests.
- TF.js and `@xyflow/svelte` are only reached from the browser (dynamic import or `onMount`/`$effect`); every route must prerender.
- No block stores its input dimension.
- Every validation problem carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.
- Verify with `npm test`, `npm run check`, `npm run lint`, `npm run build` after every task.

## Design reference

`docs/superpowers/specs/2026-09-17-visnet-autoencoder-design.md`

---

### Task 1: The reconstruction task model

**Files:**
- Modify: `src/lib/editor/networkStore.svelte.ts`, `src/lib/network/problems.ts`, `src/lib/training/Trainer.ts`, `src/lib/examples/runtime.ts`, `src/lib/examples/experiment.svelte.ts`, `src/lib/components/TrainingPanel.svelte`
- Modify tests: `src/lib/editor/networkStore.svelte.test.ts`, `src/lib/network/problems.test.ts`, `src/lib/training/Trainer.test.ts`, `src/lib/components/TrainingPanel.test.ts`

**Interfaces:**
- Produces: `NetworkStore.task`, `ProblemOptions.task`, `Trainer` `computeAccuracy`, `createTrainer(..., computeAccuracy)`, `createExperiment({ task })`, `TrainingPanel` `showAccuracy`.

- [ ] **Step 1: Write the failing tests**

In `src/lib/editor/networkStore.svelte.test.ts`:

```ts
  it('defaults to the classification task', () => {
    expect(new NetworkStore().task).toBe('classification');
  });

  it('suppresses the image-without-convolution warning for reconstruction', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    expect(store.warnings.some((w) => w.title === 'Image input without a Convolution layer')).toBe(
      true
    );
    store.task = 'reconstruction';
    expect(store.warnings.some((w) => w.title === 'Image input without a Convolution layer')).toBe(
      false
    );
  });
```

In `src/lib/network/problems.test.ts`:

```ts
  it('skips the image-without-convolution warning for reconstruction', () => {
    const network = net([IMAGE_INPUT, { id: 'dense', kind: 'linear', units: 2 }, OUTPUT]);
    const titles = (task: 'classification' | 'reconstruction') =>
      findProblems(network, { task }).map((problem) => problem.title);
    expect(titles('classification')).toContain('Image input without a Convolution layer');
    expect(titles('reconstruction')).not.toContain('Image input without a Convolution layer');
  });
```

In `src/lib/training/Trainer.test.ts`:

```ts
  it('omits accuracy when computeAccuracy is false', async () => {
    const model = buildModel(createEmptyNetwork());
    models.push(model);
    const stats: TrainStats[] = [];
    const trainer = new Trainer(model, makeData(), 4, (s) => stats.push(s), undefined, undefined, false);
    await trainer.step();
    await trainer.step();
    expect(stats[1].epochAccuracy).toBeNull();
  });
```

In `src/lib/components/TrainingPanel.test.ts`:

```ts
  it('hides the accuracy cell when showAccuracy is false', () => {
    panel({
      showAccuracy: false,
      stats: { epoch: 1, batch: 0, batchLoss: 0.1, epochMeanLoss: 0.2, epochAccuracy: 0.9 }
    });
    expect(screen.queryByTestId('stats-accuracy')).toBeNull();
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/editor/networkStore.svelte.test.ts src/lib/network/problems.test.ts src/lib/training/Trainer.test.ts src/lib/components/TrainingPanel.test.ts`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement**

`networkStore.svelte.ts`: add `task = $state<'classification' | 'reconstruction'>('classification');` and pass it into the `issues` derived:

```ts
  issues = $derived(
    validate(this.network, {
      expectedClasses: this.expectedClasses,
      expectedInputShape: this.expectedInputShape,
      task: this.task
    })
  );
```

(Use the module's existing `findProblems` import.)

`problems.ts`: add to `ProblemOptions`:

```ts
  task?: 'classification' | 'reconstruction';
```

and change the image-input warning condition to also require the task not to be reconstruction:

```ts
    if (inputKind === 'image' && !hasConvolution && options.task !== 'reconstruction') {
```

`Trainer.ts`: add a constructor parameter after `onError`:

```ts
    private readonly computeAccuracy: boolean = true
```

and in the epoch branch use it:

```ts
        const accuracy = this.computeAccuracy ? this.accuracy() : null;
```

`runtime.ts`: extend `Runtime.createTrainer` and its implementation to take and forward `computeAccuracy`:

```ts
    createTrainer: (model, data, batchSize, onStats, onError, computeAccuracy) =>
      new trainerModule.Trainer(model, data, batchSize, onStats, undefined, onError, computeAccuracy),
```

and add the parameter to the `Runtime` interface's `createTrainer` signature.

`experiment.svelte.ts`: add `task?: 'classification' | 'reconstruction'` to the options, default `'classification'`, and pass `computeAccuracy: options.task !== 'reconstruction'` to `createTrainer`.

`TrainingPanel.svelte`: add `showAccuracy = true` to the props destructuring and type; conditionally render the accuracy cell and set the grid columns:

```svelte
  <div class="stats" style="grid-template-columns: repeat({showAccuracy ? 3 : 2}, minmax(0, 1fr))">
```

and wrap the accuracy `<div>` in `{#if showAccuracy}…{/if}`.

- [ ] **Step 4: Run the tests**

Run: `npm test && npm run check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src
git commit -m "feat: add a reconstruction task mode"
```

---

### Task 2: Reconstruction tensors

**Files:**
- Modify: `src/lib/data/tensors.ts`
- Modify tests: `src/lib/data/tensors.test.ts`

**Interfaces:**
- Produces: `imagesToReconstruction(dataset, indices?)`.

- [ ] **Step 1: Write the failing test**

In `src/lib/data/tensors.test.ts`:

```ts
describe('imagesToReconstruction', () => {
  const dataset = {
    count: 2,
    rows: 2,
    cols: 2,
    numClasses: 10,
    pixels: Uint8Array.of(0, 255, 128, 64, 10, 20, 30, 40),
    labels: Uint8Array.of(3, 7)
  };

  it('builds matching input and target tensors', () => {
    const { xs, ys } = imagesToReconstruction(dataset);
    expect(xs.shape).toEqual([2, 2, 2, 1]);
    expect(ys.shape).toEqual([2, 2, 2, 1]);
    expect(Array.from(xs.dataSync())).toEqual(Array.from(ys.dataSync()));
    xs.dispose();
    ys.dispose();
  });

  it('normalises pixels to the zero-to-one range', () => {
    const { xs, ys } = imagesToReconstruction(dataset);
    expect(Array.from(xs.dataSync())[1]).toBeCloseTo(1, 5);
    xs.dispose();
    ys.dispose();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/data/tensors.test.ts`
Expected: FAIL with "imagesToReconstruction is not a function".

- [ ] **Step 3: Implement**

Add to `src/lib/data/tensors.ts`:

```ts
export function imagesToReconstruction(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor4D } {
  const selected = indices ?? Array.from({ length: dataset.count }, (_, index) => index);
  const size = dataset.rows * dataset.cols;
  const pixels = new Float32Array(selected.length * size);

  selected.forEach((imageIndex, row) => {
    const source = imageAt(dataset, imageIndex);
    for (let i = 0; i < size; i++) pixels[row * size + i] = source[i] / 255;
  });

  const shape: [number, number, number, number] = [selected.length, dataset.rows, dataset.cols, 1];
  return { xs: tf.tensor4d(pixels, shape), ys: tf.tensor4d(pixels.slice(), shape) };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/data/tensors.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/data
git commit -m "feat: build reconstruction tensors"
```

---

### Task 3: Reconstruction and code render modules

**Files:**
- Create: `src/lib/render/reconstruction.ts`, `src/lib/render/reconstruction.test.ts`
- Create: `src/lib/render/codes.ts`, `src/lib/render/codes.test.ts`

**Interfaces:**
- Consumes: `forwardActivations` from `./activations`; `LatentBounds` from `./latent`.
- Produces: `reconstruct(model, pixels, rows, cols, count)`, `codeScatter(model, pixels, rows, cols, count, layerIndex, dimA)`, `CodeSample`.

- [ ] **Step 1: Write the failing tests**

Create `src/lib/render/reconstruction.test.ts`:

The model must output an image, so use a small image-to-image network:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { buildModel } from '../tf/buildModel';
import { reconstruct } from './reconstruction';

let models: tf.Sequential[] = [];

const AUTOENCODER: Network = {
  version: 2,
  blocks: [
    { id: 'in', kind: 'input', shape: [4, 4, 1] },
    { id: 'conv', kind: 'conv2d', filters: 1, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'out', kind: 'output', shape: [4, 4, 1] }
  ],
  training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
  positions: {}
};

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('reconstruct', () => {
  it('returns one grayscale image per input', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const images = reconstruct(model, new Uint8Array(4 * 4).fill(200), 4, 4, 1);
    expect(images).toHaveLength(4 * 4);
    expect(Math.max(...images)).toBeLessThanOrEqual(255);
  });

  it('does not leak tensors', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const before = tf.memory().numTensors;
    reconstruct(model, new Uint8Array(4 * 4).fill(10), 4, 4, 1);
    expect(tf.memory().numTensors).toBe(before);
  });
});
```

Create `src/lib/render/codes.test.ts`:

The selected layer must be rank 1, so use a network with a dense code layer (model layers are conv, flatten, linear; `layerIndex` 2 is the linear):

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { buildModel } from '../tf/buildModel';
import { codeScatter } from './codes';

let models: tf.Sequential[] = [];

const AUTOENCODER: Network = {
  version: 2,
  blocks: [
    { id: 'in', kind: 'input', shape: [4, 4, 1] },
    { id: 'conv', kind: 'conv2d', filters: 1, kernelSize: 3, stride: 1, padding: 'same' },
    { id: 'flat', kind: 'flatten' },
    { id: 'code', kind: 'linear', units: 3 },
    { id: 'out', kind: 'output', shape: [3] }
  ],
  training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
  positions: {}
};

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('codeScatter', () => {
  it('projects each sample onto the chosen dimension pair', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const sample = codeScatter(model, new Uint8Array(3 * 4 * 4).fill(120), 4, 4, 3, 2, 0);
    expect(sample.points).toHaveLength(6);
    expect(Number.isFinite(sample.bounds.minX)).toBe(true);
  });

  it('does not leak tensors', () => {
    const model = buildModel(AUTOENCODER);
    models.push(model);
    const before = tf.memory().numTensors;
    codeScatter(model, new Uint8Array(4 * 4).fill(50), 4, 4, 1, 2, 0);
    expect(tf.memory().numTensors).toBe(before);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/render/reconstruction.test.ts src/lib/render/codes.test.ts`
Expected: FAIL with missing modules.

- [ ] **Step 3: Implement**

Create `src/lib/render/reconstruction.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { forwardActivations } from './activations';

export function reconstruct(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  count: number
): Uint8ClampedArray {
  return tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const input = tf.tensor4d(normalised, [count, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    const output = activations[activations.length - 1].dataSync();
    const result = new Uint8ClampedArray(count * rows * cols);
    for (let index = 0; index < result.length; index++) {
      result[index] = Math.round(Math.max(0, Math.min(1, output[index])) * 255);
    }
    return result;
  });
}
```

Create `src/lib/render/codes.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import type { LatentBounds } from './latent';
import { forwardActivations } from './activations';

export interface CodeSample {
  points: Float32Array;
  bounds: LatentBounds;
}

function boundsOf(points: Float32Array): LatentBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < points.length; index += 2) {
    const x = points[index];
    const y = points[index + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

export function codeScatter(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  count: number,
  layerIndex: number,
  dimA: number
): CodeSample {
  return tf.tidy(() => {
    const normalised = new Float32Array(pixels.length);
    for (let index = 0; index < pixels.length; index++) normalised[index] = pixels[index] / 255;
    const input = tf.tensor4d(normalised, [count, rows, cols, 1]);
    const activations = forwardActivations(model, input);
    const activation = activations[layerIndex];
    const a = activation.slice([0, dimA], [count, 1]);
    const b = activation.slice([0, dimA + 1], [count, 1]);
    const points = Float32Array.from(tf.concat([a, b], 1).dataSync());
    return { points, bounds: boundsOf(points) };
  });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/render/reconstruction.test.ts src/lib/render/codes.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/render
git commit -m "feat: render reconstructions and layer codes"
```

---

### Task 4: Reconstruction and code-scatter components

**Files:**
- Create: `src/lib/components/ReconstructionGrid.svelte`, `src/lib/components/ReconstructionGrid.test.ts`
- Create: `src/lib/components/CodeScatter.svelte`, `src/lib/components/CodeScatter.test.ts`

**Interfaces:**
- Consumes: `reconstruct`/`codeScatter` (dynamic), `probeTargetFor`, `imageAt`/`ImageDataset`.
- Produces: `ReconstructionGrid` with props `{ model, dataset, indices, redrawKey }`; `CodeScatter` with props `{ model, store, dataset, indices, redrawKey }`.

- [ ] **Step 1: Write the failing tests**

Model both test files on `src/lib/components/FeatureMaps.test.ts` and `src/lib/components/LatentSpace.test.ts` (fake 2D context, `vi.mock` of the render module, `ImageDataset`/`NetworkStore` fixtures).

`ReconstructionGrid.test.ts` must assert:
- the render module is called and no `reconstruction-error` appears;
- with `model` null the `reconstruction-message` explains the network must be fixed.

`CodeScatter.test.ts` must assert:
- the label reads `dimensions 1 & 2 of N` for a selected layer;
- the "Next dimensions" cycler advances and wraps (assert an intermediate label);
- no selection shows the select-a-block message;
- `model` null shows the fix message.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/ReconstructionGrid.test.ts src/lib/components/CodeScatter.test.ts`
Expected: FAIL with missing components.

- [ ] **Step 3: Implement**

`ReconstructionGrid.svelte`: a canvas with two rows of `indices.length` cells (originals above, reconstructions below), sized like `SampleGrid`. On mount dynamically import `render/reconstruction`. In a `redrawKey`-dependent effect, read the sample pixels with `imageAt`, call `reconstruct`, and draw each original from its pixels and each reconstruction from the returned grayscale. Include `data-testid="reconstruction-grid"`, `reconstruction-message`, and `reconstruction-error` elements.

`CodeScatter.svelte`: follow `LatentSpace.svelte`'s structure. Compute `target = probeTargetFor(store.network, store.selectedBlockId)`; render only when `target.dims.length === 1` and `dims[0] >= 2`. Own a `pair` (dimension index) that resets on selection change and wraps with a "Next dimensions" button. On each redraw, read the sample pixels, call `codeScatter(model, pixels, rows, cols, indices.length, target.source as number, pair)`, auto-fit the bounds, and draw the points coloured by `dataset.labels[indices[i]]`. Include `data-testid="code-scatter"`, `code-next`, `code-label`, `code-message`, `code-error`. Dynamically import `render/codes`.

- [ ] **Step 4: Run the tests, check, lint and build**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/components
git commit -m "feat: add reconstruction and code-scatter panels"
```

---

### Task 5: The autoencoder example, route and docs

**Files:**
- Create: `src/lib/examples/autoencoder/example.ts`, `src/lib/examples/autoencoder/example.test.ts`
- Create: `src/routes/examples/autoencoder/+page.svelte`
- Modify: `src/routes/+page.svelte`, `AGENTS.md`, `README.md`

**Interfaces:**
- Consumes: `createBlock`, `Network`, `findProblems`, `createExperiment`, `imagesToReconstruction`, `ReconstructionGrid`, `CodeScatter`, `FeatureMaps`, `TrainingPanel`, `LossChart`, `NetworkEditor`, `ExampleLayout`.
- Produces: `AUTOENCODER_PALETTE`, `AUTOENCODER_STORAGE_KEYS`, `AUTOENCODER_WEIGHTS_ID`, `PRESETS`, `TRAIN_COUNT`, `SCATTER_COUNT`, `RECONSTRUCTION_COUNT`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/examples/autoencoder/example.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { findProblems } from '../../network/problems';
import { AUTOENCODER_PALETTE, PRESETS } from './example';

describe('autoencoder presets', () => {
  it('offers a dense and a convolutional preset', () => {
    expect(PRESETS.map((preset) => preset.id)).toEqual(['dense', 'conv']);
  });

  it.each(PRESETS.map((preset) => preset.id))('%s builds with no errors', (id) => {
    const preset = PRESETS.find((candidate) => candidate.id === id)!;
    expect(findProblems(preset.create(), { task: 'reconstruction' })).toEqual([]);
  });

  it('offers reshape and upsampling', () => {
    expect(AUTOENCODER_PALETTE).toContain('reshape');
    expect(AUTOENCODER_PALETTE).toContain('upsampling2d');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/examples/autoencoder/example.test.ts`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement the example module**

Create `src/lib/examples/autoencoder/example.ts` with:

- `AUTOENCODER_PALETTE: BlockKind[] = ['conv2d','maxpool2d','upsampling2d','flatten','reshape','linear','relu','sigmoid','tanh','softmax']`
- `AUTOENCODER_STORAGE_KEYS = { network: 'visnet:autoencoder:network:v1', dataset: 'visnet:autoencoder:dataset:v1' }`
- `AUTOENCODER_WEIGHTS_ID = 'autoencoder'`
- `TRAIN_COUNT = 5000`, `SCATTER_COUNT = 500`, `RECONSTRUCTION_COUNT = 40`
- `PRESETS` with the two networks from spec §5. Build each with `createBlock`, overriding fields, and set `training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 }`.

- [ ] **Step 4: Implement the route and wire the docs**

`routes/examples/autoencoder/+page.svelte`: follow `routes/examples/cnn/+page.svelte`. Load MNIST, own a `NetworkStore` with `task = 'reconstruction'`, create the experiment with `{ task: 'reconstruction' }`, load the first `TRAIN_COUNT` train images via `imagesToReconstruction`, and the `SCATTER_COUNT`/`RECONSTRUCTION_COUNT` test samples. Render a preset `<select>` (switching calls `store.load(preset.create())` and resets the model), the `ReconstructionGrid`, the selected-layer view (rank 1 → `CodeScatter`, rank 3 → `FeatureMaps`), `TrainingPanel` with `showAccuracy={false}`, and `LossChart`.

Add a link to the new example on `routes/+page.svelte`.

Add `src/lib/render/reconstruction.ts` and `src/lib/render/codes.ts` to the `AGENTS.md` TF.js allow-list, and a bullet for the autoencoder example to `README.md`.

- [ ] **Step 5: Run the tests, check, lint and build**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds (including `/examples/autoencoder`).

- [ ] **Step 6: Commit**

```bash
git add -A src AGENTS.md README.md
git commit -m "feat: add the autoencoder example"
```

---

## Self-review

- **Spec coverage:** Task 1 is spec §4; Task 2 is §6; Task 3 is §7; Task 4 is §8; Task 5 is §5, §9, §10, §12. Spec §11's test list maps across the tasks.
- **Placeholder scan:** No "TBD"/"handle edge cases". The component tasks specify exact test IDs, props, and behaviours; they reference existing components to model on rather than inventing structure.
- **Type consistency:** `imagesToReconstruction`, `reconstruct`, `codeScatter`/`CodeSample`, and the preset/palette names are defined once and used by later tasks. `task` is added in Task 1 and consumed in Task 5.
- **Known risk:** Task 5 is an integration task (a new page); it mirrors the CNN page closely and the build gate covers prerendering.
