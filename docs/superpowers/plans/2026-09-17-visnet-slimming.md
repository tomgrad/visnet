# VisNet Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the bespoke "engine bureaucracy" around TensorFlow.js while keeping the visual teaching loop (palette, canvas, live shape badges, friendly errors, train, live visualization).

**Architecture:** Keep `src/lib/network/` as the pure domain, but delete the clamping/announcement machinery and the versioned trust boundary, collapse the two example controllers into one shared rune controller, and merge the smaller UI panels. TensorFlow.js stays confined to its current five modules; the visual editor stays a projection of the network.

**Tech Stack:** SvelteKit 2, Svelte 5 runes, TypeScript strict, `@tensorflow/tfjs`, `@xyflow/svelte`, Vitest (two projects: `engine` in Node, `ui` in jsdom).

## Global Constraints

- `src/lib/network/**` stays pure: no Svelte, no TF.js, no DOM, runs in plain Node.
- `@tensorflow/tfjs` runtime imports stay confined to `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, `src/lib/persist/weights.ts` and their colocated tests.
- `@tensorflow/tfjs` and `@xyflow/svelte` are only reached from the browser (dynamic import or `onMount`/`$effect`); every route must prerender.
- `src/lib/editor/flow.ts` and `src/lib/editor/history.ts` stay free of Svelte, TF.js, DOM, and `@xyflow/svelte`.
- No block stores its input dimension; shapes derive from the previous block's output.
- Node positions are cosmetic; the `blocks` array defines chain order.
- Every validation problem carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.
- Verify with `npm test`, `npm run check`, `npm run lint`, `npm run build` after every task.

---

## Size reality check (read before starting)

Production source is currently ~4,920 LOC (excluding tests, stubs, and CSS-only content). This moderate plan removes roughly **850–950 LOC**, landing at **~4.0k LOC** — about an 18% cut. It does **not** reach 2.5k.

To get below 3k you must drop a headline feature. The levers, with measured sizes:

| Lever | LOC saved | What is lost |
| --- | --- | --- |
| Drop the CNN example (`components/SampleGrid.svelte`, `routes/examples/cnn/+page.svelte`, `examples/cnn/*`, `data/images.ts`, `data/mnist.ts`, CNN slices of storage/runtime) | ~780 | MNIST teaching example |
| Drop undo/redo (`editor/history.ts`, toolbar wiring, shortcut, tests) | ~180 | "Mistakes are cheap" |
| Drop the shape table and per-edge labels (keep node badges + inspector shapes) | ~150 | Whole-pipeline shape view |

This plan performs the five cuts below and leaves those three levers to a follow-up decision.

---

## Target file disposition

| File | Action | Reason |
| --- | --- | --- |
| `src/lib/network/constraints.ts` + test | **Delete** | Clamping + announcement machinery; replace with one small clamp helper |
| `src/lib/network/serialize.ts` + test | **Replace** with `src/lib/persist/networkCodec.ts` | Versioned trust boundary is speculative; keep a shallow structural check |
| `src/lib/network/validate.ts` + test | **Rename** to `problems.ts`, keep rules, factor shared helpers | Name matches the UI; no behavior cut |
| `src/routes/examples/mlp/+page.svelte` | **Slim** via shared controller | ~250 duplicated lines |
| `src/routes/examples/cnn/+page.svelte` | **Slim** via shared controller | ~250 duplicated lines |
| `src/lib/examples/experiment.svelte.ts` | **Create** | Owns model/trainer/data/persistence lifecycle |
| `src/lib/components/ShapeTable.svelte` + test | **Keep** | Must stay visible when nothing is selected |
| `src/lib/components/StatsReadout.svelte` + test | **Merge** into `TrainingPanel.svelte` | Same panel concern |
| `src/lib/components/DecisionBoundaryHarness.svelte` | **Delete** | Test-only; move to `__stubs__` |
| `src/lib/examples/notices.ts` + test | **Delete** | Single constant; inline it |
| `src/lib/editor/networkStore.svelte.ts` | **Modify** | Drop announcements/clamping; `issues` → `problems` |
| `src/lib/tf/buildModel.ts` | **Modify** | Wrap TF errors as plain-language `Problem[]` |
| `src/lib/persist/storage.ts` | **Modify** | Use `networkCodec`; drop `clear()` |
| `src/lib/components/NetworkEditor.svelte` | **Modify** | Remove announcements region; merge panels |
| `src/lib/components/InspectorPanel.svelte` | **Modify** | Inline clamp; render shape table |
| `src/lib/components/TrainingPanel.svelte` | **Modify** | Render stats readout |
| `README.md`, `AGENTS.md` | **Modify** | Reflect removed modules |

---

### Task 1: Replace the versioned serializer with a plain codec

**Files:**
- Create: `src/lib/persist/networkCodec.ts`
- Create: `src/lib/persist/networkCodec.test.ts`
- Delete: `src/lib/network/serialize.ts`, `src/lib/network/serialize.test.ts`
- Modify: `src/lib/persist/storage.ts`

**Interfaces:**
- Consumes: `Network` from `src/lib/network/types.ts`.
- Produces: `encodeNetwork(net: Network): string`, `decodeNetwork(raw: string): Network | null`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/persist/networkCodec.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { decodeNetwork, encodeNetwork } from './networkCodec';

describe('networkCodec', () => {
  it('round-trips a network', () => {
    const net = createEmptyNetwork();
    expect(decodeNetwork(encodeNetwork(net))).toEqual(net);
  });

  it('returns null for invalid JSON', () => {
    expect(decodeNetwork('not json')).toBeNull();
  });

  it('returns null when the blocks array is missing', () => {
    expect(decodeNetwork(JSON.stringify({ training: {} }))).toBeNull();
  });

  it('returns null when a block lacks an id or kind', () => {
    expect(decodeNetwork(JSON.stringify({ blocks: [{ id: 'a' }] }))).toBeNull();
  });

  it('returns null when training is missing', () => {
    const net = createEmptyNetwork();
    expect(decodeNetwork(JSON.stringify({ blocks: net.blocks }))).toBeNull();
  });

  it('accepts a payload with no positions and defaults them to empty', () => {
    const net = createEmptyNetwork();
    const restored = decodeNetwork(
      JSON.stringify({ blocks: net.blocks, training: net.training })
    );
    expect(restored?.positions).toEqual({});
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/persist/networkCodec.test.ts`
Expected: FAIL with "Cannot find module './networkCodec'".

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/persist/networkCodec.ts`:

```ts
import type { Block, Network, NodePosition, TrainingConfig } from '../network/types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isBlock(value: unknown): value is Block {
  return isRecord(value) && typeof value.id === 'string' && typeof value.kind === 'string';
}

function isTraining(value: unknown): value is TrainingConfig {
  return isRecord(value) && typeof value.loss === 'string' && typeof value.optimizer === 'string';
}

function readPositions(value: unknown): Record<string, NodePosition> {
  if (!isRecord(value)) return {};
  const positions: Record<string, NodePosition> = {};
  for (const [id, position] of Object.entries(value)) {
    if (isRecord(position) && typeof position.x === 'number' && typeof position.y === 'number') {
      positions[id] = { x: position.x, y: position.y };
    }
  }
  return positions;
}

export function encodeNetwork(net: Network): string {
  return JSON.stringify(net);
}

export function decodeNetwork(raw: string): Network | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return null;
    if (!Array.isArray(parsed.blocks) || !parsed.blocks.every(isBlock)) return null;
    if (!isTraining(parsed.training)) return null;
    return {
      version: 2,
      blocks: parsed.blocks,
      training: parsed.training,
      positions: readPositions(parsed.positions)
    };
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/persist/networkCodec.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Point storage at the codec and delete the serializer**

In `src/lib/persist/storage.ts`, replace the import and the two network methods:

```ts
import { decodeNetwork, encodeNetwork } from './networkCodec';
```

```ts
    saveNetwork(net) {
      backing.setItem(keys.network, encodeNetwork(net));
    },
    loadNetwork() {
      const raw = backing.getItem(keys.network);
      return raw === null ? null : decodeNetwork(raw);
    },
```

Also delete the `clear()` method from the `NetworkStorage` interface and the returned object (no production caller).

Delete `src/lib/network/serialize.ts` and `src/lib/network/serialize.test.ts`.

- [ ] **Step 6: Run the affected tests**

Run: `npx vitest run src/lib/persist`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A src/lib/persist src/lib/network
git commit -m "refactor: replace versioned serializer with a plain network codec"
```

---

### Task 2: Delete the clamping and announcement machinery

**Files:**
- Delete: `src/lib/network/constraints.ts`, `src/lib/network/constraints.test.ts`
- Modify: `src/lib/editor/networkStore.svelte.ts`
- Modify: `src/lib/components/NetworkEditor.svelte`
- Modify: `src/lib/components/InspectorPanel.svelte`
- Modify: `src/lib/components/NetworkEditor.test.ts`, `src/lib/editor/networkStore.svelte.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `NetworkStore` without `announcements`/`announce`/`dismissAnnouncements`; `updateBlock` stores the patch as-is.

- [ ] **Step 1: Write the failing test**

In `src/lib/editor/networkStore.svelte.test.ts`, replace any test that asserts an announcement with:

```ts
  it('stores a unit count without an announcement', () => {
    const store = new NetworkStore();
    const id = store.network.blocks[1].id;
    store.updateBlock(id, { units: 4 });
    expect(store.network.blocks[1]).toMatchObject({ units: 4 });
  });
```

In `src/lib/components/NetworkEditor.test.ts`, delete the `announces corrections in a live region` test and the `surfaces an automatic correction as a dismissible announcement` test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/editor/networkStore.svelte.test.ts src/lib/components/NetworkEditor.test.ts`
Expected: FAIL because `NetworkStore` still requires the clamping path, or because deleted tests no longer compile against removed helpers. Confirm the new test fails only on the missing behavior.

- [ ] **Step 3: Remove the machinery**

In `src/lib/editor/networkStore.svelte.ts`:
- Delete `announcements`, `canUndo`/`canRedo` stay, delete `announce()` and `dismissAnnouncements()`.
- Delete the `clampBlockPatch`/`clampNetwork` imports.
- Replace `updateBlock`:

```ts
  updateBlock(id: string, patch: Partial<Block>): void {
    this.#commit(replaceBlock(this.network, id, patch));
  }
```

- Replace `#commitClamped` call sites (`addBlock`, `moveBlock`) with `#commit`, and delete `#commitClamped`.
- In `load`, replace `clampNetwork(net)` with `net` and drop the announcement assignment.

Delete `src/lib/network/constraints.ts` and `src/lib/network/constraints.test.ts`.

In `src/lib/components/NetworkEditor.svelte`, delete the `{#if store.announcements.length > 0}` block (the live region added recently).

In `src/lib/components/InspectorPanel.svelte`, remove the `parameterBounds` import and derive size/stride choices from the incoming shape directly:

```ts
  const limit = $derived(inShape && inShape.length === 3 ? Math.min(inShape[0], inShape[1]) : null);
  const sizeChoices = $derived(
    spatialSize === null || limit === null
      ? spatialSize === null
        ? []
        : [spatialSize]
      : Array.from({ length: Math.floor(limit) }, (_, index) => index + 1)
  );
  const strideChoices = $derived(
    spatialStride === null || limit === null
      ? spatialStride === null
        ? []
        : [spatialStride]
      : Array.from({ length: Math.floor(limit) }, (_, index) => index + 1)
  );
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/lib/editor src/lib/components`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/network src/lib/editor src/lib/components
git commit -m "refactor: drop clamping and announcement machinery"
```

---

### Task 3: Rename `validate` to `problems` and wrap build errors

**Files:**
- Rename: `src/lib/network/validate.ts` → `src/lib/network/problems.ts`; `validate.test.ts` → `problems.test.ts`
- Modify: `src/lib/tf/buildModel.ts`
- Modify: `src/lib/editor/networkStore.svelte.ts`
- Modify: `src/lib/examples/cnn/example.test.ts`

**Interfaces:**
- Consumes: `inferShapes` from `src/lib/network/inferShapes.ts`.
- Produces: `Problem`, `ProblemOptions`, `findProblems(net, options?): Problem[]`; `NetworkInvalidError` carrying `Problem[]`; `describeBuildError(error: unknown): Problem`.

- [ ] **Step 1: Write the failing test**

In `src/lib/tf/buildModel.test.ts`, add:

```ts
  it('turns an unexpected TF.js build failure into a plain-language problem', () => {
    const problem = describeBuildError(new Error('Shape mismatch: expected 2, got 3'));
    expect(problem.severity).toBe('error');
    expect(problem.title.length).toBeGreaterThan(0);
    expect(problem.message).toContain('Shape mismatch');
    expect(problem.fix.length).toBeGreaterThan(0);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/tf/buildModel.test.ts`
Expected: FAIL with "describeBuildError is not a function".

- [ ] **Step 3: Implement**

Rename the module and its exports:

```bash
git mv src/lib/network/validate.ts src/lib/network/problems.ts
git mv src/lib/network/validate.test.ts src/lib/network/problems.test.ts
```

In `problems.ts`, rename `Issue` → `Problem`, `ValidateOptions` → `ProblemOptions`, `validate` → `findProblems`, and add shared constructors to cut boilerplate:

```ts
function error(problem: Omit<Problem, 'severity'>): Problem {
  return { severity: 'error', ...problem };
}

function warning(problem: Omit<Problem, 'severity'>): Problem {
  return { severity: 'warning', ...problem };
}
```

Convert each `issues.push({ severity: 'error', ... })` to `issues.push(error({ ... }))` and each warning likewise. Update the import in `src/lib/editor/networkStore.svelte.ts` and rename `issues` → `problems` throughout, including `errors`/`warnings` derived values.

In `src/lib/tf/buildModel.ts`, add:

```ts
export function describeBuildError(cause: unknown): Problem {
  const detail = cause instanceof Error ? cause.message : String(cause);
  return {
    severity: 'error',
    title: 'The network could not be built',
    message: `TensorFlow.js rejected this network: ${detail}`,
    fix: 'Check the layer sizes and the input shape, then try again.'
  };
}
```

and change `NetworkInvalidError` to hold `Problem[]`.

Update `src/lib/examples/cnn/example.test.ts` to import `findProblems` and expect `[]`.

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/lib/network src/lib/tf src/lib/editor`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/network src/lib/tf src/lib/editor src/lib/examples
git commit -m "refactor: rename validate to problems and describe build errors"
```

---

### Task 4: Extract a shared experiment controller

**Files:**
- Create: `src/lib/examples/experiment.svelte.ts`
- Modify: `src/routes/examples/mlp/+page.svelte`
- Modify: `src/routes/examples/cnn/+page.svelte`

**Interfaces:**
- Consumes: `loadRuntime`, `Runtime`, `Model`, `ModelData`, `TrainerHandle` from `src/lib/examples/runtime.ts`; `NetworkStore`; `NetworkStorage`.
- Produces:

```ts
export interface Experiment {
  runtime: Runtime | null;
  model: Model | null;
  data: ModelData | null;
  playing: boolean;
  stats: TrainStats | null;
  lossPoints: number[];
  banner: string | null;
  saving: boolean;
  redrawKey: number;
  setData(next: ModelData | null): void;
  announce(message: string): void;
  play(): Promise<void>;
  pause(): void;
  step(): void;
  resetModel(): void;
  save(): Promise<void>;
  load(): Promise<void>;
  dispose(): void;
}

export function createExperiment(options: {
  store: NetworkStore;
  weightsId: string;
  storage: NetworkStorage | null;
}): Experiment;
```

This is a behavior-preserving refactor: the existing test suite is the safety net, and the controller is a new seam that has no unit test of its own. Do not invent a test that only exercises unrelated helpers.

- [ ] **Step 1: Record the green baseline**

Run: `npm test`
Expected: PASS. Record the test and file counts so Task 4 can prove nothing regressed.

- [ ] **Step 2: Implement the controller**

Create `src/lib/examples/experiment.svelte.ts` by lifting the shared lifecycle out of `src/routes/examples/mlp/+page.svelte:32-223`:

```ts
import { onDestroy, onMount } from 'svelte';
import type { NetworkStore } from '../editor/networkStore.svelte';
import type { NetworkStorage } from '../persist/storage';
import type { TrainStats } from '../training/Trainer';
import { weightsDiscardedNotice } from './notices';
import {
  loadRuntime,
  type Model,
  type ModelData,
  type Runtime,
  type TrainerHandle
} from './runtime';

export interface Experiment {
  runtime: Runtime | null;
  model: Model | null;
  data: ModelData | null;
  playing: boolean;
  stats: TrainStats | null;
  lossPoints: number[];
  banner: string | null;
  saving: boolean;
  redrawKey: number;
  setData(next: ModelData | null): void;
  announce(message: string): void;
  play(): Promise<void>;
  pause(): void;
  step(): void;
  resetModel(): void;
  save(): Promise<void>;
  load(): Promise<void>;
  dispose(): void;
}

export function createExperiment(options: {
  store: NetworkStore;
  weightsId: string;
  storage: NetworkStorage | null;
}): Experiment {
  const { store, storage } = options;

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
  let data = $state.raw<ModelData | null>(null);
  let playing = $state(false);
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let redrawKey = $state(0);
  let builtSignature = $state('');
  let trained = $state(false);

  let trainer: TrainerHandle | null = null;
  let compiledTraining = '';

  const architecture = $derived(JSON.stringify(store.network.blocks));
  const trainingSignature = $derived(JSON.stringify(store.network.training));

  function handleStats(next: TrainStats): void {
    redrawKey += 1;
    stats = next;
    if (next.epochMeanLoss !== null) {
      trained = true;
      lossPoints = [...lossPoints, next.epochMeanLoss].slice(-200);
    }
  }

  function handleError(error: unknown): void {
    playing = false;
    console.error(error);
    banner = 'Training stopped. Reset the model and try again.';
  }

  function releaseTrainer(): void {
    trainer?.dispose();
    trainer = null;
    playing = false;
  }

  function setData(next: ModelData | null): void {
    releaseTrainer();
    runtime?.disposeData(data);
    data = next;
  }

  $effect(() => {
    const api = runtime;
    const currentModel = model;
    const currentData = data;
    const batchSize = store.network.training.batchSize;
    releaseTrainer();
    if (!api || !currentModel || !currentData || !store.isValid || currentData.xs.shape[0] === 0) {
      return;
    }
    trainer = api.createTrainer(currentModel, currentData, batchSize, handleStats, handleError);
  });

  $effect(() => {
    const api = runtime;
    if (!api || architecture === builtSignature) return;
    builtSignature = architecture;
    compiledTraining = trainingSignature;
    releaseTrainer();
    stats = null;
    lossPoints = [];
    const hadTrained = trained;
    trained = false;
    api.disposeModel(model);
    model = null;
    if (hadTrained) banner = weightsDiscardedNotice(true);
    if (!store.isValid) return;
    try {
      model = api.buildModel(store.network);
    } catch (error) {
      console.error(error);
      model = null;
      banner = 'This network could not be built. Check the problems listed in the editor.';
    }
  });

  $effect(() => {
    const api = runtime;
    const training = trainingSignature;
    if (!api || !model || training === compiledTraining) return;
    compiledTraining = training;
    api.compileModel(model, store.network.training);
  });

  $effect(() => {
    const currentStorage = storage;
    if (!currentStorage) return;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(store.network);
      } catch {
        banner = 'Your work could not be saved. The browser storage may be full.';
      }
    }, 500);
    return () => clearTimeout(timer);
  });

  onMount(async () => {
    runtime = await loadRuntime(options.weightsId);
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeModel(model);
  });

  return {
    get runtime() {
      return runtime;
    },
    get model() {
      return model;
    },
    get data() {
      return data;
    },
    get playing() {
      return playing;
    },
    get stats() {
      return stats;
    },
    get lossPoints() {
      return lossPoints;
    },
    get banner() {
      return banner;
    },
    set banner(value: string | null) {
      banner = value;
    },
    get saving() {
      return saving;
    },
    get redrawKey() {
      return redrawKey;
    },
    setData,
    announce(message: string) {
      banner = message;
    },
    async play() {
      if (!trainer) return;
      playing = true;
      await trainer.play();
      playing = false;
    },
    pause() {
      trainer?.pause();
      playing = false;
    },
    step() {
      void trainer?.step();
    },
    resetModel() {
      releaseTrainer();
      runtime?.disposeModel(model);
      model = null;
      stats = null;
      lossPoints = [];
      trained = false;
      builtSignature = '';
    },
    async save() {
      const api = runtime;
      if (!api || !model) return;
      saving = true;
      try {
        await api.saveWeights(model);
        banner = 'Model saved in this browser.';
      } catch {
        banner = 'The model could not be saved in this browser.';
      } finally {
        saving = false;
      }
    },
    async load() {
      const api = runtime;
      if (!api || !model) return;
      saving = true;
      try {
        const loaded = await api.loadWeightsInto(model);
        banner = loaded
          ? 'Saved weights loaded.'
          : 'No saved weights match this network. Train and save again.';
      } finally {
        saving = false;
      }
    },
    dispose() {
      releaseTrainer();
      runtime?.disposeData(data);
      runtime?.disposeModel(model);
    }
  };
}
```

- [ ] **Step 3: Rewire the MLP page**

Replace `src/routes/examples/mlp/+page.svelte:32-223` with the controller plus the MLP-only dataset logic:

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import DecisionBoundary from '$lib/components/DecisionBoundary.svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import { GENERATOR_DESCRIPTIONS, GENERATOR_NAMES } from '$lib/data/points';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import { DatasetStore } from '$lib/examples/mlp/datasetStore.svelte';
  import { createExperiment } from '$lib/examples/experiment.svelte';
  import { CLASS_LABELS, MLP_PALETTE, MLP_STORAGE_KEYS, MLP_WEIGHTS_ID } from '$lib/examples/mlp/example';
  import { createBrowserStorage } from '$lib/persist/storage';

  const store = new NetworkStore();
  const datasetStore = new DatasetStore();
  store.expectedClasses = 2;
  const storage = createBrowserStorage(MLP_STORAGE_KEYS);
  const experiment = createExperiment({ store, weightsId: MLP_WEIGHTS_ID, storage });

  $effect(() => {
    if (!experiment.runtime) return;
    experiment.setData(experiment.runtime.toTensors(datasetStore.dataset));
  });
</script>
```

The template keeps the existing dataset controls, `DecisionBoundary`, `TrainingPanel`, `LossChart`, and `StatsReadout` (still passing `stats={experiment.stats}`; Task 5 merges it into `TrainingPanel`). Replace `banner` with `experiment.banner` and call `experiment.play/pause/step/resetModel/save/load`.

- [ ] **Step 4: Rewire the CNN page**

In `src/routes/examples/cnn/+page.svelte`, replace the model/trainer/persistence block with the controller plus the MNIST-only dataset logic:

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import SampleGrid from '$lib/components/SampleGrid.svelte';
  import StatsReadout from '$lib/components/StatsReadout.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import type { ImageDataset } from '$lib/data/images';
  import { loadMnistData } from '$lib/data/mnist';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import {
    CNN_PALETTE,
    CNN_STORAGE_KEYS,
    CNN_WEIGHTS_ID,
    createCnnNetwork,
    defaultSampleIndices
  } from '$lib/examples/cnn/example';
  import { createExperiment } from '$lib/examples/experiment.svelte';
  import type { ModelData } from '$lib/examples/runtime';
  import { createBrowserStorage } from '$lib/persist/storage';

  const store = new NetworkStore(createCnnNetwork());
  store.expectedClasses = 10;
  store.expectedInputShape = [28, 28, 1];
  const storage = createBrowserStorage(CNN_STORAGE_KEYS);
  const experiment = createExperiment({ store, weightsId: CNN_WEIGHTS_ID, storage });

  let testData = $state.raw<ImageDataset | null>(null);
  let sampleData = $state.raw<ModelData | null>(null);
  let trainData = $state.raw<ImageDataset | null>(null);
  let loadState = $state<'loading' | 'ready' | 'unavailable'>('loading');
  const sampleIndices = $derived(testData ? defaultSampleIndices(testData.count) : []);

  onMount(async () => {
    try {
      const mnist = await loadMnistData();
      trainData = mnist.train;
      testData = mnist.test;
      loadState = 'ready';
    } catch (error) {
      console.error(error);
      loadState = 'unavailable';
    }
  });

  $effect(() => {
    const api = experiment.runtime;
    if (!api || !trainData) return;
    experiment.setData(api.imagesToTensors(trainData));
  });

  $effect(() => {
    const api = experiment.runtime;
    const test = testData;
    if (!api || !test) return;
    sampleData = api.imagesToTensors(test, sampleIndices);
  });
</script>
```

The template keeps the banner, load-state notes, `SampleGrid` (with `sampleXs={sampleData?.xs ?? null}`), `TrainingPanel`, `LossChart`, and `StatsReadout`; replace `banner` with `experiment.banner` and the control handlers with `experiment.*`. Note: the original page disposed the previous sample tensors; `createExperiment` only owns training `data`, so keep a small `onDestroy` in the page that disposes `sampleData` via `experiment.runtime`.

- [ ] **Step 5: Run the tests and build**

Run: `npm test && npm run check && npm run build`
Expected: PASS, same test count as the baseline, 0 errors, prerender succeeds.

- [ ] **Step 6: Commit**

```bash
git add -A src/lib/examples src/routes
git commit -m "refactor: extract a shared experiment controller from the example pages"
```

---

### Task 5: Merge the stats readout into the training panel

**Files:**
- Delete: `src/lib/components/StatsReadout.svelte`, `StatsReadout.test.ts`
- Modify: `src/lib/components/TrainingPanel.svelte`
- Modify: `src/routes/examples/mlp/+page.svelte`, `src/routes/examples/cnn/+page.svelte`

**Interfaces:**
- Consumes: `TrainStats` from `src/lib/training/Trainer.ts`.
- Produces: `TrainingPanel` accepts an optional `stats?: TrainStats | null` prop (default `null`), so existing callers and the test helper keep compiling.

Do **not** merge `ShapeTable` into `InspectorPanel`: the inspector only renders when a block is selected, so the shape table would disappear when nothing is selected. `ShapeTable` stays a standalone component.

- [ ] **Step 1: Write the failing test**

In `src/lib/components/TrainingPanel.test.ts`, add:

```ts
  it('shows the epoch, loss and accuracy of the latest stats', () => {
    panel({
      stats: { epoch: 3, batch: 0, batchLoss: 0.12, epochMeanLoss: 0.2, epochAccuracy: 0.9 }
    });
    const text = screen.getByTestId('training-panel').textContent ?? '';
    expect(text).toContain('3');
    expect(text).toContain('0.20');
    expect(text).toContain('90');
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/components/TrainingPanel.test.ts`
Expected: FAIL because no stats markup exists.

- [ ] **Step 3: Implement**

Move the body of `StatsReadout.svelte` into `TrainingPanel.svelte` under the controls, guarded by `{#if stats}`. Add `stats = null` to the props destructuring and type it `stats?: TrainStats | null`. Keep the fixed three-column grid and `tabular-nums` styling from `StatsReadout.svelte:65-92`.

Delete `StatsReadout.svelte` and `StatsReadout.test.ts`. In both example pages, remove the `StatsReadout` import and its `<StatsReadout ... />` render (the panel now shows the same numbers), and pass `stats={experiment.stats}` to `TrainingPanel`.

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/components src/routes
git commit -m "refactor: merge the stats readout into the training panel"
```

---

### Task 6: Remove dead code and test-only production files

**Files:**
- Delete: `src/lib/components/DecisionBoundaryHarness.svelte`; move its markup to `src/lib/components/__stubs__/DecisionBoundaryHarness.svelte`
- Delete: `src/lib/examples/notices.ts`, `notices.test.ts` (inline the constant in `experiment.svelte.ts`)
- Modify: `src/lib/components/DecisionBoundary.test.ts` (import path)
- Modify: `src/lib/editor/networkStore.svelte.ts` (delete unused `paramCount`)
- Modify: `src/lib/editor/networkStore.svelte.test.ts` (drop the `paramCount` assertion)
- Modify: `src/lib/components/EditorToolbar.svelte` (delete the never-passed `onfit` prop)

**Interfaces:**
- Consumes: Task 4's `experiment.svelte.ts`.
- Produces: no public interface changes.

- [ ] **Step 1: Write the failing test**

In `src/lib/components/DecisionBoundary.test.ts`, change the import to:

```ts
import DecisionBoundaryHarness from './__stubs__/DecisionBoundaryHarness.svelte';
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/components/DecisionBoundary.test.ts`
Expected: FAIL with "Cannot find module './__stubs__/DecisionBoundaryHarness.svelte'".

- [ ] **Step 3: Move the file and delete dead code**

```bash
git mv src/lib/components/DecisionBoundaryHarness.svelte src/lib/components/__stubs__/DecisionBoundaryHarness.svelte
```

Fix any relative imports inside the moved file (`../` becomes `../../`).

In `experiment.svelte.ts`, replace the `notices` import with the constant inline:

```ts
const WEIGHTS_DISCARDED_NOTICE = 'The network changed, so training restarted with fresh weights.';
```

Delete `src/lib/examples/notices.ts` and `notices.test.ts`. In `src/lib/editor/networkStore.svelte.ts`, delete the unused `paramCount` derived, and delete the `expect(instance.paramCount).toBe(42)` assertion in `src/lib/editor/networkStore.svelte.test.ts`. In `src/lib/components/EditorToolbar.svelte`, delete the `onfit` prop and its `{#if onfit}` button.

- [ ] **Step 4: Run the tests, check, lint and build**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A src
git commit -m "chore: remove dead code and test-only production files"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the architecture sections**

In `README.md`, update the Architecture paragraph to remove `serialize.ts` and name `persist/networkCodec.ts`. In `AGENTS.md`, update the shipped-modules list, the TF.js confinement list, and remove the sentence that calls `render/boundary.ts` "the only render module that runs a forward pass" (it is not).

- [ ] **Step 2: Verify nothing is stale**

Run: `npm test && npm run check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: reflect the slimmed module layout"
```

---

## Optional further cuts (decide after Task 7)

If the result still feels too large, pick from the table in "Size reality check". Each is independent and can be its own plan. Do not start them in the same branch as the tasks above.

## Self-review

- **Spec coverage:** The five cuts map to the disposition table. Live shape badges (`inferShapes`), friendly errors (`problems.ts`), the editor, both examples, training, and persistence all remain. Undo/redo remains. No teaching-loop feature is removed by this plan.
- **Placeholder scan:** No "TBD"/"handle edge cases" steps. Code is provided for every new module and every changed function signature.
- **Type consistency:** `Problem`/`ProblemOptions`/`findProblems`/`describeBuildError` are defined in Task 3 and used in Tasks 4–6. `Experiment`/`createExperiment` are defined in Task 4 and used in Tasks 5–6. `stats` is added to `TrainingPanel` in Task 5 and passed by Task 4's pages in Task 5's step 3.
- **Known risk:** Task 4's controller depends on `$effect`/`$state` in a `.svelte.ts` module called during component init. If Svelte rejects an effect outside a component context, call `createExperiment` at the top of the page `<script>` (it already is) and move the two `$effect` blocks back into each page as thin wrappers that call controller methods.
