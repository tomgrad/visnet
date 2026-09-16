# VisNet CNN Example Implementation Plan (C2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CNN example page — a convolutional network trained on handwritten digits, with a live grid of test images showing what the network predicts — on top of the data pipeline that phase C1 already delivered.

**Architecture:** A pure example-configuration module declares the convolutional palette, the CNN's storage keys and weight id, the sample-grid geometry, and the default network. A pure grid module turns logits into predictions and cell indices into pixel rectangles. The page reuses the MLP example's editor, training panel, loss chart, and stats readout, with the runtime facade generalised so both examples share it. The sample grid is one canvas, redrawn at most once per animation frame.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@tensorflow/tfjs`, Vitest (engine project in Node, ui project in jsdom).

## Global Constraints

- Package manager is **npm**. Do not use pnpm, yarn, or bun.
- TypeScript strict mode is on; `npm run check` must report 0 errors and 0 warnings.
- Svelte 5 runes only. Do not use `svelte/store`.
- `src/lib/network/**` stays pure: no Svelte, no `@tensorflow/tfjs`, no DOM.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, and `src/lib/persist/weights.ts`. This plan adds no new import site.
- No backend, no server code. All routes prerendered; the build output is static files.
- **No dataset is committed.** `static/mnist/` is gitignored and nothing downloads it at build or install time; the page degrades to an instruction when it is absent.
- No code comments unless a non-obvious constraint requires one.
- No `any`.
- Plain CSS with the tokens in `src/lib/styles/tokens.css`; no new tokens.
- Commit at the end of every task with the exact message shown in that task.
- Do not commit anything under `.superpowers/`.
- Design reference: `docs/superpowers/specs/2026-09-16-visnet-cnn-design.md`.

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/lib/examples/cnn/example.ts` | The CNN palette, storage keys, weight id, sample-grid geometry, and the default convolutional network |
| `src/lib/examples/cnn/grid.ts` | Pure: logits to predictions, cell index to pixel rectangle |
| `src/lib/network/validate.ts` | Modify: a new `expectedInputShape` option and its warning |
| `src/lib/editor/networkStore.svelte.ts` | Modify: an `expectedInputShape` field passed through to `validate` |
| `src/lib/training/Trainer.ts` | Modify: accept any `tf.Tensor` for the feature batch, so image batches work |
| `src/lib/examples/runtime.ts` | Moved from `examples/mlp/runtime.ts`, generalised to take a weight id, and given image conversion |
| `src/lib/examples/mlp/+page.svelte` | Modify: use the shared runtime with the MLP's weight id |
| `src/lib/components/SampleGrid.svelte` | One canvas: the sample digits, their predicted label, and a correct/incorrect border |
| `src/routes/examples/cnn/+page.svelte` | The CNN example page |
| `README.md`, `AGENTS.md` | Modify: both examples ship |

---

### Task 1: CNN example configuration

**Files:**
- Create: `src/lib/examples/cnn/example.ts`
- Test: `src/lib/examples/cnn/example.test.ts`

**Interfaces:**
- Consumes: `createBlock` from `../../network/factory`; `StorageKeys` from `../../persist/storage`; `BlockKind`, `InputBlock`, `LinearBlock`, `Network`, `OutputBlock` from `../../network/types`.
- Produces:

```ts
export const CNN_PALETTE: BlockKind[];
export const CNN_STORAGE_KEYS: StorageKeys;
export const CNN_WEIGHTS_ID: string;
export const SAMPLE_GRID_SIZE: number;
export const SAMPLE_GRID_COLUMNS: number;
export const SAMPLE_GRID_CELL_WIDTH: number;
export const SAMPLE_GRID_CELL_HEIGHT: number;
export const SAMPLE_GRID_GAP: number;
export function createCnnNetwork(): Network;
export function defaultSampleIndices(): number[];
```

- `CNN_PALETTE` is `['conv2d', 'flatten', 'linear', 'relu', 'sigmoid', 'softmax']` — the MLP's palette plus convolution and flatten, and without the input/output blocks that are pinned rather than chosen.
- `CNN_STORAGE_KEYS` is `{ network: 'visnet:cnn:network:v1', dataset: 'visnet:cnn:dataset:v1' }`. The dataset key is never used — the CNN does not persist its data — but `StorageKeys` requires both fields, deliberately, so the interface has no "sometimes absent" branch.
- `CNN_WEIGHTS_ID` is `'cnn'`.
- The grid geometry is 40 cells in 8 columns, each 36×48 pixels with a 4-pixel gap: 36 wide holds a 28-pixel image with a 4-pixel margin, and 48 tall holds the image plus a strip for the predicted digit. Eight columns plus gaps is 316 pixels, which fits the example column's 320-pixel minimum.
- `createCnnNetwork()` returns `input [28, 28, 1] → conv2d(filters 8, kernel 3, stride 1, same) → relu → flatten → linear(10) → softmax → output(10)`, with training `{ loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }` and empty positions.
- `defaultSampleIndices()` returns `[0, 1, …, SAMPLE_GRID_SIZE - 1]` — the first 40 test images.

- [ ] **Step 1: Write the failing test**

`src/lib/examples/cnn/example.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { validate } from '../../network/validate';
import {
  CNN_PALETTE,
  CNN_STORAGE_KEYS,
  CNN_WEIGHTS_ID,
  SAMPLE_GRID_COLUMNS,
  SAMPLE_GRID_SIZE,
  createCnnNetwork,
  defaultSampleIndices
} from './example';

describe('CNN_PALETTE', () => {
  it('offers convolution and flatten, and no pinned blocks', () => {
    expect(CNN_PALETTE).toEqual([
      'conv2d',
      'flatten',
      'linear',
      'relu',
      'sigmoid',
      'softmax'
    ]);
    expect(CNN_PALETTE).not.toContain('input');
    expect(CNN_PALETTE).not.toContain('output');
  });
});

describe('CNN_STORAGE_KEYS', () => {
  it('is namespaced away from the MLP', () => {
    expect(CNN_STORAGE_KEYS.network).toBe('visnet:cnn:network:v1');
    expect(CNN_STORAGE_KEYS.network).not.toBe('visnet:network:v1');
    expect(CNN_WEIGHTS_ID).toBe('cnn');
  });
});

describe('createCnnNetwork', () => {
  it('builds a convolutional network with no validation errors', () => {
    const network = createCnnNetwork();
    expect(network.version).toBe(2);
    expect(network.positions).toEqual({});
    expect(network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'conv2d',
      'relu',
      'flatten',
      'linear',
      'softmax',
      'output'
    ]);
    expect(validate(network, { expectedClasses: 10 })).toEqual([]);
  });

  it('reads a 28 by 28 image and predicts ten classes', () => {
    const network = createCnnNetwork();
    expect(network.blocks[0]).toMatchObject({ kind: 'input', shape: [28, 28, 1] });
    expect(network.blocks[4]).toMatchObject({ kind: 'linear', units: 10 });
    expect(network.blocks[6]).toMatchObject({ kind: 'output', units: 10 });
  });
});

describe('defaultSampleIndices', () => {
  it('is the first forty images', () => {
    const indices = defaultSampleIndices();
    expect(indices).toHaveLength(SAMPLE_GRID_SIZE);
    expect(indices[0]).toBe(0);
    expect(indices[SAMPLE_GRID_SIZE - 1]).toBe(SAMPLE_GRID_SIZE - 1);
    expect(SAMPLE_GRID_SIZE % SAMPLE_GRID_COLUMNS).toBe(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/examples/cnn/example.test.ts`
Expected: FAIL — `Failed to resolve import "./example"`.

- [ ] **Step 3: Write the implementation**

`src/lib/examples/cnn/example.ts`:

```ts
import { createBlock } from '../../network/factory';
import type { Block, BlockKind, InputBlock, LinearBlock, Network, OutputBlock } from '../../network/types';
import type { StorageKeys } from '../../persist/storage';

export const CNN_PALETTE: BlockKind[] = [
  'conv2d',
  'flatten',
  'linear',
  'relu',
  'sigmoid',
  'softmax'
];

export const CNN_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:cnn:network:v1',
  dataset: 'visnet:cnn:dataset:v1'
};

export const CNN_WEIGHTS_ID = 'cnn';

export const SAMPLE_GRID_SIZE = 40;
export const SAMPLE_GRID_COLUMNS = 8;
export const SAMPLE_GRID_CELL_WIDTH = 36;
export const SAMPLE_GRID_CELL_HEIGHT = 48;
export const SAMPLE_GRID_GAP = 4;

export function createCnnNetwork(): Network {
  const input: InputBlock = { ...(createBlock('input') as InputBlock), shape: [28, 28, 1] };
  const conv = createBlock('conv2d');
  const relu = createBlock('relu');
  const flatten = createBlock('flatten');
  const dense: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 10 };
  const softmax = createBlock('softmax');
  const output: OutputBlock = { ...(createBlock('output') as OutputBlock), units: 10 };

  return {
    version: 2,
    blocks: [input, conv, relu, flatten, dense, softmax, output],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
    positions: {}
  };
}

export function defaultSampleIndices(): number[] {
  return Array.from({ length: SAMPLE_GRID_SIZE }, (_, index) => index);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/examples/cnn/example.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/examples/cnn/example.ts src/lib/examples/cnn/example.test.ts
git commit -m "feat: add the CNN example configuration"
```

---

### Task 2: Grid helpers

**Files:**
- Create: `src/lib/examples/cnn/grid.ts`
- Test: `src/lib/examples/cnn/grid.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:

```ts
export interface GridPrediction {
  predicted: number;
  correct: boolean;
}

export function predictionsFromLogits(
  logits: number[][],
  labels: Uint8Array,
  indices: number[]
): GridPrediction[];

export function gridCellRect(
  cell: number,
  columns: number,
  cellWidth: number,
  cellHeight: number,
  gap: number
): { x: number; y: number; width: number; height: number };
```

- `predictionsFromLogits` takes the `argMax` of each row and compares it with `labels[indices[position]]`, so `predictions[i]` describes the image at `indices[i]`. A tie resolves to the lowest class index.
- `gridCellRect` returns the cell's top-left pixel and its width and height, **excluding** the gap; the gap is added only between cells.
- Both are pure. They exist so the component is only drawing.

- [ ] **Step 1: Write the failing test**

`src/lib/examples/cnn/grid.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { gridCellRect, predictionsFromLogits } from './grid';

describe('predictionsFromLogits', () => {
  it('takes the argmax and compares it with the label', () => {
    const logits = [
      [0.1, 0.7, 0.2],
      [0.6, 0.3, 0.1],
      [0.2, 0.2, 0.6]
    ];
    const labels = Uint8Array.from([1, 0, 2]);

    expect(predictionsFromLogits(logits, labels, [0, 1, 2])).toEqual([
      { predicted: 1, correct: true },
      { predicted: 0, correct: true },
      { predicted: 2, correct: true }
    ]);
  });

  it('marks a wrong prediction', () => {
    const result = predictionsFromLogits([[0.1, 0.9]], Uint8Array.from([0]), [0]);
    expect(result).toEqual([{ predicted: 1, correct: false }]);
  });

  it('reads the label through the index list', () => {
    const labels = Uint8Array.from([9, 9, 4]);
    const result = predictionsFromLogits([[0.1, 0.2, 0.3, 0.4, 0.5]], labels, [2]);
    expect(result).toEqual([{ predicted: 4, correct: true }]);
  });

  it('breaks a tie towards the lowest class', () => {
    const result = predictionsFromLogits([[0.5, 0.5, 0.5]], Uint8Array.from([0]), [0]);
    expect(result[0].predicted).toBe(0);
  });

  it('returns one entry per logits row', () => {
    expect(predictionsFromLogits([], Uint8Array.from([]), [])).toEqual([]);
  });
});

describe('gridCellRect', () => {
  const width = 36;
  const height = 48;
  const gap = 4;

  it('places the first cell at the origin', () => {
    expect(gridCellRect(0, 8, width, height, gap)).toEqual({ x: 0, y: 0, width, height });
  });

  it('advances across a row, then wraps', () => {
    expect(gridCellRect(7, 8, width, height, gap)).toEqual({
      x: 7 * (width + gap),
      y: 0,
      width,
      height
    });
    expect(gridCellRect(8, 8, width, height, gap)).toEqual({
      x: 0,
      y: height + gap,
      width,
      height
    });
  });

  it('places the last of forty cells in the final row', () => {
    expect(gridCellRect(39, 8, width, height, gap)).toEqual({
      x: 7 * (width + gap),
      y: 4 * (height + gap),
      width,
      height
    });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/examples/cnn/grid.test.ts`
Expected: FAIL — `Failed to resolve import "./grid"`.

- [ ] **Step 3: Write the implementation**

`src/lib/examples/cnn/grid.ts`:

```ts
export interface GridPrediction {
  predicted: number;
  correct: boolean;
}

export function predictionsFromLogits(
  logits: number[][],
  labels: Uint8Array,
  indices: number[]
): GridPrediction[] {
  return logits.map((row, position) => {
    let predicted = 0;
    for (let candidate = 1; candidate < row.length; candidate++) {
      if (row[candidate] > row[predicted]) predicted = candidate;
    }
    return { predicted, correct: predicted === labels[indices[position]] };
  });
}

export function gridCellRect(
  cell: number,
  columns: number,
  cellWidth: number,
  cellHeight: number,
  gap: number
): { x: number; y: number; width: number; height: number } {
  return {
    x: (cell % columns) * (cellWidth + gap),
    y: Math.floor(cell / columns) * (cellHeight + gap),
    width: cellWidth,
    height: cellHeight
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/examples/cnn/grid.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/examples/cnn/grid.ts src/lib/examples/cnn/grid.test.ts
git commit -m "feat: add the sample grid helpers"
```

---

### Task 3: Warn when the Input block disagrees with the data

**Files:**
- Modify: `src/lib/network/validate.ts`
- Modify: `src/lib/editor/networkStore.svelte.ts`
- Test: `src/lib/network/validate.test.ts` (extend)
- Test: `src/lib/editor/networkStore.svelte.test.ts` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ValidateOptions` gains `expectedInputShape?: number[]`, and `NetworkStore` gains a settable `expectedInputShape = $state<number[] | undefined>(undefined)` passed through to `validate`.

**Why:** the CNN page builds its tensors as `[n, 28, 28, 1]` from the dataset, and the model takes its input shape from the Input block. Nothing checks that the two agree, so a user who edits the Input block to, say, `[4, 4, 1]` gets an opaque TensorFlow.js shape error at training time instead of a plain-language explanation. This is the same idea as the existing `expectedClasses` option.

The new rule is a **warning**, not an error: the network is still buildable, and a mismatch is a statement about the data rather than about the network's structure. Like the existing output-units warning, it tells the user what to change.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/network/validate.test.ts`:

```ts
describe('expectedInputShape', () => {
  it('says nothing when the input matches', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(
      warnings(network, { expectedInputShape: [28, 28, 1] }).map((issue) => issue.title)
    ).not.toContain('Input shape does not match the data');
  });

  it('warns when the input shape differs', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network, { expectedInputShape: [28, 28, 1] }).find(
      (i) => i.title === 'Input shape does not match the data'
    );
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('in');
    expect(issue?.message).toContain('[4, 4, 1]');
    expect(issue?.message).toContain('[28, 28, 1]');
    expect(issue?.fix).toContain('[28, 28, 1]');
  });

  it('warns when the ranks differ', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [784] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(
      warnings(network, { expectedInputShape: [28, 28, 1] }).map((i) => i.title)
    ).toContain('Input shape does not match the data');
  });

  it('says nothing when no shape is expected', () => {
    expect(warnings(createEmptyNetwork())).toEqual([]);
  });
});
```

Note: `warnings` is the existing helper in that file, and it currently takes `(network, options?: { expectedClasses?: number })`. Widen its options type to `ValidateOptions` so the new key type-checks.

Note also why the matching case asserts *absence of the new title* rather than `toEqual([])`: a rank-3 input with no convolution already triggers the pre-existing "Image input without a Convolution layer" warning, so an empty-array assertion could never pass. The other new tests can use `toEqual`-style checks on their own titles freely.

Then extend the existing message-contract table in the same file so the new rule is covered by it. That table is the spec's guarantee that every rule produces a non-empty title, message, and fix, so a rule missing from it weakens the guarantee.

Widen the table's options type:

```ts
interface RuleCase {
  title: string;
  severity: Severity;
  network: Network;
  options?: ValidateOptions;
}
```

Add the new title to `WARNING_RULE_TITLES`:

```ts
const WARNING_RULE_TITLES = [
  'Add a Softmax for probabilities',
  'Softmax is unusual with mean squared error',
  'Softmax is not the last layer',
  'Output size does not match the data',
  'Image input without a Convolution layer',
  'Convolution layer without image input',
  'Input shape does not match the data'
];
```

And append this case to `RULE_CASES`, immediately before the closing `];`:

```ts
  {
    title: 'Input shape does not match the data',
    severity: 'warning',
    options: { expectedInputShape: [28, 28, 1] },
    network: net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ])
  }
```

That case's network also produces the pre-existing image-without-convolution warning, which is fine: the table's cross-check compares the *union* of every title produced against the full title set, and that title is already in it. The table's per-case assertion only requires the case's own title to be among the issues produced.

Append to `src/lib/editor/networkStore.svelte.test.ts`:

```ts
describe('expected input shape', () => {
  it('surfaces a mismatch as a warning', () => {
    const instance = store();
    instance.expectedInputShape = [28, 28, 1];

    expect(instance.warnings.map((issue) => issue.title)).toContain(
      'Input shape does not match the data'
    );
  });

  it('says nothing when the shape matches', () => {
    const instance = store();
    instance.expectedInputShape = [2];

    expect(instance.warnings.map((issue) => issue.title)).not.toContain(
      'Input shape does not match the data'
    );
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: FAIL — the warning never appears, and `expectedInputShape` is not a known option.

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: FAIL — `expectedInputShape` does not exist on the store.

- [ ] **Step 3: Write the implementation**

In `src/lib/network/validate.ts`, widen the options:

```ts
export interface ValidateOptions {
  expectedClasses?: number;
  expectedInputShape?: number[];
}
```

and add this rule beside the existing `expectedClasses` block, after the input block has been found:

```ts
  if (inputBlock && inputBlock.kind === 'input' && options.expectedInputShape) {
    const expected = options.expectedInputShape;
    const actual = inputBlock.shape;
    const matches =
      actual.length === expected.length && actual.every((size, i) => size === expected[i]);

    if (!matches) {
      issues.push({
        severity: 'warning',
        title: 'Input shape does not match the data',
        message: `The Input block is ${shapeText(actual)}, but the images are ${shapeText(expected)}.`,
        fix: `Set the Input block to ${shapeText(expected)}.`,
        blockId: inputBlock.id
      });
    }
  }
```

In `src/lib/editor/networkStore.svelte.ts`, add the field beside `expectedClasses`:

```ts
  expectedInputShape = $state<number[] | undefined>(undefined);
```

and pass it through:

```ts
  issues = $derived(
    validate(this.network, {
      expectedClasses: this.expectedClasses,
      expectedInputShape: this.expectedInputShape
    })
  );
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: PASS.

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify nothing else regressed**

Run: `npm test`
Expected: the full suite passes. The default network has a `[2]` input, and no existing caller passes `expectedInputShape`, so no existing assertion changes.

- [ ] **Step 6: Commit**

```bash
git add src/lib/network/validate.ts src/lib/network/validate.test.ts src/lib/editor/networkStore.svelte.ts src/lib/editor/networkStore.svelte.test.ts
git commit -m "feat: warn when the input shape disagrees with the data"
```

---

### Task 4: One runtime facade for both examples

**Files:**
- Move: `src/lib/examples/mlp/runtime.ts` → `src/lib/examples/runtime.ts`
- Modify: `src/lib/training/Trainer.ts`
- Modify: `src/routes/examples/mlp/+page.svelte`
- Test: `src/lib/training/Trainer.test.ts` (extend)

**Interfaces:**
- Consumes: `PointDataset` from `../data/points`; `ImageDataset` from `../data/images`; `Network`, `TrainingConfig` from `../network/types`; `TrainStats` from `../training/Trainer`.
- Produces:

```ts
export type Model = tf.LayersModel;
export type ModelData = { xs: tf.Tensor; ys: tf.Tensor2D };
export type TrainerHandle = import('../training/Trainer').Trainer;

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  imagesToTensors(dataset: ImageDataset, indices?: number[]): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
}

export async function loadRuntime(weightsId: string): Promise<Runtime>;
```

**Why:** the CNN page needs the same facade the MLP page uses, differing only in the weight id and in needing image conversion. Duplicating the file would mean two places to keep in step; instead the facade takes the id and gains `imagesToTensors`.

**Why the `Trainer` changes:** its data type is `{ xs: tf.Tensor2D; ys: tf.Tensor2D }`, but an image batch is `Tensor4D`. Widening `xs` to `tf.Tensor` is enough — the trainer only reads `xs.shape[0]`, gathers along axis 0, and predicts, all of which are shape-agnostic. `Tensor2D` is assignable to `tf.Tensor`, so the MLP is unaffected.

- [ ] **Step 1: Move the file**

```bash
git mv src/lib/examples/mlp/runtime.ts src/lib/examples/runtime.ts
```

- [ ] **Step 2: Write the failing test**

Append to `src/lib/training/Trainer.test.ts`:

```ts
describe('a four-dimensional feature batch', () => {
  it('trains on image-shaped inputs', async () => {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [2, 2, 1] }));
    model.add(tf.layers.dense({ units: 2 }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    const xs = tf.tensor4d(
      [0, 1, 1, 0, 1, 0, 0, 1],
      [2, 2, 2, 1]
    );
    const ys = tf.tensor2d(
      [
        [1, 0],
        [0, 1]
      ],
      [2, 2]
    );
    tensors.push(xs, ys);

    const stats: TrainStats[] = [];
    const trainer = new Trainer(model, { xs, ys }, 2, (s) => stats.push(s));

    await trainer.step();

    expect(stats).toHaveLength(1);
    expect(Number.isFinite(stats[0].batchLoss)).toBe(true);
  });
});
```

- [ ] **Step 3: Verify the test fails to type-check**

Run: `npm run check`
Expected: FAIL — `tf.Tensor4D` is not assignable to the `xs: tf.Tensor2D` parameter at the new test.

Note that `npx vitest run src/lib/training/Trainer.test.ts` will **pass** at this point. Vitest strips types, so a type-only mismatch is invisible to it, and the type check is the real RED step for this change. Do not "fix" the test to make Vitest fail.

- [ ] **Step 4: Widen the Trainer's data type**

In `src/lib/training/Trainer.ts`, change the constructor's data parameter and the field it assigns to:

```ts
    private readonly data: { xs: tf.Tensor; ys: tf.Tensor2D },
```

Both the constructor parameter and the private field declaration use that type. Nothing else in the file changes.

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run src/lib/training/Trainer.test.ts`
Expected: PASS.

- [ ] **Step 6: Rewrite the moved facade**

`src/lib/examples/runtime.ts`:

```ts
import type { ImageDataset } from '../data/images';
import type { PointDataset } from '../data/points';
import type { Network, TrainingConfig } from '../network/types';
import type { TrainStats } from '../training/Trainer';
import type * as tf from '@tensorflow/tfjs';

export type Model = tf.LayersModel;
export type ModelData = {
  xs: tf.Tensor;
  ys: tf.Tensor2D;
};
export type TrainerHandle = import('../training/Trainer').Trainer;

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  imagesToTensors(dataset: ImageDataset, indices?: number[]): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
}

export async function loadRuntime(weightsId: string): Promise<Runtime> {
  const [builder, tensors, trainerModule, weights] = await Promise.all([
    import('../tf/buildModel'),
    import('../data/tensors'),
    import('../training/Trainer'),
    import('../persist/weights')
  ]);

  return {
    buildModel: (net) => builder.buildModel(net),
    compileModel: (model, training) => builder.compileModel(model, training),
    toTensors: (dataset) => tensors.toTensors(dataset),
    imagesToTensors: (dataset, indices) => tensors.imagesToTensors(dataset, indices),
    disposeData: (data) => {
      data?.xs.dispose();
      data?.ys.dispose();
    },
    disposeModel: (model) => {
      model?.dispose();
    },
    createTrainer: (model, data, batchSize, onStats, onError) =>
      new trainerModule.Trainer(model, data, batchSize, onStats, undefined, onError),
    saveWeights: (model) => weights.saveWeights(model, weightsId),
    loadWeightsInto: (model) => weights.loadWeightsInto(model, weightsId)
  };
}
```

- [ ] **Step 7: Point the MLP page at the shared facade**

In `src/routes/examples/mlp/+page.svelte`:

- Change the import specifier from `'$lib/examples/mlp/runtime'` to `'$lib/examples/runtime'`, leaving the imported names as they are.
- Import `MLP_WEIGHTS_ID` from `'$lib/examples/mlp/example'` — that module is already imported there, so add it to the existing import list.
- Change `runtime = await loadRuntime();` to `runtime = await loadRuntime(MLP_WEIGHTS_ID);`.

Nothing else in the page changes.

- [ ] **Step 8: Verify the whole suite**

Run: `npm test`
Expected: the full suite passes. The MLP page's behaviour is unchanged because `MLP_WEIGHTS_ID` is `'main'`, the same id the facade previously hardcoded.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings` — this is the evidence that no import still points at the old path.

Run: `npm run build`
Expected: build succeeds, which is the evidence that the dynamic imports in the moved file resolve from their new directory.

- [ ] **Step 9: Commit**

```bash
git add src/lib/examples/runtime.ts src/lib/training/Trainer.ts src/lib/training/Trainer.test.ts src/routes/examples/mlp/+page.svelte
git commit -m "refactor: share one runtime facade between the examples"
```

The old path is deliberately absent from that `git add`: `git mv` already staged the rename, and naming a path that no longer exists makes git exit non-zero. Confirm with `git status --short` that the rename is staged as a rename rather than an add plus a delete.

---

### Task 5: The sample grid

**Files:**
- Create: `src/lib/components/SampleGrid.svelte`
- Create: `src/lib/components/__stubs__/SampleGridHarness.svelte` (test-only)
- Test: `src/lib/components/SampleGrid.test.ts`

**Interfaces:**
- Consumes: `ImageDataset`, `imageAt` from `../data/images`; the grid geometry constants from `../examples/cnn/example`; `gridCellRect`, `predictionsFromLogits`, `GridPrediction` from `../examples/cnn/grid`.
- Produces: `SampleGrid.svelte` with props:

```ts
{
  model: tf.LayersModel | null;
  dataset: ImageDataset | null;
  indices: number[];
  sampleXs: tf.Tensor | null;
  redrawKey: number;
  onerror?: (message: string) => void;
}
```

**Two stacked canvases, and why.** The digit images never change during training, but the borders and predicted labels change on every frame. Drawing the images into a base canvas once per dataset change, and only the marks into an overlay canvas each frame, keeps the per-frame work to `strokeRect` and `fillText` and confines the fiddly `ImageData` code to a path that runs once. It also makes the per-frame path — the one that can regress silently — testable with a fake context.

**Tensor ownership.** The page builds the sample batch and passes it in as `sampleXs`; the grid never creates or disposes tensors. That keeps every tensor's lifetime in one place, the page.

**Colours are literals.** A canvas context cannot resolve a CSS custom property, so the border colours are hex literals matching the tokens (`#10b981` for `--color-success`, `#dc2626` for `--color-error`, `#cbd5e1` for `--color-border`). This is the same duplication `src/lib/render/palette.ts` already accepts for the class colours.

**Behaviour:**
- The wrapper is `columns * cellWidth + (columns - 1) * gap` wide and `rows * cellHeight + (rows - 1) * gap` tall, where `rows = indices.length / columns`. Both canvases are sized to the device pixel ratio and scaled back down with CSS, so the digits stay crisp.
- The base layer draws each image at its cell's top-left plus a 4-pixel margin, at the dataset's own `cols × rows` size.
- The overlay draws, for every cell, a border that is green when the prediction matches the label, red when it does not, and the neutral border colour when there is no prediction; and, when there is a prediction, the predicted digit centred in the strip beneath the image.
- With no model, or no `sampleXs`, the images are drawn with neutral borders and no digits, and the caption says training has not started.
- The overlay is redrawn at most once per animation frame, driven by `redrawKey`.
- If predicting throws — which happens when the model is disposed between the frame being scheduled and the prediction running — `onerror` is called with a fixed plain-language sentence and the detail is logged, and the grid falls back to neutral borders until a model exists again.

Element test ids: `sample-grid`, `sample-grid-caption`.

- [ ] **Step 1: Write the failing test**

`src/lib/components/__stubs__/SampleGridHarness.svelte`:

```svelte
<script lang="ts">
  import type * as tf from '@tensorflow/tfjs';
  import type { ImageDataset } from '../../data/images';
  import SampleGrid from '../SampleGrid.svelte';

  let {
    model,
    dataset,
    indices,
    sampleXs,
    onerror
  }: {
    model: tf.LayersModel | null;
    dataset: ImageDataset | null;
    indices: number[];
    sampleXs: tf.Tensor | null;
    onerror?: (message: string) => void;
  } = $props();

  let redrawKey = $state(0);

  export function bump(): void {
    redrawKey += 1;
  }
</script>

<SampleGrid {model} {dataset} {indices} {sampleXs} {redrawKey} {onerror} />
```

`src/lib/components/SampleGrid.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImageDataset } from '../data/images';
import SampleGridHarness from './__stubs__/SampleGridHarness.svelte';

const SIZE = 40;

class FakeImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;

  constructor(data: Uint8ClampedArray, width: number, height: number) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

function dataset(count = SIZE): ImageDataset {
  const rows = 2;
  const cols = 2;
  return {
    count,
    rows,
    cols,
    numClasses: 3,
    pixels: Uint8Array.from({ length: count * rows * cols }, (_, i) => i % 256),
    labels: Uint8Array.from({ length: count }, (_, i) => i % 3)
  };
}

function fakeContext() {
  return {
    scale: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    font: '',
    textAlign: '',
    textBaseline: '',
    imageSmoothingEnabled: false
  };
}

let context: ReturnType<typeof fakeContext>;

beforeEach(() => {
  context = fakeContext();
  vi.stubGlobal('ImageData', FakeImageData);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as never);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('SampleGrid', () => {
  it('draws one border and no digit for every cell when there is no model', async () => {
    render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    expect(context.strokeRect).toHaveBeenCalledTimes(SIZE);
    expect(context.fillText).not.toHaveBeenCalled();
    expect(screen.getByTestId('sample-grid-caption').textContent).toContain('not started');
  });

  it('draws the predicted digit for every cell once there is a model', async () => {
    const model = {
      predict: () => ({
        arraySync: () => Array.from({ length: SIZE }, () => [0.1, 0.8, 0.1]),
        dispose: () => {}
      })
    };
    render(SampleGridHarness, {
      props: {
        model: model as never,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: {} as never
      }
    });
    await tick();

    expect(context.fillText).toHaveBeenCalledTimes(SIZE);
  });

  it('redraws when the redraw key changes', async () => {
    const view = render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    const before = context.strokeRect.mock.calls.length;
    expect(before).toBeGreaterThan(0);

    const harness = view.component as unknown as { bump: () => void };
    harness.bump();
    await tick();

    expect(context.strokeRect.mock.calls.length).toBeGreaterThan(before);
  });

  it('reports a failed prediction instead of throwing', async () => {
    const onerror = vi.fn();
    const model = {
      predict: () => {
        throw new Error('disposed');
      }
    };
    render(SampleGridHarness, {
      props: {
        model: model as never,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: {} as never,
        onerror
      }
    });
    await tick();

    expect(onerror).toHaveBeenCalledTimes(1);
    expect(context.fillText).not.toHaveBeenCalled();
  });

  it('draws the images once rather than on every frame', async () => {
    const view = render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    const before = context.drawImage.mock.calls.length;
    expect(before).toBeGreaterThan(0);

    const harness = view.component as unknown as { bump: () => void };
    harness.bump();
    await tick();

    expect(context.drawImage.mock.calls.length).toBe(before);
  });

  it('does not throw when the canvas has no 2d context', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);

    expect(() =>
      render(SampleGridHarness, {
        props: {
          model: null,
          dataset: dataset(),
          indices: Array.from({ length: SIZE }, (_, i) => i),
          sampleXs: null
        }
      })
    ).not.toThrow();
    await tick();
  });

  it('keeps the grid the expected size', async () => {
    render(SampleGridHarness, {
      props: {
        model: null,
        dataset: dataset(),
        indices: Array.from({ length: SIZE }, (_, i) => i),
        sampleXs: null
      }
    });
    await tick();

    const canvases = screen.getByTestId('sample-grid').querySelectorAll('canvas');
    expect(canvases).toHaveLength(2);
    expect(Number(canvases[0].getAttribute('width'))).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run --project ui src/lib/components/SampleGrid.test.ts`
Expected: FAIL — `Failed to resolve import "./__stubs__/SampleGridHarness.svelte"`.

- [ ] **Step 3: Write the implementation**

`src/lib/components/SampleGrid.svelte`:

```svelte
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
      Training has not started. These are test digits the network has never seen; each will show
      its predicted digit once training begins.
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run --project ui src/lib/components/SampleGrid.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify the types and the build**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/lib/components/SampleGrid.svelte src/lib/components/SampleGrid.test.ts src/lib/components/__stubs__/SampleGridHarness.svelte
git commit -m "feat: add the live digit sample grid"
```

---

### Task 6: The CNN example page

**Files:**
- Modify: `src/routes/examples/cnn/+page.svelte` (replaces the placeholder)

**Interfaces:**
- Consumes: `NetworkStore`; `ImageDataset` from `$lib/data/images`; `loadMnistData`, `ImageDataUnavailableError` from `$lib/data/mnist`; `CNN_PALETTE`, `CNN_STORAGE_KEYS`, `CNN_WEIGHTS_ID`, `createCnnNetwork`, `defaultSampleIndices` from `$lib/examples/cnn/example`; `loadRuntime`, `Model`, `ModelData`, `Runtime`, `TrainerHandle` from `$lib/examples/runtime`; `createBrowserStorage`, `NetworkStorage` from `$lib/persist/storage`; `TrainStats` from `$lib/training/Trainer`; the components `ExampleLayout`, `NetworkEditor`, `TrainingPanel`, `LossChart`, `StatsReadout`, `SampleGrid`.
- Produces: the working CNN example.

**This page is the MLP page's structure with a convolutional default, a dataset that must be fetched, and a sample grid instead of a decision boundary.** Three things are load-bearing:

1. **Tensor ownership.** The training tensors and the sample batch are owned here. Each has a plain, non-reactive mirror (`data`, `currentSampleData`) for disposal and a reactive value for the prop, because an effect that both reads and writes the same reactive value loops. This mirrors the MLP page exactly; do not "simplify" it.
2. **`$state.raw` for `runtime`, `model`, `testData` and `sampleData`.** Plain `$state` deep-proxies its value, which would wrap TensorFlow.js objects in a Proxy and break them.
3. **`builtSignature` is `$state`** so `resetModel` can force a rebuild by clearing it. `currentModel`, `data`, `trainer`, `storage` and `compiledTraining` are deliberately plain variables.

- [ ] **Step 1: Write the page**

`src/routes/examples/cnn/+page.svelte`:

```svelte
<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
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
  import {
    loadRuntime,
    type Model,
    type ModelData,
    type Runtime,
    type TrainerHandle
  } from '$lib/examples/runtime';
  import { createBrowserStorage, type NetworkStorage } from '$lib/persist/storage';
  import type { TrainStats } from '$lib/training/Trainer';

  const store = new NetworkStore(createCnnNetwork());
  store.expectedClasses = 10;
  store.expectedInputShape = [28, 28, 1];

  const sampleIndices = defaultSampleIndices();

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
  let testData = $state.raw<ImageDataset | null>(null);
  let sampleData = $state.raw<ModelData | null>(null);
  let playing = $state(false);
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let builtSignature = $state('');
  let trainData = $state.raw<ImageDataset | null>(null);
  let loadState = $state<'loading' | 'ready' | 'unavailable'>('loading');
  let redrawKey = $state(0);
  let trainCount = $state(0);

  let currentModel: Model | null = null;
  let data: ModelData | null = null;
  let currentSampleData: ModelData | null = null;
  let trainer: TrainerHandle | null = null;
  let storage: NetworkStorage | null = null;
  let compiledTraining = '';

  const architecture = $derived(JSON.stringify(store.network.blocks));
  const trainingSignature = $derived(JSON.stringify(store.network.training));

  function handleStats(next: TrainStats): void {
    stats = next;
    redrawKey += 1;
    if (next.epochMeanLoss !== null) {
      lossPoints = [...lossPoints, next.epochMeanLoss].slice(-200);
    }
  }

  function handleError(error: unknown): void {
    playing = false;
    console.error(error);
    banner = 'Training stopped because the model changed. Press Reset model and try again.';
  }

  function releaseTrainer(): void {
    trainer?.dispose();
    trainer = null;
    playing = false;
  }

  onMount(async () => {
    runtime = await loadRuntime(CNN_WEIGHTS_ID);
    storage = createBrowserStorage(CNN_STORAGE_KEYS);
    if (storage) {
      const saved = storage.loadNetwork();
      if (saved) store.load(saved);
      else if (storage.hasStoredNetwork()) {
        banner = 'The saved network could not be read, so a fresh one has been loaded.';
      }
    } else {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
    }

    try {
      const mnist = await loadMnistData();
      trainData = mnist.train;
      trainCount = mnist.train.count;
      testData = mnist.test;
      loadState = 'ready';
    } catch (error) {
      console.error(error);
      loadState = 'unavailable';
    }
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeData(currentSampleData);
    runtime?.disposeModel(currentModel);
    data = null;
    currentSampleData = null;
    currentModel = null;
    model = null;
  });

  $effect(() => {
    const api = runtime;
    if (!api) return;
    if (architecture === builtSignature) return;
    builtSignature = architecture;
    compiledTraining = trainingSignature;

    releaseTrainer();
    stats = null;
    lossPoints = [];
    api.disposeModel(currentModel);
    currentModel = null;
    model = null;

    if (!store.isValid) return;
    try {
      const built = api.buildModel(store.network);
      currentModel = built;
      model = built;
    } catch (error) {
      console.error(error);
      currentModel = null;
      model = null;
      banner = 'This network could not be built. Check the problems listed in the editor.';
    }
  });

  $effect(() => {
    const api = runtime;
    const training = trainingSignature;
    if (!api || !currentModel || training === compiledTraining) return;
    compiledTraining = training;
    api.compileModel(currentModel, store.network.training);
  });

  $effect(() => {
    const api = runtime;
    const train = trainData;
    const currentModelRef = model;
    if (!api || !train) return;

    const next = api.imagesToTensors(train);
    api.disposeData(data);
    data = next;
    releaseTrainer();

    if (!currentModelRef || !store.isValid || next.xs.shape[0] === 0) return;
    trainer = api.createTrainer(
      currentModelRef,
      next,
      store.network.training.batchSize,
      handleStats,
      handleError
    );
  });

  $effect(() => {
    const api = runtime;
    const test = testData;
    if (!api || !test) return;

    const next = api.imagesToTensors(test, sampleIndices);
    api.disposeData(currentSampleData);
    currentSampleData = next;
    sampleData = next;
  });

  $effect(() => {
    const net = store.network;
    const currentStorage = storage;
    if (!currentStorage || loadState !== 'ready') return;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(net);
      } catch {
        banner = 'Your work could not be saved. The browser storage may be full.';
      }
    }, 500);
    return () => clearTimeout(timer);
  });

  async function play(): Promise<void> {
    if (!trainer) return;
    playing = true;
    await trainer.play();
    playing = false;
  }

  function pause(): void {
    trainer?.pause();
    playing = false;
  }

  function step(): void {
    void trainer?.step();
  }

  function resetModel(): void {
    releaseTrainer();
    runtime?.disposeModel(currentModel);
    currentModel = null;
    model = null;
    stats = null;
    lossPoints = [];
    builtSignature = '';
  }

  async function save(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      await api.saveWeights(currentModel);
      banner = 'Model saved in this browser.';
    } catch {
      banner = 'The model could not be saved in this browser.';
    } finally {
      saving = false;
    }
  }

  async function load(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      const loaded = await api.loadWeightsInto(currentModel);
      banner = loaded
        ? 'Saved weights loaded.'
        : 'No saved weights match this network. Train and save again.';
    } finally {
      saving = false;
    }
  }
</script>

<ExampleLayout
  title="Handwritten digits"
  intro="Build a small convolutional network, train it on handwritten digits, and watch it get them right."
>
  {#snippet editor()}
    <NetworkEditor {store} palette={CNN_PALETTE} onsave={save} onload={load} {saving} />
  {/snippet}

  {#snippet experiment()}
    {#if banner}
      <p class="banner" role="status">{banner}</p>
    {/if}

    {#if loadState === 'loading'}
      <p class="note">Loading the digit images…</p>
    {:else if loadState === 'unavailable'}
      <div class="note">
        <h2>The digit images are not prepared</h2>
        <p>
          Run <code>npm run data:mnist</code> in the project, then reload this page. The images
          are downloaded locally and are never part of the repository.
        </p>
      </div>
    {:else}
      <p class="note">Training on {trainCount} digits, checking against {testData.count}.</p>
      <SampleGrid
        {model}
        dataset={testData}
        indices={sampleIndices}
        sampleXs={sampleData?.xs ?? null}
        {redrawKey}
        onerror={(message) => (banner = message)}
      />
    {/if}

    <TrainingPanel
      {store}
      {playing}
      disabled={!store.isValid || loadState !== 'ready'}
      onplay={play}
      onpause={pause}
      onstep={step}
      onreset={resetModel}
    />

    <LossChart points={lossPoints} />
    <StatsReadout {stats} />
  {/snippet}
</ExampleLayout>

<style>
  .banner {
    margin: 0;
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-warning) 12%, var(--color-surface));
    border: 1px solid var(--color-warning);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .note {
    display: grid;
    gap: var(--space-2);
    margin: 0;
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  .note h2 {
    margin: 0;
    font-size: var(--text-base);
    color: var(--color-text);
  }

  .note p {
    margin: 0;
  }
</style>
```

- [ ] **Step 2: Verify**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: the full suite passes; this task adds no tests.

Run: `npm run build`
Expected: build succeeds, and `build/examples/cnn.html` exists. The build must succeed **without** `static/mnist/` being present — if you want to prove that, move the directory aside, build, and move it back.

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/routes/examples/cnn/+page.svelte
git commit -m "feat: add the CNN example page"
```

---

### Task 7: Document both examples and verify

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: everything built in Tasks 1-6.
- Produces: documentation that matches the app, and a green full verification.

- [ ] **Step 1: Update `README.md`**

Replace the line saying the convolutional example is not built with a description of it, and add it to the "What works today" list. The list gains:

```markdown
- Train a small convolutional network on handwritten digits and watch a grid of test
  digits turn from wrong to right.
```

And replace the "not built yet" sentence with:

```markdown
## Digit data

The handwritten-digit example needs its data prepared once, locally:

```sh
npm run data:mnist
```

That downloads a subset of MNIST into `static/mnist/`, which is gitignored — no dataset
is committed to this repository and nothing downloads it at build or install time. Until
you run it, the example page explains what to do rather than failing. MNIST is a
derivative of the NIST Special Database 19.
```

- [ ] **Step 2: Update `AGENTS.md`**

The Status section should say both examples work: the 2D points example with its decision boundary, and the handwritten-digits example with its sample grid, whose data is prepared locally. Remove the sentence calling the CNN page a placeholder.

Add to the architecture rules:

```markdown
- The CNN example's data is prepared by `npm run data:mnist` into the gitignored
  `static/mnist/`. No build or install step downloads it; the page degrades to an
  instruction when it is absent.
```

- [ ] **Step 3: Run the full verification**

Run: `npm run lint`
Expected: no errors.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: both projects green, 0 failures, no warnings.

Run: `npm run build`
Expected: build succeeds and `build/index.html`, `build/examples/mlp.html`, and `build/examples/cnn.html` all exist.

Run: `git status --short`
Expected: clean — `static/mnist/` must not appear.

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document both examples and the digit data step"
```

---

## Plan Self-Review

**Spec coverage.** §3 (prep script) and §4 (format) were phase C1. §5.1-§5.3 (prerequisite fixes) were C1. §6 (storage keys) was C1. §7 (image data modules) was C1. §8 (the page) → Task 6, with the configuration in Task 1 and the shared runtime in Task 4. §9 (the sample grid) → Tasks 2 and 5. §10 (error handling) → Task 6's missing-asset panel, the prediction-failure path in Task 5, and the existing model-build and training error paths. §11 (testing) → each task's tests. §12 (manual checklist) → items 1-4 were C1; items 5-11 are exercised by Task 6's verification and remain for a human with a browser. §13 (documentation) → Task 7. §14 (decisions) → realised across Tasks 1-6.

**Gap found and closed during this review.** The spec's §9 describes `gridCellRect(cell, columns, cellSize, gap)` returning a `size`, and the cells as square. The plan uses `gridCellRect(cell, columns, cellWidth, cellHeight, gap)` returning `width` and `height`, because a cell needs room for the predicted digit beneath the image. The spec should be updated to match: cells are 36×48, not square. This is a refinement of the spec, not a deviation from its intent.

**Second gap found and closed.** The spec does not mention checking the Input block against the dataset's dimensions, which the previous phase's final review flagged as a latent problem: a user editing the Input block would get an opaque TensorFlow.js shape error. Task 3 adds that warning and the spec should record it in §10.

**Placeholder scan.** No "TBD", "TODO", "similar to Task N", or steps that describe work without showing it. Every code step carries complete code, and every command has an expected result.

**Type consistency.** `Runtime` is defined once in Task 4 and consumed by Task 6. `ModelData`'s `xs` is `tf.Tensor` after Task 4, which is what lets both a `Tensor2D` point batch and a `Tensor4D` image batch flow through it. `GridPrediction` and `gridCellRect` are defined in Task 2 and consumed by Task 5. The grid geometry constants are defined in Task 1 and consumed by Tasks 2's tests and 5. `CNN_STORAGE_KEYS`, `CNN_WEIGHTS_ID` and `createCnnNetwork` are defined in Task 1 and consumed by Task 6. `expectedInputShape` is defined in Task 3 and set by Task 6. The `Network` fixtures in Task 3's tests match the version 2 model, including `positions`.

