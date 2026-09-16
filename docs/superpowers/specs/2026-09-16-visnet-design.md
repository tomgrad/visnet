# VisNet — Design

Date: 2026-09-16
Status: approved (pending written-spec review)

## 1. Overview

VisNet is a static web app for teaching neural networks. Users build a network by
dragging blocks onto a node canvas, connect them into a pipeline, configure each
block, then train the network against a dataset and watch the result update live.

The first shippable version delivers one complete teaching loop: a visual editor
plus 2D point classification, where the user builds an MLP, trains it, and sees a
live decision boundary. Convolutional networks on MNIST follow in a later phase.

### Goals

- Build a network visually from blocks, with drag-and-drop and connectable ports.
- Train it in the browser with TensorFlow.js and watch learning happen live.
- Teach the relationship between network shape and behaviour through explicit
  tensor-shape annotations.
- Keep the editor a self-contained, reusable component so example pages can embed
  it next to their own dataset and visualization.
- Ship as a fully static site with no backend.

### Non-goals (this phase)

- General DAGs, branching, skip connections, or multi-input blocks. Topology is a
  linear chain.
- The CNN / MNIST example. The type system and model builder accommodate it; the
  page does not ship yet.
- Training in a Web Worker, model export/import to files, sharing via URL,
  multiple named saved networks, regression tasks, or custom layer authoring.

## 2. Stack and tooling

| Concern | Choice |
| --- | --- |
| Framework | SvelteKit 2, Svelte 5 (runes), TypeScript strict |
| Build | Vite, `@sveltejs/adapter-static`, all routes prerendered |
| Package manager | npm |
| Canvas | `@xyflow/svelte` (Svelte Flow) 1.6.x, MIT |
| ML runtime | `@tensorflow/tfjs` |
| Styling | Plain CSS + design tokens, component-scoped styles |
| Charts | Hand-rolled SVG (no charting dependency) |
| Tests | Vitest |
| Lint/format | ESLint + Prettier |

Output is a directory of static files. There is no server component, no API, and
no persistence beyond the browser.

### SSR boundary

SvelteKit prerenders page shells at build time, but TF.js and Svelte Flow require
a browser. All TF.js access is confined to `src/lib/tf/` and `src/lib/training/`,
which are imported dynamically from browser-only code paths (`onMount` or an
`$effect` that checks `browser`). Svelte Flow is rendered only after mount.
Prerendering therefore succeeds and pages hydrate client-side without
`ssr = false`.

### Scripts

`dev`, `build`, `preview`, `check` (svelte-check), `lint`, `format`, `test`.

## 3. Architecture

```
                     ┌────────────────────────────────────────────┐
                     │  Example page (owns dataset + experiment)  │
                     │  e.g. /examples/mlp                        │
                     └───────────────┬────────────────────────────┘
                                     │ Network (value) + onchange
                     ┌───────────────▼────────────────────────────┐
                     │ NetworkEditor.svelte  (controlled, embed)   │
                     │  BlockPalette · BlockCanvas · Inspector     │
                     │  IssuesPanel                               │
                     └───────────────┬────────────────────────────┘
                                     │ reads/writes
                     ┌───────────────▼────────────────────────────┐
                     │ editor/networkStore.svelte.ts (runes)       │
                     └───────────────┬────────────────────────────┘
                                     │ delegates to
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼─────────┐        ┌─────────▼────────┐
│ network/       │          │ editor/flow.ts   │        │ persist/         │
│  pure domain   │          │  model ⇄ canvas  │        │  storage, weights│
│  (no Svelte,   │          │  projection      │        └──────────────────┘
│   no TF.js)    │          └──────────────────┘
└───────┬────────┘
        │ consumed by
┌───────▼────────┐   ┌──────────────────┐   ┌──────────────────┐
│ tf/buildModel  │──▶│ training/Trainer │──▶│ render/boundary  │
└────────────────┘   └──────────────────┘   └──────────────────┘
```

The load-bearing rule: **`src/lib/network/` is pure.** No Svelte, no TF.js, no
DOM. It is the single source of truth for what a network is and whether it is
valid. Everything else derives from it.

## 4. Domain model

`src/lib/network/types.ts`:

```ts
export type BlockKind =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'flatten'
  | 'relu'
  | 'sigmoid'
  | 'softmax'
  | 'output';

interface BlockBase {
  id: string;
}

export interface InputBlock extends BlockBase {
  kind: 'input';
  shape: number[]; // e.g. [2] for MLP, [28, 28, 1] for CNN
}

export interface LinearBlock extends BlockBase {
  kind: 'linear';
  units: number;
}

export interface Conv2dBlock extends BlockBase {
  kind: 'conv2d';
  filters: number;
  kernelSize: number;
  stride: number;
  padding: 'same' | 'valid';
}

export interface FlattenBlock extends BlockBase {
  kind: 'flatten';
}

export interface ActivationBlock extends BlockBase {
  kind: 'relu' | 'sigmoid' | 'softmax';
}

export interface OutputBlock extends BlockBase {
  kind: 'output';
  units: number; // number of classes
}

export type Block =
  | InputBlock
  | LinearBlock
  | Conv2dBlock
  | FlattenBlock
  | ActivationBlock
  | OutputBlock;

export interface TrainingConfig {
  loss: 'mse' | 'crossEntropy';
  optimizer: 'sgd' | 'adam';
  learningRate: number;
  batchSize: number;
}

export interface Network {
  version: 1;
  blocks: Block[];
  training: TrainingConfig;
}
```

### Structural invariants

- Exactly one `input` block, first in the array.
- Exactly one `output` block, last in the array.
- Neither the input nor the output block can be deleted or moved.
- Block ids are unique within a network (crypto random, generated in `factory.ts`).

### The `output` block is a marker, not a layer

The `output` block declares the target shape and class count; it adds **no**
TensorFlow.js layer. The last non-marker block is what actually produces the
output. This keeps softmax placement natural (it belongs on the last real layer)
and makes the terminal node a readable statement of what the network predicts.
Shape inference warns when the last real layer's shape does not match the output
block's `units`.

### Defaults

`factory.ts` provides defaults so every block is immediately valid:

| Block | Defaults |
| --- | --- |
| `input` | `shape: [2]` |
| `linear` | `units: 8` |
| `conv2d` | `filters: 8, kernelSize: 3, stride: 1, padding: 'same'` |
| `relu` / `sigmoid` / `softmax` / `flatten` | none |
| `output` | `units: 2` |

`createEmptyNetwork()` returns:

```ts
{
  version: 1,
  blocks: [input([2]), linear(8), relu(), linear(2), softmax(), output(2)],
  training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
}
```

This default is a working MLP for the 2D example, so the app is useful on first
load.

## 5. Pure network modules

`src/lib/network/` — no Svelte, no TF.js, no DOM.

**`factory.ts`** — `createBlock(kind)`, `createEmptyNetwork()`, `cloneNetwork(net)`.

**`chain.ts`** — immutable chain operations. Each returns a new `Network`.
- `insertAt(net, index, block)` — rejects indices that would place a block before
  the input or after the output.
- `moveBlock(net, fromIndex, toIndex)` — clamps to the interior; input and output
  never move.
- `removeBlock(net, id)` — refuses to remove input or output.
- `replaceBlock(net, id, patch)` — used by the inspector for parameter edits.

**`inferShapes.ts`** — `inferShapes(net): ShapeResult` walks the chain and returns
the shape produced by each block. Rules:

| Block | Output shape |
| --- | --- |
| `input` | `block.shape` |
| `linear` | `[units]` |
| `conv2d` | `[ceil(H/stride), ceil(W/stride), filters]` for `same`; `[floor((H - kernelSize)/stride) + 1, …]` for `valid` |
| `flatten` | `[product(shape)]` |
| `relu` / `sigmoid` / `softmax` | unchanged |
| `output` | unchanged (marker) |

**`validate.ts`** — `validate(net, options?): Issue[]`, where
`Issue = { severity: 'error' | 'warning'; blockId?: string; message: string }`.

Errors (block training):
- missing input or output block; more than one of either
- fewer than one layer between input and output
- `linear` receiving a rank-3 tensor (needs a `flatten` first)
- `conv2d` receiving a non-rank-3 tensor
- `flatten` receiving a rank-1 tensor
- `conv2d` producing a zero or negative spatial dimension (kernel/stride too large)
- unknown block kind

Warnings (do not block training):
- cross-entropy loss without a `softmax` block
- mean squared error loss with a `softmax` block
- `softmax` not on the last real layer
- output block `units` differing from the dataset's class count, when the caller
  passes `expectedClasses`
- `input.shape` rank 3 with no `conv2d` block, or rank 1 with a `conv2d` block

**`serialize.ts`** — `toJSON(net): string`, `fromJSON(raw: string): Network | null`.
`fromJSON` validates the version field, runs `migrate(raw)` when the version is
older, and returns `null` for corrupt data or a version newer than supported so
callers can fall back to the default network with a notice.

## 6. Reactive store

`src/lib/editor/networkStore.svelte.ts` — a Svelte 5 runes class:

```ts
class NetworkStore {
  network = $state<Network>(createEmptyNetwork());
  selectedBlockId = $state<string | null>(null);

  issues = $derived(validate(this.network, { expectedClasses: this.expectedClasses }));
  errors = $derived(this.issues.filter((i) => i.severity === 'error'));
  warnings = $derived(this.issues.filter((i) => i.severity === 'warning'));
  shapes = $derived(inferShapes(this.network));
  isValid = $derived(this.errors.length === 0);

  // mutators delegate to network/chain.ts, then bump a revision counter
}
```

`expectedClasses` is settable by the embedding page (2 for the MLP example) and
defaults to `undefined`. The store holds no rules of its own — it is a thin
reactive shell over the pure modules.

## 7. Canvas projection

`src/lib/editor/flow.ts` converts between the domain model and Svelte Flow:

- `toFlow(net): { nodes: Node[]; edges: Edge[] }` — one node per block, positioned
  left-to-right from the chain index, with one edge between each pair of adjacent
  blocks. Positions are computed, never stored.
- `connectionToIntent(connection, net): ChainOp | null` — converts a user-drawn
  wire into an operation on the array:
  - dragging block A's output onto block B's input where A precedes B → move A to
    B's index
  - dragging A's output onto B's input where A follows B → move A to B's index - 1
  - self-loops, output-as-source, input-as-target, and connections between already
    adjacent blocks → `null` (ignored)

Because edges are always derived, the graph cannot drift out of sync with the
network. A wire is a gesture that reorders an array, not stored state.

### Canvas behaviour

- Pan and zoom are enabled (Svelte Flow defaults), with a fit-view control.
- Blocks are draggable; dropping a block onto a wire or between neighbours
  reorders the chain.
- Clicking a block selects it and opens the inspector.
- Each block node shows its kind, its most important parameter, a tensor-shape
  badge, and a delete affordance (hidden for input/output).
- The palette adds a block by click (appends after the selection, or before the
  output) or by dragging onto the canvas. A drag-drop is resolved to an index by
  finding the gap between existing blocks whose centre is nearest the drop point;
  drops outside the interior clamp to the first or last interior slot.

## 8. Model building

`src/lib/tf/buildModel.ts` — `buildModel(net): tf.Sequential`.

Refuses to build when validation reports errors (throws `NetworkInvalidError`).
Skips the `input` block and uses its `shape` as `inputShape` on the first layer
actually added. Skips the `output` block.

| Block | TF.js layer |
| --- | --- |
| `linear` | `tf.layers.dense({ units })` |
| `conv2d` | `tf.layers.conv2d({ filters, kernelSize, strides, padding, activation: 'linear' })` |
| `flatten` | `tf.layers.flatten()` |
| `relu` / `sigmoid` / `softmax` | `tf.layers.activation({ activation: kind })` |
| `output` | nothing |

Then `model.compile(...)`:

| Config | Mapping |
| --- | --- |
| `optimizer: 'sgd'` | `tf.train.sgd(learningRate)` |
| `optimizer: 'adam'` | `tf.train.adam(learningRate)` |
| `loss: 'mse'` | `'meanSquaredError'` |
| `loss: 'crossEntropy'` | `'categoricalCrossentropy'` |

### Architecture vs. compilation

Changing training config calls `model.compile()` only, which preserves weights.
Changing architecture rebuilds the model and discards weights. The UI must
therefore warn before an architecture change when the model has been trained, and
must dispose the previous model on every rebuild.

## 9. Training

`src/lib/training/Trainer.ts` — runs on the main thread, deliberately.

```ts
interface TrainStats {
  epoch: number;
  batch: number;
  batchLoss: number;
  epochMeanLoss: number;
}

class Trainer {
  constructor(
    model: tf.LayersModel,
    data: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    batchSize: number,
    onStats: (s: TrainStats) => void,
  );
  play(): void;
  pause(): void;
  step(): Promise<void>;
  dispose(): void;
}
```

- One animation frame performs exactly one `trainOnBatch` call, then one decision
  boundary render, then `await tf.nextFrame()`. At ~60 fps this yields roughly 60
  training steps per second — fast enough to watch, slow enough to follow.
- Batches are drawn by shuffling an index array each epoch and using `tf.gather`
  to slice the dataset tensors. `trainOnBatch` returns a scalar loss, read once
  with `dataSync()` and disposed inside the same `tf.tidy`.
- Epoch bookkeeping: `batchesPerEpoch = ceil(N / batchSize)`; the epoch counter
  advances when the batch counter wraps, and the mean of that epoch's batch
  losses is recorded for the chart.
- `step()` runs one batch without entering the play loop.

**Why the main thread:** the models are tiny, `tf.nextFrame()` yields to the
browser so the UI stays responsive, and a worker would require running TF.js in
the worker plus marshalling weights and dataset tensors. The `Trainer` interface
is narrow enough to move into a worker later without changing callers.

## 10. Data (2D points)

`src/lib/data/rng.ts` — `mulberry32(seed)`, a seeded PRNG so generators and tests
are reproducible.

`src/lib/data/points.ts`:

```ts
interface Point { x: number; y: number; label: 0 | 1 }
interface PointDataset { points: Point[]; numClasses: 2 }
```

- Domain is `x, y ∈ [-1, 1]`, matching the rendering surface.
- Generators, each taking a point count and a seed: `twoGaussians`, `spirals`,
  `xor`, `circles`.
- `addPoint(dataset, x, y, label)` and `clearPoints(dataset)` are pure.
- `toTensors(dataset): { xs: tf.Tensor2D; ys: tf.Tensor2D }` — `xs` is `[N, 2]`
  float32, `ys` is `[N, 2]` one-hot float32. Tensors are rebuilt when the dataset
  changes and disposed when replaced.

## 11. Decision boundary rendering

`src/lib/render/boundary.ts`:

- A 64×64 grid is sampled over `[-1, 1]²`, forwarded through the model in a single
  batched `predict`, argmax'd into class indices, and written into an offscreen
  64×64 `ImageData` using class colors from the design tokens.
- The offscreen image is drawn onto the visible canvas scaled up with
  `imageSmoothingEnabled = true`, which gives a soft boundary without per-pixel
  cost on the display canvas.
- Data points are drawn on top: filled circle, white stroke, radius 4 px, colored
  by class.
- The canvas handles click-to-add: a click maps client coordinates to the
  `[-1, 1]²` domain and appends a point with the currently selected class.
- The boundary is recomputed at most once per animation frame while training, and
  once after every model rebuild, step, pause, or dataset change. If the model is
  invalid, the canvas renders points on a neutral background and no boundary.

## 12. Loss chart

`src/lib/components/LossChart.svelte` — an SVG polyline of per-epoch mean loss, a
rolling window of the most recent 200 epochs, with min/max axis labels and the
latest value shown. No dependency; roughly 40 lines.

## 13. Persistence

**Architecture** — `localStorage`, key `visnet:network:v1`, storing
`{ version: 1, network }` via `serialize.toJSON`. Autosaved, debounced ~500 ms
after a change. On load, `fromJSON` is used; corrupt or unsupported data falls
back to `createEmptyNetwork()` with a notice.

**Weights** — IndexedDB via TF.js's built-in handler:
`model.save('indexeddb://visnet/weights/main')` and
`tf.loadLayersModel('indexeddb://visnet/weights/main')`. Weights are Float32
arrays; localStorage's ~5 MB *string* quota is a poor fit and TF.js offers no
localStorage weight handler. Architecture and weights share a single network id,
and the UI presents one "Save model" / "Load model" pair, so the split is
invisible to users.

**Dataset** — the MLP example persists its point set to `localStorage` under
`visnet:mlp:dataset:v1` (a small JSON array), so reloading restores both the
network and the data it was trained on.

Named/multiple saved networks are deferred.

## 14. Error handling

- Validation errors disable training; the Issues panel lists each error with the
  offending block highlighted and selectable.
- Warnings are shown but do not block.
- `buildModel` failures are caught: the error is surfaced in a banner, the model
  is disposed, training stays disabled.
- Backend selection tries WebGL and falls back to the CPU backend, showing a
  one-line notice when it does.
- Resource hygiene: rebuilds dispose the previous model and its tensors;
  training wraps tensor creation in `tf.tidy`; route teardown disposes the model,
  dataset tensors, and render resources to avoid leaking WebGL contexts on
  navigation.
- Storage failures (quota exceeded, IndexedDB unavailable) are non-fatal: state
  stays in memory and the user sees a notice.
- Illegal canvas connections are rejected at `connectionToIntent` and produce no
  change.

## 15. Routes and the embedding contract

- `/` — landing page: what VisNet is, and links to the examples.
- `/examples/mlp` — the complete MVP experience.
- `/examples/cnn` — placeholder describing the upcoming CNN phase.

`NetworkEditor.svelte` is the embedding contract:

```ts
interface Props {
  network: Network;          // controlled value
  palette: BlockKind[];      // which blocks this example permits
  expectedClasses?: number;  // drives the output-units warning
  readonly?: boolean;
  onchange: (net: Network) => void;
}
```

The editor knows nothing about datasets, training, or visualization. An example
page owns its dataset, its training panel, and its visualization, and composes
them around `NetworkEditor`. Adding a new example means adding a route, not
touching the editor.

`ExampleLayout.svelte` is the shared page shell (editor on one side, experiment
panel on the other) so example pages stay small.

## 16. Testing strategy

Vitest, run with `npm test`. Tests live beside their modules as `*.test.ts`.

Covered:
- `chain.ts` — insert/move/remove/replace, including refusal to move or delete
  input and output and index clamping.
- `inferShapes.ts` — each block kind, conv output sizes for `same` and `valid`,
  flatten, and the MLP default chain.
- `validate.ts` — every error and warning rule, including the `expectedClasses`
  warning.
- `serialize.ts` — round-trip equality, corrupt input, unsupported version,
  migration seam.
- `flow.ts` — `toFlow` node/edge derivation and `connectionToIntent` for legal
  reorders and every illegal case.
- `points.ts` / `rng.ts` — seeded generators are deterministic; one-hot encoding
  is correct; `addPoint`/`clearPoints` are pure.
- `buildModel.ts` — on the CPU backend in Node: a valid MLP builds with the
  expected number of layers, input shape `[null, 2]`, output shape `[null, 2]`;
  an invalid network throws; changing only training config preserves weights while
  changing architecture does not.

Not covered by automated tests in this phase: the Svelte components, the canvas
interaction, and the training animation. These are verified manually.

## 17. Manual verification checklist

1. `npm run dev`, open `/examples/mlp`; the default network renders as a
   left-to-right pipeline with shape badges.
2. Add, delete, and drag blocks; the pipeline reorders and edges stay consistent.
3. Try to delete the input and output blocks — both are refused.
4. Drag a wire into an illegal configuration — nothing changes.
5. Pick each dataset generator; points appear and the boundary renders.
6. Click the canvas to add points; change the selected class and add more.
7. Press play — the loss curve descends and the boundary visibly adapts.
8. Press pause, then step — exactly one batch of progress.
9. Break the network (e.g. delete the softmax with cross-entropy selected) — a
   warning appears and training still works.
10. Make the network invalid (e.g. a linear layer directly after a rank-3 input)
    — training is disabled and the error names the block.
11. Reload the page — network, dataset, and (after Save model) weights are
    restored and the boundary redraws.
12. Resize the window and navigate away and back — no WebGL context warnings in
    the console.

## 18. Implementation order

1. Scaffold SvelteKit + TypeScript + tooling, design tokens, ESLint/Prettier,
   Vitest wiring. Verify `dev`, `build`, `check`, `test` all run.
2. `src/lib/network/` pure modules with their tests.
3. `src/lib/tf/buildModel.ts` and `src/lib/training/Trainer.ts` with tests.
4. `src/lib/data/` and `src/lib/render/boundary.ts`.
5. Editor UI: `BlockNode`, `BlockPalette`, `BlockCanvas`, `InspectorPanel`,
   `IssuesPanel`, `NetworkEditor`.
6. `/examples/mlp`: dataset controls, `TrainingPanel`, `LossChart`,
   `DecisionBoundary`, and the wiring that ties them to the editor.
7. `src/lib/persist/` and the save/load UI.
8. Landing page, `/examples/cnn` placeholder.
9. Update `AGENTS.md` and `README.md` to match this design; run `lint`, `check`,
   and `test` to green.

## 19. Deferred work

- CNN / MNIST example, including enabling the `flatten` block in the palette.
- Training in a Web Worker behind the existing `Trainer` interface.
- Multiple named saved networks.
- SGD momentum, learning-rate schedules, and other optimizer options.
- Model export/import as files, and share-via-URL.
- Component and end-to-end tests.

## 20. Decisions and deviations

Recorded so later readers know these were deliberate.

1. **SvelteKit replaces the literal "Svelte + Vite" in `AGENTS.md`.** Needed for
   example subpages and a static, prerendered build. `AGENTS.md` will be updated.
2. **Softmax is added to the README's module list.** Cross-entropy needs it, and a
   visible block is more honest than hidden machinery.
3. **Linear chain only.** No DAGs, branching, or multi-input blocks.
4. **The `output` block is a marker that adds no layer.** Keeps softmax on the last
   real layer and makes the terminal node declarative.
5. **Auto-layout pipeline.** Node positions derive from chain index and are never
   stored; wiring gestures reorder the array.
6. **Weights in IndexedDB, architecture in localStorage.** Quota and Float32
   considerations; the UI presents one save/load action.
7. **Training on the main thread.** Simplicity now, with a documented path to a
   worker.
8. **`flatten` exists in the type system and model builder but is not in the MVP
   palette.** Avoids dead UI while removing rework when the CNN example lands.
