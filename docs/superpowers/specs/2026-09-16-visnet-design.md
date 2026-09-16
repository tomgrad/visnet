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
- Teach the relationship between network shape and behaviour through tensor shapes
  that are visible at every point in the pipeline.
- Be genuinely beginner-friendly: sensible defaults, plain-language explanations
  for every block and parameter, and error messages that say what is wrong and how
  to fix it.
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

## 2. User experience principles

These are requirements, not aspirations. They are the reason several later
decisions look the way they do.

1. **Nothing is hidden.** The shape of the data flowing through the network is
   visible at every point: on every block, on every connection, and as a complete
   table of the whole pipeline.
2. **No block stores its input size.** Every block's input is derived from the
   previous block's output. Users describe their data once, at the Input block,
   and the rest follows automatically.
3. **Every parameter is explained.** Each block and each parameter has a one-line
   plain-language description, shown in the palette, the inspector, and on hover.
4. **Errors are instructions, not verdicts.** Every error and warning names what
   is wrong in ordinary language and states exactly how to fix it. No error is a
   bare type name or a stack trace.
5. **No dead ends.** The app opens on a working network. Invalid states are
   explainable and recoverable, never a blank screen or a disabled button with no
   reason given.
6. **Automatic corrections are announced.** If a parameter must change to stay
   valid, the change is applied and the user is told what changed and why.
7. **Mistakes are cheap.** Undo and redo cover every structural and parameter edit,
   including drag-and-drop reorders.
8. **Feedback is immediate.** Shapes, warnings, and the decision boundary update as
   soon as the network changes.

## 3. Stack and tooling

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
a browser. All TF.js access is confined to `src/lib/tf/`, `src/lib/training/`,
`src/lib/data/tensors.ts`, and `src/lib/render/boundary.ts`, which are imported
dynamically from browser-only code paths (`onMount` or an
`$effect` that checks `browser`). Svelte Flow is rendered only after mount.
Prerendering therefore succeeds and pages hydrate client-side without
`ssr = false`.

### Scripts

`dev`, `build`, `preview`, `check` (svelte-check), `lint`, `format`, `test`.

## 4. Architecture

```
                     ┌────────────────────────────────────────────┐
                     │  Example page (owns dataset + experiment)  │
                     │  e.g. /examples/mlp                        │
                     └───────────────┬────────────────────────────┘
                                     │ Network (value) + onchange
                     ┌───────────────▼────────────────────────────┐
                     │ NetworkEditor.svelte  (controlled, embed)   │
                     │  Toolbar · Palette · Canvas · Inspector     │
                     │  ShapeTable · IssuesPanel                   │
                     └───────────────┬────────────────────────────┘
                                     │ reads/writes
                     ┌───────────────▼────────────────────────────┐
                     │ editor/networkStore.svelte.ts (runes)       │
                     │ editor/history.ts (undo/redo)        │
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
DOM. It is the single source of truth for what a network is, what shape flows
through each block, and whether it is valid. Everything else derives from it.

## 5. Domain model

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

export interface NodePosition {
  x: number;
  y: number;
}

export interface Network {
  version: 2;
  blocks: Block[];
  training: TrainingConfig;
  positions: Record<string, NodePosition>; // block id -> where the user put it
}
```

**Note what is absent: no block stores its input dimension.** A `LinearBlock`
stores only `units`. Its input size comes from whatever precedes it. This is the
data-model expression of principle 2, and it is also exactly how TensorFlow.js
works — each layer infers its input size from the previous layer's output, and
only the very first layer needs an explicit `inputShape`, which comes from the
Input block.

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
Shape inference reports an **error** when the last real layer's shape does not match
the output block's `units` (see the catalog below). It cannot be a warning: the
labels are one-hot with `numClasses` columns, so a mismatched output width makes
training fail at runtime, and an error blocks training before that can happen.

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
load and a new user starts from something that already trains.

### Block explanations

Every block kind has a one-line plain-language description, defined once in
`src/lib/network/descriptions.ts` and used by the palette, the inspector, and the
canvas tooltip:

| Kind | Description |
| --- | --- |
| `input` | "Describes the shape of one example your network receives." |
| `linear` | "Learns a weighted sum of its inputs. Also called a fully connected or dense layer." |
| `conv2d` | "Slides small filters over an image to detect local patterns such as edges." |
| `flatten` | "Turns image-shaped data into a flat list so Linear layers can read it." |
| `relu` | "Keeps positive values and turns negative ones into zero. Helps the network learn curved patterns." |
| `sigmoid` | "Squashes each value into the range 0 to 1." |
| `softmax` | "Turns raw scores into probabilities that add up to 1." |
| `output` | "Declares what the network predicts and how many classes there are." |

Each parameter has a one-line description in the same module, e.g. `units`: "How
many numbers this layer produces.", `kernelSize`: "How large the window sliding
over the image is.", `learningRate`: "How big each learning step is. Smaller is
slower but steadier.".

## 6. Pure network modules

`src/lib/network/` — no Svelte, no TF.js, no DOM.

**`factory.ts`** — `createBlock(kind)`, `createEmptyNetwork()`, `cloneNetwork(net)`.

**`chain.ts`** — immutable chain operations. Each returns a new `Network`.
- `insertAt(net, index, block)` — rejects indices that would place a block before
  the input or after the output.
- `moveBlock(net, fromIndex, toIndex)` — clamps to the interior; input and output
  never move.
- `removeBlock(net, id)` — refuses to remove input or output.
- `replaceBlock(net, id, patch)` — used by the inspector for parameter edits.

**`inferShapes.ts`** — the source of every shape the UI displays.

```ts
export interface ShapeInfo {
  blockId: string;
  inShape: number[] | null;   // null when the shape is unknown (earlier error)
  outShape: number[] | null;
  paramCount: number | null;  // trainable parameters this block contributes
}

export function inferShapes(net: Network): {
  perBlock: ShapeInfo[];
  edges: { fromId: string; toId: string; shape: number[] | null }[];
}
```

Shape rules:

| Block | Output shape |
| --- | --- |
| `input` | `block.shape` |
| `linear` | `[units]` |
| `conv2d` | `[ceil(H/stride), ceil(W/stride), filters]` for `same`; `[floor((H - kernelSize)/stride) + 1, …]` for `valid` |
| `flatten` | `[product(shape)]` |
| `relu` / `sigmoid` / `softmax` | unchanged |
| `output` | unchanged (marker) |

`paramCount` is computed per block so the inspector can show, for example, "this
layer has 24 trainable numbers", and so a whole-network total can be displayed.
This is informative for teaching and costs nothing.

**`validate.ts`** — `validate(net, options?): Issue[]`, where

```ts
export interface Issue {
  severity: 'error' | 'warning';
  title: string;          // short, plain language, e.g. "Linear layer needs flat input"
  message: string;        // what is wrong, in ordinary words
  fix: string;            // what to do about it, imperative
  blockId?: string;       // offending block, if any
}
```

Every issue carries a `title`, a `message`, and a `fix`. There is no issue shape
that permits a message without a fix — that is enforced by the type.

### Message catalog

Errors (block training):

| Condition | Title | Message | Fix |
| --- | --- | --- | --- |
| No input block | "Missing Input block" | "A network needs exactly one Input block to describe the shape of the data it receives." | "Add an Input block to the start of the network." |
| No output block | "Missing Output block" | "A network needs exactly one Output block to say what it predicts." | "Add an Output block to the end of the network." |
| More than one input | "More than one Input block" | "There are {n} Input blocks, but a network can only have one." | "Delete the extra Input blocks." |
| More than one output | "More than one Output block" | "There are {n} Output blocks, but a network can only have one." | "Delete the extra Output blocks." |
| No layers between input and output | "Nothing to learn" | "The Input connects straight to the Output, so there are no layers for the network to learn with." | "Add at least one layer, such as a Linear layer, between Input and Output." |
| `linear` on rank-3 input | "Linear layer needs a flat list" | "This Linear layer receives {shape}, which is image-shaped. Linear layers need a flat list of numbers." | "Add a Flatten layer before this Linear layer." |
| `conv2d` on non-rank-3 input | "Convolution layer needs image data" | "This Convolution layer receives {shape}. It expects image data shaped [height, width, channels]." | "Give the Input block a 3D shape such as [28, 28, 1], or remove the Convolution layer." |
| `flatten` on rank-1 input | "Nothing to flatten" | "This Flatten layer receives {shape}, which is already a flat list." | "Remove this Flatten layer, or move it after a Convolution layer." |
| `conv2d` output dimension ≤ 0 | "Kernel is larger than the image" | "A {kernelSize}×{kernelSize} kernel with stride {stride} leaves no room to slide over a {H}×{W} image." | "Use a smaller kernel or stride, or set padding to 'same'." |
| Unknown block kind | "Unrecognised block" | "This network contains a block type this version of VisNet does not understand ({kind})." | "Delete the block, or reset the network to start fresh." |
| Last real layer is not rank 1 | "Output must be a list of scores" | "The last layer before the Output produces {shape}, which is not a list of class scores." | "End the network with a Linear layer so the Output is a list of numbers." |
| Last real layer width ≠ Output units | "Last layer size does not match the Output block" | "The last layer produces {n} numbers, but the Output block says {units} classes." | "Set the last layer to {units} units, or change the Output block to {n}." |

Warnings (do not block training):

| Condition | Title | Message | Fix |
| --- | --- | --- | --- |
| Cross-entropy without softmax | "Add a Softmax for probabilities" | "Cross-entropy works best when the network's outputs are probabilities, but the network currently ends with raw scores. Training will still run, but it may be less stable." | "Add a Softmax block after the last Linear layer." |
| MSE with softmax | "Softmax is unusual with mean squared error" | "Mean squared error is normally used with raw scores, not probabilities." | "Switch the loss to cross-entropy, or remove the Softmax block." |
| Softmax not last | "Softmax is not the last layer" | "This Softmax block is followed by more layers, so the probabilities it produces get transformed again." | "Move the Softmax block to just before the Output block." |
| Output units ≠ dataset classes | "Output size does not match the data" | "The Output block says {units} classes, but the dataset has {n}." | "Set the Output block to {n} units." |
| Rank-3 input, no conv2d | "Image input without a Convolution layer" | "The Input block is image-shaped {shape}, but the network has no Convolution layer to look at it." | "Add a Convolution layer, or change the Input shape to a flat list." |
| Rank-1 input with conv2d | "Convolution layer without image input" | "The network has a Convolution layer, but the Input block is a flat list {shape}." | "Set the Input shape to 3D such as [28, 28, 1], or remove the Convolution layer." |

Messages are composed by small functions so values like `{shape}` and `{n}` are
substituted at validation time, and so each message can be unit-tested.

**`descriptions.ts`** — block and parameter descriptions (section 5).

**`serialize.ts`** — `toJSON(net): string`, `fromJSON(raw: string): Network | null`.
`fromJSON` is the trust boundary for data read back from browser storage, so it
validates structure, not semantics: the version field; each block's `id` and
`kind`; every kind's required parameters as finite positive numbers (with
`padding` restricted to `'same'` or `'valid'`); the training enums with finite
positive `learningRate` and `batchSize` (so `NaN` and `Infinity` are rejected,
not just non-numbers); and the structural invariants that there are at least two
blocks, exactly one `input` first and exactly one `output` last. It returns
`null` for corrupt data or a version newer than supported so callers can fall
back to the default network with a notice. A network that is well-formed but
semantically invalid — a Linear directly after a rank-3 input, or cross-entropy
with no Softmax — still loads; reporting that is `validate.ts`'s job.

## 7. Reactive store and history

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
  paramCount = $derived(/* sum of shapes.perBlock paramCount */);

  // mutators delegate to network/chain.ts and record history
}
```

`expectedClasses` is settable by the embedding page (2 for the MLP example) and
defaults to `undefined`. The store holds no rules of its own — it is a thin
reactive shell over the pure modules.

`src/lib/editor/history.ts` — undo/redo as a bounded stack of `Network`
snapshots (limit 50). Because `Network` is an immutable value, history is a
`push`/`pop` of a reference and costs almost nothing. Every structural and
parameter change records a snapshot; the stack is cleared when a different network
is loaded. Exposed as `undo()`, `redo()`, `canUndo`, `canRedo`, wired to toolbar
buttons and to `Ctrl/Cmd+Z` and `Ctrl/Cmd+Shift+Z`.

## 8. Canvas projection

`src/lib/editor/flow.ts` converts between the domain model and Svelte Flow:

- `toFlow(net, shapes): { nodes: Node[]; edges: Edge[] }` — one node per block,
  with one edge between each pair of adjacent blocks. Node and edge data carry the
  shapes from `inferShapes`.
  - **Positions are stored per block and are cosmetic.** `positionFor(net, index)`
    returns `net.positions[block.id]` when the user has moved that block, and
    otherwise the auto slot `{ x: 0, y: index * (NODE_HEIGHT + NODE_GAP) }`, with
    `NODE_WIDTH` 200, `NODE_HEIGHT` 90, and `NODE_GAP` 80. So a network nobody has
    rearranged reads as a tidy top-to-bottom pipeline, and only blocks the user has
    deliberately moved sit off it. A block's target handle is on its top edge and
    its source handle on its bottom edge, so data flows downward from the Input to
    the Output.
  - **The chain array, not the drawing, defines the order.** Dragging a node moves
    it visually and nothing else; reordering remains an explicit gesture (a wire
    drag, or the inspector's move arrows). A stray drag can therefore never change
    the network or discard training. A `Tidy up` toolbar action clears every stored
    position in one undoable step, so a scattered canvas is always recoverable.
  - Because positions can be scattered, a canvas drop resolves to the **nearest
    edge midpoint** rather than to a y coordinate: `dropIndexFor(point, centres)`
    takes the node centres and picks the wire whose midpoint is closest to the drop
    point, inserting at that edge's index + 1. For a tidy vertical column this is
    identical to the nearest-block-centre rule it replaces, since the boundary
    between adjacent midpoints falls exactly on the node centre.
  - `NODE_HEIGHT` is a layout constant, so `BlockNode` sets `min-height` from it.
    That keeps the spacing math and the rendered box from drifting apart, the same
    guarantee the node's width already has.
  - Because the chain is vertical, a canvas drop is resolved against the **y**
    coordinate: `dropIndexFor(flowY, blockCount, NODE_HEIGHT, NODE_GAP)` returns the
    interior slot whose block centre is nearest below the drop point.
- `connectionToIntent(connection, net): ChainOp | null` — converts a user-drawn
  wire into an operation on the array:
  - dragging block A's output onto block B's input where A precedes B → move A to
    just before B, i.e. destination `B's index - 1` in the current array
  - dragging A's output onto B's input where A follows B → move A to B's index
  - self-loops, output-as-source, input-as-target, and connections between already
    adjacent blocks → `null` (ignored)

Because edges are always derived, the graph cannot drift out of sync with the
network. A wire is a gesture that reorders an array, not stored state.

### Canvas behaviour

- Pan and zoom are enabled (Svelte Flow defaults), with a fit-view control.
- Svelte Flow's "Svelte Flow" attribution label is hidden via
  `proOptions={{ hideAttribution: true }}`. The library is MIT and hiding it is
  supported, though it is a courtesy request: the library asks that you subscribe to
  Svelte Flow Pro if you hide it. Hiding it also produces a one-time
  development-only console warning that cannot be suppressed, and never appears in a
  production build.
- Blocks are draggable; dropping a block onto a wire or between neighbours
  reorders the chain. Every reorder is one undoable step.
- Clicking a block selects it and opens the inspector.
- Each block node shows its kind, its most important parameter, and its shapes
  (section 9), plus a delete affordance hidden for input and output.
- Hovering a block shows a tooltip with its description, all its parameters, its
  input and output shape, and its parameter count.
- The palette adds a block by click (appends after the selection, or before the
  output) or by dragging onto the canvas. A drag-drop is resolved to an index by
  finding the gap between existing blocks whose centre is nearest the drop point;
  drops outside the interior clamp to the first or last interior slot.
- While a palette block is dragged over the canvas, the **wire it will land on**
  is highlighted: accent-coloured, thicker, and animated. Inserting at interior
  index *k* puts the block between the blocks that wire already connects, so the
  highlighted wire is the edge at `k - 1`. The highlight follows the pointer, is
  cleared when the pointer leaves the canvas, and is cleared on drop. Dragging
  over the canvas without a block (any other drag type) highlights nothing.
  Reordering by dragging a wire is a different gesture and keeps Svelte Flow's own
  connection line as its feedback.
- Palette entries and their tooltips use the descriptions from
  `descriptions.ts`, so the user learns what a block does before adding it.

## 9. Shapes: visibility and automatic input selection

This section implements principles 1, 2, and 6.

### Visibility

Shapes come from `inferShapes` and are rendered in four places:

1. **On every block node** — `in → out`, e.g. `[2] → [8]`, so the pipeline reads as
   a sequence of transformations.
2. **On every connection** — the edge label shows the tensor shape travelling
   along that wire, so it is clear what is handed from one block to the next.
3. **In the inspector**, for the selected block — a read-only "Input: [8], from
   ReLU" line, the output shape, and the block's parameter count.
4. **In a shape table** (`ShapeTable.svelte`) — a compact list of the entire
   pipeline: position, block name, input shape, output shape, parameter count, and
   a total parameter count for the network. This is the "see the shape at any
   point in the pipeline" view, and it doubles as an explanation of why a
   particular layer has the number of weights it does.

Shapes recompute reactively on every change, including mid-drag, so the numbers
never lag behind the diagram. When a shape cannot be computed because an earlier
block is invalid, the affected entries show `—` rather than a wrong value.

### Automatic input selection

- The user sets the shape of their data exactly once, on the Input block.
- No other block accepts or stores an input size. Each block's input is the
  previous block's output.
- The inspector renders the incoming shape as read-only context above the
  parameters, labelled with the block it came from.
- Parameters that depend on the incoming shape are constrained to valid values and
  pre-selected with a valid default:
  - `conv2d` `kernelSize`: choices from 1 to `min(H, W)`
  - `conv2d` `stride`: choices from 1 to `min(H, W)`
  - `conv2d` `padding`: `same` and `valid`, each shown with the output size it
    would produce (e.g. "'same' — stays 28×28", "'valid' — becomes 26×26")
  - `linear` `units`: any positive integer, default 8
- If a change upstream makes a stored parameter invalid, the value is clamped to
  the nearest valid option and the correction is **announced inline** in the
  inspector: "Kernel size changed from 5 to 3 to fit a 3×3 input." Silent
  correction is not permitted.
- The Output block's `units` defaults to the dataset's class count when the
  embedding page provides `expectedClasses`, and a mismatch produces the
  corresponding warning rather than being auto-corrected (the user may genuinely
  be changing the problem).

## 10. Model building

`src/lib/tf/buildModel.ts` — `buildModel(net): tf.Sequential`.

Refuses to build when validation reports errors (throws `NetworkInvalidError`,
which carries the `Issue[]` so callers can display the same plain-language
messages the Issues panel shows). Skips the `input` block and uses its `shape` as
`inputShape` on the first layer actually added. Skips the `output` block.

| Block | TF.js layer |
| --- | --- |
| `linear` | `tf.layers.dense({ units })` |
| `conv2d` | `tf.layers.conv2d({ filters, kernelSize, strides, padding, activation: 'linear' })` |
| `flatten` | `tf.layers.flatten()` |
| `relu` / `sigmoid` / `softmax` | `tf.layers.activation({ activation: kind })` |
| `output` | nothing |

No layer is given an explicit input size except the first, which receives the
Input block's shape. Every later layer's input size is inferred by TensorFlow.js
from the previous layer, which is the runtime counterpart of principle 2.

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

## 11. Training

`src/lib/training/Trainer.ts` — runs on the main thread, deliberately.

```ts
interface TrainStats {
  epoch: number;
  batch: number;
  batchLoss: number;
  epochMeanLoss: number | null; // null until the epoch completes
  epochAccuracy: number | null; // fraction correct on the training set; null until the epoch completes
}

class Trainer {
  constructor(
    model: tf.LayersModel,
    data: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    batchSize: number,
    onStats: (s: TrainStats) => void,
    onError?: (error: unknown) => void,
  );
  play(): Promise<void>;   // resolves when the loop stops, on every path
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
  advances when the batch counter wraps. At the end of each epoch the mean batch
  loss and the training accuracy are computed and emitted, so the UI can show both
  numbers and a learner can see that loss going down is not the same thing as
  accuracy going up.
- `step()` runs one batch without entering the play loop. `LayersModel.trainOnBatch`
  is asynchronous and returns `Promise<number | number[]>`, so `step()` awaits it and
  disposes the gathered batch tensors explicitly rather than inside `tf.tidy`, which
  cannot span an `await`.
- Batch sizes are sized to what remains in the epoch, so every example is used exactly
  once per epoch even when the dataset size is not a multiple of the batch size.
- `epoch` counts completed epochs: mid-epoch steps report the current count, and the
  step that completes an epoch reports the incremented count with `batch: 0`.
- A failing step stops the loop, clears `playing`, and reports through `onError`.
  `play()` therefore resolves rather than rejecting, because the editor calls it
  without awaiting it and a rejected promise would surface as an unhandled rejection.
  The embedding page supplies `onError` to show a plain-language banner.

**Why the main thread:** the models are tiny, `tf.nextFrame()` yields to the
browser so the UI stays responsive, and a worker would require running TF.js in
the worker plus marshalling weights and dataset tensors. The `Trainer` interface
is narrow enough to move into a worker later without changing callers.

## 12. Data (2D points)

`src/lib/data/rng.ts` — `mulberry32(seed)`, a seeded PRNG so generators and tests
are reproducible.

`src/lib/data/points.ts`:

```ts
interface Point { x: number; y: number; label: 0 | 1 }
interface PointDataset { points: Point[]; numClasses: 2 }
```

- Domain is `x, y ∈ [-1, 1]`, matching the rendering surface.
- Generators, each taking a point count and a seed: `twoGaussians`, `spirals`,
  `xor`, `circles`. Each generator is labelled with a plain-language description
  in the dataset picker (e.g. spirals: "Two interlocking spirals. Needs a hidden
  layer to separate.").
- `addPoint(dataset, x, y, label)` and `clearPoints(dataset)` are pure.
- `toTensors(dataset): { xs: tf.Tensor2D; ys: tf.Tensor2D }` — `xs` is `[N, 2]`
  float32, `ys` is `[N, 2]` one-hot float32. Tensors are rebuilt when the dataset
  changes and disposed when replaced.

## 13. Decision boundary rendering

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
  invalid, the canvas renders points on a neutral background and no boundary, with
  a short caption explaining that the network needs fixing first.

## 14. Loss chart and stats

`src/lib/components/LossChart.svelte` — an SVG polyline of per-epoch mean loss, a
rolling window of the most recent 200 epochs, with min/max axis labels and the
latest value shown. No dependency; roughly 40 lines.

Beside it, a compact stats readout shows epoch, current loss, and training
accuracy as percentages or decimals, with a one-line explanation of each. Accuracy
is shown for classification only.

## 15. Persistence

**Architecture** — `localStorage`, key `visnet:network:v1`, storing
`{ version: 1, network }` via `serialize.toJSON`. Autosaved, debounced ~500 ms
after a change. On load, `fromJSON` is used; corrupt or unsupported data falls
back to `createEmptyNetwork()` with a notice explaining that the saved network
could not be read.

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

## 16. Error handling and messages

- Validation errors disable training; the Issues panel lists each error by its
  `title`, with the `message` underneath and the `fix` highlighted as an
  instruction. Clicking an issue selects and centres the offending block.
- Warnings are shown the same way but do not block training.
- No raw exception text or type name is ever shown to the user. `NetworkInvalidError`
  carries `Issue[]`; other failures are mapped to a plain-language banner plus a
  collapsible technical detail for anyone who wants it.
- `buildModel` failures are caught: the error is surfaced in a banner, the model
  is disposed, training stays disabled.
- Backend selection tries WebGL and falls back to the CPU backend, showing a
  one-line notice that training will be slower and why.
- Resource hygiene: rebuilds dispose the previous model and its tensors; training
  wraps tensor creation in `tf.tidy`; route teardown disposes the model, dataset
  tensors, and render resources to avoid leaking WebGL contexts on navigation.
- Storage failures (quota exceeded, IndexedDB unavailable) are non-fatal: state
  stays in memory and the user sees a notice saying what was not saved.
- Illegal canvas connections are rejected at `connectionToIntent`, produce no
  change, and show a brief inline hint explaining why the connection was not
  allowed.
- Destructive actions are undoable rather than gated behind confirmation dialogs,
  except where weights would be lost, which warns explicitly (section 10).

## 17. Routes and the embedding contract

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

## 18. Testing strategy

Vitest, run with `npm test`. Tests live beside their modules as `*.test.ts`.

Covered:
- `chain.ts` — insert/move/remove/replace, including refusal to move or delete
  input and output and index clamping.
- `inferShapes.ts` — each block kind, conv output sizes for `same` and `valid`,
  flatten, `paramCount` for a known MLP, edge shapes, and `null` propagation when
  an earlier block is invalid.
- `validate.ts` — every error and warning rule in the catalog, including the
  `expectedClasses` warning, plus an assertion that every issue produced by every
  rule has a non-empty `title`, `message`, and `fix` (guarding the beginner-friendly
  contract).
- `descriptions.ts` — every `BlockKind` and every tunable parameter has a
  non-empty description.
- `serialize.ts` — round-trip equality, corrupt input, unsupported version,
  migration seam.
- `flow.ts` — `toFlow` node/edge derivation, shape attachment, and
  `connectionToIntent` for legal reorders and every illegal case.
- `points.ts` / `rng.ts` — seeded generators are deterministic; one-hot encoding
  is correct; `addPoint`/`clearPoints` are pure.
- `history.ts` — undo/redo across structural and parameter edits, stack
  limit, and that a new load clears history.
- `buildModel.ts` — on the CPU backend in Node: a valid MLP builds with the
  expected number of layers, input shape `[null, 2]`, output shape `[null, 2]`;
  an invalid network throws `NetworkInvalidError` carrying issues; changing only
  training config preserves weights while changing architecture does not.

Not covered by automated tests in this phase: the Svelte components, the canvas
interaction, and the training animation. These are verified manually.

## 19. Manual verification checklist

1. `npm run dev`, open `/examples/mlp`; the default network renders as a
   top-to-bottom pipeline with `in → out` shape badges on every block and shape
   labels on every wire.
2. The shape table lists the whole pipeline with correct shapes and parameter
   counts, and the network total matches.
3. Add, delete, and drag blocks; shapes, edges, and the shape table update
   immediately, and every change is undoable with `Ctrl/Cmd+Z` and redoable with
   `Ctrl/Cmd+Shift+Z`.
4. Try to delete the input and output blocks — both are refused.
5. Drag a wire into an illegal configuration — nothing changes and a hint
   explains why.
6. Hover a block — the tooltip explains what it does, its parameters, its shapes,
   and its parameter count.
7. Add a Convolution layer to the default 2D network — an error appears whose
   message and fix are both in plain language, naming the block.
8. Change the Input shape so a stored kernel size becomes invalid — the value is
   clamped and the inspector says what changed and why.
9. Pick each dataset generator; points appear and the boundary renders.
10. Click the canvas to add points; change the selected class and add more.
11. Press play — the loss curve descends, accuracy is shown, and the boundary
    visibly adapts.
12. Press pause, then step — exactly one batch of progress.
13. Break the network (e.g. delete the softmax with cross-entropy selected) — a
    warning appears and training still works.
14. Reload the page — network, dataset, and (after Save model) weights are
    restored and the boundary redraws.
15. Resize the window and navigate away and back — no WebGL context warnings in
    the console.

## 20. Implementation order

1. Scaffold SvelteKit + TypeScript + tooling, design tokens, ESLint/Prettier,
   Vitest wiring. Verify `dev`, `build`, `check`, `test` all run.
2. `src/lib/network/` pure modules, `descriptions.ts`, and the message catalog,
   with their tests.
3. `src/lib/tf/buildModel.ts` and `src/lib/training/Trainer.ts` with tests.
4. `src/lib/data/` and `src/lib/render/boundary.ts`.
5. Editor UI: `BlockNode` (with shapes and tooltip), `BlockPalette`,
   `BlockCanvas`, `InspectorPanel`, `ShapeTable`, `IssuesPanel`,
   `EditorToolbar`, `NetworkEditor`.
6. `editor/history.ts` and keyboard shortcuts.
7. `/examples/mlp`: dataset controls, `TrainingPanel`, `LossChart`, stats
   readout, `DecisionBoundary`, and the wiring that ties them to the editor.
8. `src/lib/persist/` and the save/load UI.
9. Landing page, `/examples/cnn` placeholder.
10. Update `AGENTS.md` and `README.md` to match this design; run `lint`, `check`,
    and `test` to green.

## 21. Deferred work

- CNN / MNIST example, including enabling the `flatten` block in the palette.
- Training in a Web Worker behind the existing `Trainer` interface.
- Multiple named saved networks.
- SGD momentum, learning-rate schedules, and other optimizer options.
- Model export/import as files, and share-via-URL.
- Component and end-to-end tests.
- A guided tour or inline lesson mode.

## 22. Decisions and deviations

Recorded so later readers know these were deliberate.

1. **SvelteKit replaces the literal "Svelte + Vite" in `AGENTS.md`.** Needed for
   example subpages and a static, prerendered build. `AGENTS.md` will be updated.
2. **Softmax is added to the README's module list.** Cross-entropy needs it, and a
   visible block is more honest than hidden machinery.
3. **Linear chain only.** No DAGs, branching, or multi-input blocks.
4. **The `output` block is a marker that adds no layer.** Keeps softmax on the last
   real layer and makes the terminal node declarative.
5. **Auto-layout by default, with stored positions — superseding the original
   "positions are never stored" decision.** The original choice avoided the "my
   blocks are scattered" failure mode, but it also meant the canvas could not be
   arranged at all. Positions are now stored per block, so unmoved blocks still
   auto-place in a tidy column while moved ones stay where the user put them. The
   safeguard the original decision was protecting is kept two other ways: the chain
   array remains the sole source of order, so dragging a node is cosmetic and can
   never change the network, and a `Tidy up` action restores the column in one
   undoable step.
6. **Weights in IndexedDB, architecture in localStorage.** Quota and Float32
   considerations; the UI presents one save/load action.
7. **Training on the main thread.** Simplicity now, with a documented path to a
   worker.
8. **`flatten` exists in the type system and model builder but is not in the MVP
   palette.** Avoids dead UI while removing rework when the CNN example lands.
9. **No block stores its input size.** Inputs are always derived from the previous
   block's output, matching both the teaching goal and TensorFlow.js's own
   behaviour.
10. **Shapes are shown in four places**, including per-edge labels and a whole
    pipeline table, because shape reasoning is the main skill the app teaches.
11. **Every validation issue must carry a plain-language fix.** Enforced by the
    `Issue` type and by a test that walks every rule.
12. **Undo/redo is in scope.** Added for beginner-friendliness; it is cheap
    because `Network` is immutable and history is a stack of snapshots.
13. **Accuracy is reported alongside loss.** Seeing the two diverge is a teaching
    moment worth the small cost of an evaluation pass per epoch.
