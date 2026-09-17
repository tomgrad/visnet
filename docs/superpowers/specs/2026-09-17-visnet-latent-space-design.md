# VisNet — Latent Space View (2D classification)

Date: 2026-09-17
Status: approved (pending written-spec review)

## 1. Overview

The 2D classification example currently shows one view of a network: the input
plane, coloured by the network's prediction, with the dataset points on top.
This phase adds a second, complementary view: the **latent space of the layer
currently selected on the canvas**.

The selected block's output space is projected to two dimensions — a pair of
adjacent output dimensions chosen with a button — and rendered as:

- a **warped mesh**: faint lines connecting the projected positions of adjacent
  cells of the input grid, showing how the layer bends the input plane;
- **class-coloured cells**: each grid cell drawn at its latent position,
  coloured by the network's final predicted class for that input;
- the **dataset points** at their latent coordinates, coloured by their true
  class.

The view updates live while training, so a learner watches the classes separate
as the loss falls.

## 2. Goals

- Show the intermediate representation of the data at any block in the network.
- Make the transformation legible: the grid warps, the classes separate.
- Stay live during training without blocking the training loop.
- Be self-contained and reusable by other examples later (the CNN example is
  out of scope for this phase, but nothing here should preclude it).

## 3. Non-goals (this phase)

- Visualizing layers whose output is not rank 1 (convolution feature maps).
  The 2D example's palette contains only dense and activation blocks.
- A per-neuron heatmap grid (the convnetjs "layer maps" view). This phase shows
  one selected layer in 2D, not every neuron.
- 3D or t-SNE/PCA projections. The projection is a literal pair of dimensions.
- Changes to the CNN example.

## 4. Where it lives

A new panel on `/examples/mlp`, rendered in the experiment column below the
decision boundary. The existing `NetworkEditor`, `Network` model, `buildModel`,
`Trainer`, and `DecisionBoundary` are unchanged except as noted in section 8.

## 5. Architecture

```
LatentSpace.svelte
  ├─ static import: network/probe.ts        (pure: block -> probe target)
  ├─ dynamic import: render/latent.ts       (TF.js, browser only)
  └─ props: model, store, dataset, redrawKey

render/latent.ts
  ├─ createProbe(model)          -> multi-output LayersModel sharing weights
  ├─ disposeProbe(probe)
  └─ projectLatent(probe, ...)   -> projected grid + points + classes + bounds
```

The probe is a `tf.model({ inputs: model.inputs, outputs: model.layers.map(l => l.output) })`.
Because it shares the training model's layer objects and weights, training is
reflected without rebuilding the probe; the probe is rebuilt only when the
architecture changes (a new `model` object) and disposed when it does.

### SSR boundary

`render/latent.ts` imports `@tensorflow/tfjs` at runtime. It is imported
dynamically inside `onMount`, exactly like `render/boundary.ts`, so all routes
still prerender. `network/probe.ts` is pure and may be imported statically.

`AGENTS.md` must add `src/lib/render/latent.ts` to the list of modules allowed
to import TF.js at runtime.

## 6. Probe target mapping (pure)

`src/lib/network/probe.ts`:

```ts
export interface ProbeTarget {
  source: 'input' | number; // 'input', or an index into model.layers
  dims: number[]; // the target's output dimensions, [] when unknown
}

export function probeTargetFor(net: Network, blockId: string | null): ProbeTarget | null;
```

Rules, matching the approved behaviour "every block's output":

- No selection, unknown id, or a network with no real layers → `null`.
- Input block → `{ source: 'input', dims: input.shape }`.
- A real block (anything that is not `input` or `output`) → its model layer
  index and its `inferShapes` output shape. Real blocks map 1:1 to model
  layers in `buildModel`, in array order.
- Output marker → the last real block's index and shape.
- When `inferShapes` reports a `null` output shape (an earlier block is
  invalid), `dims` is `[]`.

`source` indexes `model.layers`; `dims` is the rank-1 output shape. A target
with `dims.length < 2` cannot be plotted.

## 7. Render module

`src/lib/render/latent.ts`:

```ts
export const LATENT_GRID_SIZE = 32;

export interface Probe {
  model: tf.LayersModel;
  layerCount: number;
}

export interface LatentBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface LatentSample {
  grid: Float32Array; // 2 * LATENT_GRID_SIZE^2 projected coordinates
  gridClasses: Int32Array; // LATENT_GRID_SIZE^2 final predicted class per cell
  points: Float32Array; // 2 * N projected coordinates
  bounds: LatentBounds;
}

export function createProbe(model: tf.LayersModel): Probe;
export function disposeProbe(probe: Probe | null): void;
export function gridInputs(size?: number): Float32Array;
export function projectLatent(
  probe: Probe,
  cells: Float32Array,
  points: Float32Array,
  source: 'input' | number,
  dimA: number
): LatentSample;
```

- `createProbe` builds the multi-output model. If `layer.output` is not yet
  defined, it first runs one warm-up `predict` on a zero input of the model's
  input shape so the layers are built.
- `projectLatent` runs one batched `predict` of the grid and points through the
  probe inside `tf.tidy`. The final predicted class per grid cell is the
  `argMax` of the probe's last output. Projected coordinates are the target's
  outputs `[dimA]` and `[dimA + 1]`, except for `source === 'input'`, where the
  coordinates are the input coordinates themselves (the class colours still
  come from the probe).
- `bounds` is the combined extent of the projected grid and points.
- All intermediate tensors are disposed; the returned arrays are plain typed
  arrays.

## 8. Component

`src/lib/components/LatentSpace.svelte`:

```ts
interface Props {
  model: tf.LayersModel | null;
  store: NetworkStore;
  dataset: PointDataset;
  redrawKey: number;
}
```

- On mount, dynamically import `render/latent`.
- Rebuild the probe whenever `model` changes; dispose the previous probe and on
  unmount. This is component-local, so `experiment.svelte.ts` is untouched.
- `pair` is component state: the index of the first dimension, starting at 0.
  It resets to 0 when the selected block changes, and is clamped to
  `dims.length - 2` when the dimensions shrink.
- The "Next dimensions" button advances `pair` and wraps to 0.
- The label reads `dimensions {pair+1} & {pair+2} of {dims.length}`.
- A redraw effect depends on `redrawKey`, `model`, the selected target, and
  `pair`.

### Drawing

Into a square canvas, each frame:

1. Compute the projected sample via `projectLatent`.
2. Auto-fit the axes to `bounds` with a small padding. If the extent is zero on
   an axis, pad to a minimum span.
3. Draw the warped mesh: for each grid cell, lines to its right and lower
   neighbours, using the projected grid coordinates. Faint stroke.
4. Draw the coloured cells: a small quad per grid cell, filled with the class
   colour for that cell's final predicted class. Drawn beneath the mesh so the
   mesh reads on top.
5. Draw the dataset points: filled circle, white stroke, coloured by true class.

The canvas reuses the class colours and background from
`src/lib/render/palette.ts`.

## 9. Error handling

- `model` is null → "Fix the problems listed in the editor before the latent
  space can be drawn."
- Nothing selected → "Select a block to see its latent space."
- `probeTargetFor` returns null, or `dims.length < 2` → "This layer has fewer
  than two dimensions, so there is nothing to plot."
- Probe creation throws → a plain-language message; the panel does not crash.

## 10. Performance

The grid is 32×32 (1024 cells). Each frame predicts 1024 grid inputs plus the
dataset points through the probe — small for the tiny networks in this example.
`redrawKey` already advances once per training batch, so the view is live at
the same cadence as the decision boundary. No additional animation frame is
introduced.

## 11. Testing

- `network/probe.test.ts` (engine project): input, each real block kind, the
  output marker, no selection, unknown id, and a `null` output shape.
- `render/latent.test.ts` (engine project, CPU backend): `createProbe` shares
  weights with the training model (changing a weight changes the probe output),
  `projectLatent` returns the expected projection for a known layer, the class
  indices match the model's argmax, bounds cover the sample, and
  `disposeProbe` returns `tf.memory().numTensors` to its baseline.
- `components/LatentSpace.test.ts` (ui project, render module mocked as in
  `DecisionBoundary.test.ts`): the cycler advances and wraps, the label is
  correct, the fallback messages appear, and the canvas redraws when
  `redrawKey` changes.

## 12. Files

- Create: `src/lib/network/probe.ts`, `src/lib/network/probe.test.ts`
- Create: `src/lib/render/latent.ts`, `src/lib/render/latent.test.ts`
- Create: `src/lib/components/LatentSpace.svelte`,
  `src/lib/components/LatentSpace.test.ts`
- Create: `src/lib/components/__stubs__/LatentSpaceHarness.svelte` (test only)
- Modify: `src/routes/examples/mlp/+page.svelte` (render the panel)
- Modify: `AGENTS.md` (add `render/latent.ts` to the TF.js allow-list)
- Modify: `README.md` (describe the new view)

## 13. Manual verification

1. Open `/examples/mlp`; select the first hidden Linear block; the panel shows a
   warped grid and separated points.
2. Press "Next dimensions"; the view changes to the next pair and the label
   updates.
3. Press play; the grid and points move and separate as the loss falls.
4. Select the Input block; the view shows the input plane (unwarped grid) with
   the points.
5. Select the Output marker; it shows the last real layer.
6. Make a Linear layer 1 unit; selecting it shows the "fewer than two
   dimensions" message.
7. Delete every hidden layer or otherwise invalidate the network; the panel
   shows the fix message rather than crashing.
8. Navigate away and back; no WebGL context or tensor warnings in the console.
