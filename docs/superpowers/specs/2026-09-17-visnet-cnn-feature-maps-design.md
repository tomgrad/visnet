# VisNet — CNN Feature Maps

Date: 2026-09-17
Status: approved (pending written-spec review)

## 1. Overview

The CNN example currently shows a grid of test digits with the network's
prediction for each. This phase adds a **feature-map panel**: for the block
selected on the canvas, it shows that layer's output for a single test digit.

- A rank-3 output `[H, W, C]` is drawn as one grayscale map per channel, tiled
  in rows.
- A rank-1 output `[N]` is drawn as `N` grayscale squares, tiled in rows.
- A button cycles the test digit; the panel updates live during training.

This mirrors the latent-space panel added for the 2D example, but the unit of
display is a feature map rather than a 2D projection.

## 2. Goals

- Show what each convolutional filter responds to, for a real input image.
- Show dense activations as a readable grayscale vector.
- Stay live during training without blocking the training loop.
- Reuse the existing forward-pass mechanism rather than duplicating it.

## 3. Non-goals (this phase)

- Choosing the input image by clicking the sample grid; a cycle button is the
  only control.
- Colour maps, per-filter labels, or filter-weight visualisation. Maps are
  grayscale, white = high activation, black = low.
- Deconvolution or receptive-field overlays.
- Changing the MLP example or its latent-space panel.

## 4. Where it lives

A new panel on `/examples/cnn`, in the experiment column below the sample grid.
`NetworkEditor`, `buildModel`, `Trainer`, and `SampleGrid` are unchanged.

## 5. Architecture

```
FeatureMaps.svelte
  ├─ static import: network/probe.ts       (pure: block -> activation source)
  ├─ static import: data/images.ts         (pure: read one image's pixels)
  ├─ dynamic import: render/features.ts    (TF.js, browser only)
  └─ props: model, store, dataset, indices, redrawKey

render/features.ts
  └─ featureMaps(model, pixels, rows, cols, source) -> FeatureMap[]

render/activations.ts
  └─ forwardActivations(model, input) -> tf.Tensor[]   (shared with render/latent.ts)
```

### SSR boundary

`render/activations.ts` and `render/features.ts` import `@tensorflow/tfjs` at
runtime and are imported dynamically inside `onMount`, exactly like
`render/latent.ts`. `network/probe.ts` and `data/images.ts` are pure and may be
imported statically. `AGENTS.md` must add both new render modules to the TF.js
runtime-import allow-list.

## 6. Shared forward pass

`src/lib/render/activations.ts`:

```ts
export function forwardActivations(model: tf.LayersModel, input: tf.Tensor): tf.Tensor[];
```

It applies each layer of the model to the running tensor and returns one tensor
per model layer, in order. It does **not** open a `tf.tidy`; the caller owns the
tidy scope so it can use the returned tensors and then reclaim them. This is the
same traversal already in `render/latent.ts`; `latent.ts` is refactored to call
it, with its existing tests as the safety net.

## 7. Feature render module

`src/lib/render/features.ts`:

```ts
export interface FeatureMap {
  width: number;
  height: number;
  values: Uint8ClampedArray; // width * height, grayscale 0..255
}

export function featureMaps(
  model: tf.LayersModel,
  pixels: Uint8Array,
  rows: number,
  cols: number,
  source: 'input' | number
): FeatureMap[];
```

- Builds the input tensor internally (`pixels / 255`, shape `[1, rows, cols, 1]`)
  so the component never touches TF.js.
- Runs `forwardActivations` inside one `tf.tidy` and selects the activation:
  `source === 'input'` uses the input tensor, otherwise `activations[source]`.
- Rank-3 output `[H, W, C]` → `C` maps of `H×W`, each normalised **per map**
  between its own min and max (min → 0, max → 255). TF.js convolution output is
  channels-last, so pixel `p` of channel `ch` is `data[p * C + ch]` — not a
  contiguous `H*W` slab.
- Rank-1 output `[N]` → `N` maps of `1×1`, all normalised against the vector's
  **global** min and max, so the squares form a single white-to-black ramp.
  (A per-element min/max would be degenerate; the vector is one map split into
  cells.)
- A map whose values are all equal becomes mid-grey (128).
- Any other rank returns an empty array; the component shows a message.
- The returned `FeatureMap`s hold no tensors; `tf.tidy` reclaims every
  intermediate.

## 8. Component

`src/lib/components/FeatureMaps.svelte`:

```ts
interface Props {
  model: tf.LayersModel | null;
  store: NetworkStore;
  dataset: ImageDataset | null;
  indices: number[];
  redrawKey: number;
}
```

- On mount, dynamically import `render/features`.
- `target = probeTargetFor(store.network, store.selectedBlockId)`.
- `digit` is component state: an index into `indices`, starting at 0. The
  "Next digit" button advances it and wraps. It is clamped when `indices`
  shrinks.
- The redraw effect depends on `redrawKey`, `model`, `target`, `digit`, and
  `indices`.

### Drawing

One canvas, sized to its content in the draw effect:

1. A thumbnail of the selected test image, about 40 px, at the top.
2. The maps below it:
   - rank-3 maps are drawn at a fixed size (about 40 px) preserving aspect,
     up to 4 per row;
   - rank-1 elements are drawn as small squares (about 4 px) with a 1 px gap,
     wrapping to the panel width.
3. A caption: `digit {label}, sample {digit+1} of {indices.length}`.

Grayscale values come straight from `FeatureMap.values`. The panel is not
auto-scaled; map and square sizes are fixed.

### Live behaviour and performance

The panel reuses `redrawKey`, which the controller already bumps once per
training batch. One forward pass over a 28×28 image, plus drawing a few hundred
small squares, is cheap.

## 9. Error handling

- `model` is null → "Fix the problems listed in the editor before the feature
  maps can be drawn."
- Nothing selected → "Select a block to see its feature maps."
- Target output rank is neither 1 nor 3 → "This layer's output cannot be shown
  as feature maps."
- Dataset not loaded → "The digit images are not loaded yet."
- `featureMaps` throws → a plain-language message; the panel does not crash.

## 10. Testing

- `render/activations.test.ts` (engine, CPU): `forwardActivations` returns one
  tensor per layer whose last entry matches `model.predict`; the model remains
  usable.
- `render/features.test.ts` (engine, CPU): a conv layer yields `C` maps of
  `H×W`; a dense layer yields `N` `1×1` maps; per-map normalisation maps the
  min to 0 and the max to 255; a flat map is mid-grey; the vector case is
  normalised globally; no tensor leak (`numTensors` returns to baseline).
- `components/FeatureMaps.test.ts` (ui, `render/features` mocked and a fake 2D
  context as in `LatentSpace.test.ts`): the cycler advances and wraps; the
  fallback messages appear; both a rank-3 and a rank-1 selection draw.

## 11. Files

- Create: `src/lib/render/activations.ts`, `src/lib/render/activations.test.ts`
- Create: `src/lib/render/features.ts`, `src/lib/render/features.test.ts`
- Create: `src/lib/components/FeatureMaps.svelte`,
  `src/lib/components/FeatureMaps.test.ts`
- Modify: `src/lib/render/latent.ts` (use `forwardActivations`)
- Modify: `src/routes/examples/cnn/+page.svelte` (render the panel)
- Modify: `AGENTS.md` (allow-list), `README.md`

## 12. Manual verification

1. Open `/examples/cnn` (with MNIST prepared); select the Convolution block;
   8 feature maps appear for the first test digit.
2. Press "Next digit"; the thumbnail and maps change.
3. Select the Flatten block; a dense grid of small squares appears.
4. Select the Linear or Softmax block; 10 squares appear.
5. Press play; the maps update as training progresses.
6. Select the Input block; the thumbnail-sized input map appears.
7. Break the network; the panel shows a fix message rather than crashing.
8. Navigate away and back; no console or WebGL warnings.
