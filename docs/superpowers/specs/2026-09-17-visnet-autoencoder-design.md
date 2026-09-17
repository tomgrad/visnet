# VisNet — Autoencoder Example

Date: 2026-09-17
Status: approved (pending written-spec review)

## 1. Overview

A third example: an **autoencoder on MNIST**. The network is asked to predict
its own input through a tight bottleneck, so it learns a compressed code. The
example offers two presets (dense and convolutional), shows the reconstruction
against the original, and visualizes the block selected on the canvas.

This is Spec B; it builds on Spec A's reshape block and shape-based Output.

## 2. Goals

- Teach unsupervised/regression learning: the target is the input itself.
- Show the reconstruction and the compressed code live during training.
- Offer a dense preset (a 2-neuron bottleneck) and a convolutional preset.
- Follow the established pattern: the visualization follows the selected block.

## 3. Non-goals

- Denoising or any input corruption. The task is plain identity reconstruction.
- New loss functions or optimizers beyond the existing MSE.
- Persisting the training dataset (MNIST is re-read, as in the CNN example).
- Changing the MLP or CNN examples.

## 4. Task model

The framework is currently classification-shaped. This phase introduces a task
distinction, contained to the example and validation:

- `NetworkStore` gains `task: 'classification' | 'reconstruction'` (default
  `'classification'`), set by the page like `expectedClasses`.
- `findProblems(net, options)` accepts `task`. When the task is `reconstruction`,
  the **"Image input without a Convolution layer"** warning is suppressed (a
  dense autoencoder on images is legitimate). All other rules are unchanged.
- `Trainer` gains a `computeAccuracy` option (default `true`). When false,
  `epochAccuracy` is always `null` and the argmax accuracy is never computed.
- `TrainingPanel` gains `showAccuracy` (default `true`). When false, the accuracy
  cell is not rendered. The autoencoder passes `false`; the loss (MSE) is the
  reconstruction error.

## 5. Route and example module

New `/examples/autoencoder` route and `src/lib/examples/autoencoder/example.ts`:

- `AUTOENCODER_PALETTE`: `conv2d`, `maxpool2d`, `upsampling2d`, `flatten`,
  `reshape`, `linear`, `relu`, `sigmoid`, `tanh`, `softmax`.
- `AUTOENCODER_STORAGE_KEYS`, `AUTOENCODER_WEIGHTS_ID`.
- `PRESETS`: an array of `{ id, label, create(): Network }` with `dense` and
  `conv`.
- `TRAIN_COUNT = 5000`, `SCATTER_COUNT = 500`, `RECONSTRUCTION_COUNT = 40`.

### Presets

Dense:

```
input [28,28,1] → flatten → linear 32 → relu → linear 2 → tanh
  → linear 32 → relu → linear 784 → sigmoid → reshape [28,28,1] → output [28,28,1]
```

Convolutional:

```
input [28,28,1] → conv2d(8,3,same) → relu → maxpool2d(2,valid)
  → conv2d(8,3,same) → relu → maxpool2d(2,valid)
  → upsampling2d(2) → conv2d(8,3,same) → relu
  → upsampling2d(2) → conv2d(1,3,same) → sigmoid → output [28,28,1]
```

Both end with a rank-3 output of `[28,28,1]`, matching the input, and use MSE.

## 6. Data

`src/lib/data/tensors.ts` gains:

```ts
export function imagesToReconstruction(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor4D };
```

`xs` is `[N, rows, cols, 1]` normalised to `[0,1]`; `ys` is a matching tensor of
the same values (the target is the input). The two are separate tensors so each
can be disposed independently.

- Training uses the first `TRAIN_COUNT` images of the MNIST train split.
- The code scatter uses `SCATTER_COUNT` test images.
- The reconstruction view uses `RECONSTRUCTION_COUNT` test images.

## 7. Rendering

Two new TF.js render modules (added to the `AGENTS.md` allow-list):

- `src/lib/render/reconstruction.ts` — `reconstruct(model, pixels, rows, cols, count)`
  builds the input tensor, runs the forward pass, and returns `count` grayscale
  `rows×cols` images (0–255) from the model's output.
- `src/lib/render/codes.ts` — `codeScatter(model, pixels, rows, cols, count, source, dimA)`
  builds the input tensor for the sample images, runs the forward pass, and
  projects the selected activation onto dimensions `dimA` and `dimA+1`, returning
  the projected points and their bounds.

Both reuse `render/activations.ts`'s `forwardActivations` and wrap their work in
`tf.tidy`.

## 8. Components

- `components/ReconstructionGrid.svelte` — a canvas showing two aligned rows of
  `RECONSTRUCTION_COUNT` test digits: originals on top, reconstructions below.
  Props `{ model, dataset, indices, redrawKey }`; dynamically imports
  `render/reconstruction`.
- `components/CodeScatter.svelte` — a canvas scatter of the sample images at the
  selected layer's activation, coloured by digit label, with a "Next dimensions"
  button cycling consecutive dimension pairs and fallback messages. Props
  `{ model, store, dataset, indices, redrawKey }`; dynamically imports
  `render/codes`; reuses `probeTargetFor`.

The page renders the **code scatter** when the selected block's output is rank 1
and the existing **`FeatureMaps`** panel when it is rank 3, computing the rank
from `probeTargetFor`.

## 9. Page

`routes/examples/autoencoder/+page.svelte`:

- Loads MNIST (degrading to an instruction when absent, as the CNN example does).
- A preset picker; switching replaces the network with the preset and resets
  training/model (the existing weights-discarded notice applies). The chosen
  preset id is persisted.
- Owns the `NetworkStore` with `task = 'reconstruction'`.
- Uses the shared `createExperiment` controller with `task: 'reconstruction'`.
- Renders `ReconstructionGrid`, the selected-layer view (scatter or feature
  maps), `TrainingPanel` with `showAccuracy={false}`, and `LossChart`.

## 10. Error handling

- MNIST absent → the existing instruction panel.
- No model / invalid network → the selected-layer view and reconstruction grid
  show a fix-the-network message.
- Nothing selected → "select a block".
- A selected rank that is neither 1 nor 3 → a plain-language message.
- Render failures → a plain-language message, never a crash.

## 11. Testing

- `examples/autoencoder/example.test.ts`: both presets produce networks with no
  errors under `findProblems(net, { task: 'reconstruction' })`; the palette.
- `data/tensors.test.ts`: `imagesToReconstruction` shapes and normalisation.
- `render/reconstruction.test.ts`, `render/codes.test.ts` (engine, CPU): output
  shapes, projection values, and no tensor leak.
- `training/Trainer.test.ts`: `computeAccuracy: false` yields `epochAccuracy: null`.
- `network/problems.test.ts`: the task option suppresses the image-without-conv
  warning and leaves classification behaviour unchanged.
- `components/ReconstructionGrid.test.ts`, `components/CodeScatter.test.ts`
  (ui, render modules mocked, fake 2D context): draw paths, cycler, fallbacks.
- `components/TrainingPanel.test.ts`: `showAccuracy={false}` hides the accuracy
  cell.

## 12. Files

- Create: `src/lib/examples/autoencoder/example.ts` (+ test)
- Create: `src/routes/examples/autoencoder/+page.svelte`
- Create: `src/lib/components/ReconstructionGrid.svelte` (+ test)
- Create: `src/lib/components/CodeScatter.svelte` (+ test)
- Create: `src/lib/render/reconstruction.ts` (+ test), `src/lib/render/codes.ts` (+ test)
- Modify: `src/lib/data/tensors.ts`, `src/lib/training/Trainer.ts`,
  `src/lib/components/TrainingPanel.svelte`, `src/lib/network/problems.ts`,
  `src/lib/editor/networkStore.svelte.ts`, `src/lib/examples/experiment.svelte.ts`,
  `src/lib/examples/runtime.ts`, `AGENTS.md`, `README.md`, `src/routes/+page.svelte`
  (link the new example)

## 13. Manual verification

1. Open `/examples/autoencoder` with MNIST prepared; the dense preset loads.
2. Press play; the reconstruction grid sharpens and the loss falls.
3. Select the `linear 2` bottleneck; the code scatter shows digits clustering by
   class as training proceeds.
4. Switch to the convolutional preset; the network is replaced and training
   resets.
5. Select a convolution block; feature maps appear instead of the scatter.
6. Reload; the network and preset are restored.
7. Confirm the training panel shows no accuracy cell, and navigating away leaves
   no console or WebGL warnings.
