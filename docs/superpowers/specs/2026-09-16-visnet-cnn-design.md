# VisNet CNN Example — Design

Date: 2026-09-16
Status: approved (pending written-spec review)
Supersedes the CNN deferral in `2026-09-16-visnet-design.md` (§1 non-goals, §21).

## 1. Overview

The MLP example classifies 2D points and shows a live decision boundary. This
phase adds the second example the README promises: a small convolutional network
trained on handwritten digits. There is no 2D boundary to watch for a 28×28 image,
so the teaching payload comes from a **grid of test digits re-labelled live as
training proceeds** — you watch the red borders disappear.

### Goals

- A prep script that downloads MNIST and writes a small subset, so that **no data
  is committed to the repository**.
- A CNN example page that reuses the existing editor, training panel, loss chart,
  and stats readout unchanged.
- A live sample grid of digits with the network's prediction and whether it is right.
- Fix the three recorded findings that the CNN example would otherwise expose.

### Non-goals

- Committing any dataset, or fetching anything during `dev` or `build`.
- Persisting the image dataset (it is deterministic, so it is re-read instead).
- A confusion matrix, per-class accuracy bars, or a drawing pad. The sample grid is
  the whole visual payload.
- Multi-input or branching networks; the topology stays a linear chain.
- Changing the training algorithm, the optimizer set, or the loss set.

## 2. Scope — two plans

The work splits into two plans because the first is testable on its own and the
second depends on it.

- **C1 — data pipeline and prerequisite fixes.** The prep script, the asset format
  and its parser, the image dataset and tensor conversion, and the three recorded
  fixes.
- **C2 — the CNN example page.** The sample grid, the page, the palette, storage
  and weight namespacing at the page level, and the documentation.

## 3. The prep script

`scripts/prepare-mnist.mjs`, run as `npm run data:mnist`. Plain Node ESM, no
dependencies: it uses global `fetch`, `node:zlib`, and `node:fs`.

1. Downloads the four canonical files from
   `https://storage.googleapis.com/cvdf-datasets/mnist/`:
   `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
   `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
2. Gunzips and parses the IDX format (big-endian: a 4-byte magic, then dimensions,
   then data).
3. Takes the first `train` images and the first `test` images.
4. Writes `static/mnist/train.bin` and `static/mnist/test.bin` in the format below.
5. Prints the source, the counts, the dimensions, the bytes written, and the output
   paths, and notes that MNIST derives from NIST.

Flags: `--train=N` (default 1000), `--test=M` (default 200), `--force`.

Behaviour:

- **Idempotent.** If both output files exist and `--force` was not passed, it prints
  that the data is already prepared and exits without touching the network.
- Creates `static/mnist/` if needed.
- Fails with a clear message and a non-zero exit code when a download fails, when
  the IDX magic is wrong, or when a requested count exceeds what the file holds.

The script exports `parseIdxImages`, `parseIdxLabels`, and `encodeSplit`, and runs
its main routine only when executed directly, so its logic is unit-testable.

### Attribution

MNIST is a derivative of the NIST Special Database 19. The script prints the
source, and `README.md` notes where the data comes from and that it is downloaded
locally rather than redistributed.

## 4. Asset format

One file per split, little-endian, so a single fetch serves a split and the parser
is trivial.

| Offset      | Bytes                 | Field                                 |
| ----------- | --------------------- | ------------------------------------- |
| 0           | 4                     | magic, ASCII `VSNT`                   |
| 4           | 1                     | format version, `1`                   |
| 5           | 1                     | rows                                  |
| 6           | 1                     | cols                                  |
| 7           | 1                     | numClasses                            |
| 8           | 4                     | count, `uint32`                       |
| 12          | 4                     | reserved, `0`                         |
| 16          | `count × rows × cols` | grayscale pixels, row-major, `0..255` |
| 16 + pixels | `count`               | labels, one byte each                 |

Total length is exactly `16 + count × rows × cols + count`. For the defaults
(1,000 train and 200 test images of 28×28, 10 classes) that is about 785 KB and
157 KB.

`static/mnist/` is added to `.gitignore`, so the data can never be committed.

## 5. Prerequisite fixes

Three findings recorded by the previous phase's final review. The CNN example is
what exposes all three.

### 5.1 `clampNetwork` runs on every mutation

`clampNetwork` is currently called only from `updateBlock`, so `addBlock`,
`moveBlock`, and `load` can leave a `conv2d` whose kernel is larger than its
incoming image, with no correction and no announcement. That is unreachable today
because `conv2d` is not in the MLP palette; it becomes reachable the moment the CNN
palette exists.

All four mutators route their result through `clampNetwork`, and every correction
is announced. A block added by drop is clamped against its resolved
position in the chain, so adding a `conv2d` to a 2×2 input corrects the kernel
immediately rather than leaving an error the user did not cause.

### 5.2 Weights are namespaced per example

`WEIGHTS_URL` is a single constant, so the two examples would overwrite each
other's weights.

`saveWeights(model, id)` and `loadWeightsInto(model, id)` take the example's id,
and a new exported `weightsUrl(id)` returns `indexeddb://visnet/weights/${id}`.
`weightsUrl` is pure and unit-tested; the two IndexedDB calls remain
browser-verified. The CNN page passes `'cnn'`.

**The MLP passes `'main'`, not `'mlp'`,** exported as `MLP_WEIGHTS_ID` from its
example module. That was its address before the ids existed, so keeping it means
weights a user has already saved still load — the same reasoning that keeps the
MLP's network storage key unchanged (§6). The name is a historical artefact and is
documented rather than renamed, because renaming it would silently orphan saved
weights.

### 5.3 Kernel bounds with `'same'` padding

`parameterBounds` clamps `kernelSize` to `min(rows, cols)` even when padding is
`'same'`, where TensorFlow.js would permit a larger kernel by padding the image.

**The stricter clamp is kept deliberately.** A kernel larger than the image is a
strange thing to teach, `'valid'` genuinely cannot exceed the image, and one rule
for both paddings is easier to explain than two. The spec records this so it reads
as a decision rather than an oversight.

## 6. Storage keys become explicit

`storage.ts` hardcodes `NETWORK_KEY` and `DATASET_KEY`, so both examples would
share one saved network.

`createStorage(backing, keys)` and `createBrowserStorage(keys)` take
`{ network: string; dataset: string }`. The **MLP page keeps its existing key
values** (`visnet:network:v1`, `visnet:mlp:dataset:v1`) so a network already saved
in someone's browser still loads; the CNN page uses `visnet:cnn:network:v1` and
`visnet:cnn:dataset:v1` and simply never calls the dataset methods.

Both keys stay required rather than optional, so the interface has no
"sometimes absent" branch.

## 7. Image data modules

### 7.1 `src/lib/data/images.ts` — pure

```ts
export interface ImageDataset {
  count: number;
  rows: number;
  cols: number;
  numClasses: number;
  pixels: Uint8Array; // count * rows * cols, row-major, 0..255
  labels: Uint8Array; // count, each < numClasses
}

export const MAGIC = 'VSNT';
export const FORMAT_VERSION = 1;
export const HEADER_BYTES = 16;

export function parseSplit(buffer: ArrayBuffer): ImageDataset | null;
export function imageAt(dataset: ImageDataset, index: number): Uint8Array;
```

`parseSplit` returns `null` — never throws — for a short buffer, a wrong magic, an
unsupported version, zero rows, cols, count or classes, a length that does not
match the header's dimensions, or any label `>= numClasses`. It copies the pixels
and labels into their own arrays so the returned dataset does not alias the source
buffer.

`imageAt` returns the `rows × cols` slice for one image, throwing a `RangeError`
for an out-of-range index.

### 7.2 `src/lib/data/mnist.ts` — fetching

```ts
export interface MnistData {
  train: ImageDataset;
  test: ImageDataset;
}

export class ImageDataUnavailableError extends Error {}

export async function loadMnistData(base?: string): Promise<MnistData>;
```

Fetches `${base}/train.bin` and `${base}/test.bin` (default base `/mnist`),
parses both, and throws `ImageDataUnavailableError` when either response is not
`ok` or either buffer fails to parse. The message names the prep command, because
that is the only reason this can fail in practice.

### 7.3 `src/lib/data/tensors.ts` — image conversion joins the existing file

`tensors.ts` already converts the point dataset to tensors, so the image
conversion belongs in the same file rather than a new TensorFlow.js import site.

```ts
export function imagesToTensors(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor2D };
```

`xs` is `[n, rows, cols, 1]` float32 with each pixel divided by 255; `ys` is one-hot
`[n, numClasses]`. `indices` selects a subset (used for the sample grid) and
defaults to every image. The caller owns the tensors.

## 8. The CNN example page

`/examples/cnn` replaces the placeholder.

- **Palette**: `['conv2d', 'flatten', 'linear', 'relu', 'sigmoid', 'softmax']`.
  Convolution and flatten are now reachable, which is why §5.1 is a prerequisite.
- **Default network**: `input [28, 28, 1] → conv2d(filters 8, kernel 3, stride 1,
same) → relu → flatten → linear(10) → softmax → output(10)`, with
  `expectedClasses = 10`. This lives in `src/lib/examples/cnn/example.ts` as
  `createCnnNetwork()`, next to `CNN_PALETTE` and `SAMPLE_GRID_SIZE`.
- **`NetworkStore` gains an initial network.** It currently constructs
  `createEmptyNetwork()` and `reset()` returns to that, so the CNN page's "Reset
  network" would produce the MLP's network. The constructor takes
  `initial: Network = createEmptyNetwork()` and remembers it for `reset()`. The MLP
  page is unchanged.
- **Reused unchanged**: `NetworkEditor`, `TrainingPanel`, `LossChart`,
  `StatsReadout`, `ExampleLayout`, `DecisionBoundary` is **not** used.
- **Persisted**: the network only, under the CNN's own key. The image data is not
  persisted, so `saveDataset`/`loadDataset` are never called.
- **Training** uses the same `runtime.ts` facade pattern as the MLP page: the
  TensorFlow.js modules are loaded once after mount and the page's effects stay
  synchronous. The model is rebuilt on architecture change, recompiled on training
  change, and disposed on destroy.
- **Dataset controls**: a "sample size" control is not offered; the dataset is
  whatever the script wrote. The page shows the counts it loaded ("1,000 training
  and 200 test digits").

## 9. The sample grid

`src/lib/components/SampleGrid.svelte`, props:

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

- `SAMPLE_GRID_SIZE` (40) cells in 8 columns × 5 rows, each cell 36 wide and 48 tall:
  36 holds a 28-pixel image with a 4-pixel margin, and 48 leaves a strip beneath it for
  the predicted digit. Eight columns plus gaps is 316 pixels, which fits the example
  column's 320-pixel minimum.
- **Two stacked canvases.** The images never change during training but the marks change
  every frame, so the images are drawn once per dataset change into a base canvas and
  only the borders and digits are redrawn into an overlay each frame. That keeps a frame
  to `strokeRect` and `fillText`, confines the fiddly `ImageData` code to a path that
  runs once, and makes the per-frame path testable with a fake context.
- **The page owns the tensors.** It builds the sample batch and passes it in as
  `sampleXs`; the grid never creates or disposes a tensor, so every tensor's lifetime
  lives in one place.
- Each cell shows the digit image, the predicted digit beneath it, and a border that
  is green when the prediction matches the label and red when it does not.
- Re-evaluated at most once per animation frame while training, driven by the same
  `redrawKey` signal the boundary uses. The 40-image tensor batch is built **once**
  when the dataset or indices change and reused every frame, so a frame costs one
  small forward pass rather than a rebuild.
- While no model exists, the grid draws the images with a neutral border and no
  predictions, with a caption explaining that training has not started.
- `onerror` is called when a prediction pass throws — which happens if the model is
  disposed between the frame being scheduled and the prediction running — so the
  page can report it through the same plain-language banner it uses for training
  failures. The grid then draws images with neutral borders until a model exists
  again.

Pure helpers live in `src/lib/examples/cnn/grid.ts` so the logic is testable and the
component is only drawing:

```ts
export function predictionsFromLogits(
  logits: number[][],
  labels: Uint8Array,
  indices: number[]
): { predicted: number; correct: boolean }[];

export function gridCellRect(
  cell: number,
  columns: number,
  cellWidth: number,
  cellHeight: number,
  gap: number
): { x: number; y: number; width: number; height: number };
```

`predictionsFromLogits` takes the `argMax` of each row and compares it with the
label at the matching index, so `predictions[i]` describes the image at
`indices[i]`; a tie resolves to the lowest class index. `gridCellRect` returns the
cell's top-left pixel and its width and height, **excluding** the gap; the gap is
added between cells only. Both are pure and
unit-tested; the canvas drawing is verified by `svelte-check`, the build, and the
manual checklist — the same accepted limitation as `DecisionBoundary`.

## 10. Error handling

- **Assets absent or unparseable.** `loadMnistData` throws
  `ImageDataUnavailableError`; the page shows a panel reading "The digit images are
  not prepared. Run `npm run data:mnist`, then reload." and disables training. This
  is the expected state on a fresh clone, so it is a designed path, not an error.
- **The Input block disagrees with the data.** The page builds its tensors as
  `[n, 28, 28, 1]` and the model takes its input shape from the Input block, so the two
  can disagree if a user edits the block — producing an opaque TensorFlow.js shape error
  at training time. `validate` therefore gains an `expectedInputShape` option, and the
  page passes `[28, 28, 1]`, which yields a plain-language **warning** naming both
  shapes and telling the user what to set. It is a warning rather than an error because
  the network is still buildable; the mismatch is a statement about the data, exactly
  like the existing output-units warning.
- **Build without assets.** `static/mnist/` is gitignored and may not exist; the
  build simply omits it and still succeeds, because the data is fetched at runtime
  and never during prerendering.
- **Model build failure.** The page catches it, disposes, and shows a plain-language
  banner, as the MLP page does.
- **Training failure.** Reported through the trainer's `onError` as a fixed
  plain-language sentence with the detail logged, as the MLP page does.
- **A stale saved network.** If the stored network cannot be read, the page
  announces it and starts from the CNN default, using the same
  `hasStoredNetwork` distinction the MLP page already relies on.

## 11. Testing strategy

Automated, in the `engine` project unless noted:

- **`parseSplit`** — a valid synthetic buffer round-trips; wrong magic, wrong
  version, a short buffer, trailing bytes, a zero count, zero dimensions, zero
  classes, and a label beyond the class count each return `null`.
- **The format contract** — a round trip that encodes with the prep script's
  `encodeSplit` and parses with `parseSplit`, so the writer and the reader cannot
  drift. This test lives in `scripts/` and therefore requires the engine project's
  `include` to cover `scripts/**/*.test.ts`.
- **The prep script's IDX parsing** — `parseIdxImages` and `parseIdxLabels` against
  synthetic IDX buffers, including a bad magic and a truncated body.
- **`imageAt`** — the correct slice, and a `RangeError` out of range.
- **`loadMnistData`** — a stubbed `fetch` returning a valid pair, a 404, and a
  corrupt body, asserting `ImageDataUnavailableError` in the last two cases.
- **`imagesToTensors`** — shapes `[n, rows, cols, 1]` and `[n, numClasses]`, float32,
  pixel 0 mapping to 0 and 255 to 1, one-hot rows summing to 1, and an `indices`
  subset selecting exactly those images.
- **`grid.ts`** — argmax and correctness for right and wrong predictions, and
  `gridCellRect` geometry including the first and last cells.
- **`weightsUrl`** — two ids produce different URLs.
- **`NetworkStore`** — the constructor's initial network is what `reset()` returns;
  `clampNetwork` runs on `addBlock`, `moveBlock`, and `load`, with the correction
  announced.
- **`storage.ts`** — two key sets do not read each other's data.

The component test project (`ui`) gains nothing for this phase: `SampleGrid` is
canvas-only. Its pure helpers are covered above.

Not covered automatically, and verified by the manual checklist: the prep script's
network behaviour, the sample grid's rendering, and the example page end to end.

## 12. Manual verification checklist

1. `npm run data:mnist` downloads, reports the counts, and writes both files.
2. Running it again does nothing and says so; `--force` re-downloads.
3. `npm run data:mnist -- --train=2000` writes 2,000 training images and the header
   reports the same count.
4. With `static/mnist/` deleted, `/examples/cnn` shows the "not prepared" panel with
   the command, training is disabled, and `npm run build` still succeeds.
5. With the data present, `/examples/cnn` shows the default convolutional network,
   the loaded counts, and a grid of digits with no predictions.
6. Press Play: the loss falls, accuracy climbs, and the grid's red borders become
   green over time.
7. Add a `conv2d` block to a network with a small input and confirm the kernel is
   clamped with an announcement.
8. Reload: the CNN network is restored, and the MLP example still has its own
   network and its own saved weights.
9. Save and load weights on the CNN page, then confirm the MLP page's weights are
   untouched.
10. Reset network on the CNN page yields the convolutional default, not the MLP's.
11. Navigate away while training and back: no console errors, no leaked-tensor
    warnings.

## 13. Documentation

- `README.md`: the CNN example moves out of "not built yet"; note that the digit
  data is downloaded locally with `npm run data:mnist` and is never committed, and
  credit NIST as the origin of MNIST.
- `AGENTS.md`: status reflects both examples; the Commands section gains
  `npm run data:mnist`; a note that `static/mnist/` is gitignored and that the CNN
  example degrades gracefully without it. `tensors.ts` remains the only new
  TensorFlow.js call site, so the allow-list is unchanged.

## 14. Decisions

1. **No data in the repository.** A prep script writes into a gitignored directory.
   Keeps the repo small and avoids redistributing a dataset.
2. **Manual prep, graceful degradation.** `dev` and `build` never touch the network.
3. **A custom binary format rather than PNG or JSON.** Compact, one fetch per split,
   and the parser is pure and testable.
4. **A sample grid rather than a confusion matrix.** It mirrors the MLP's
   "watch it learn" role and is readable at a glance.
5. **The image dataset is not persisted.** It is deterministic, so re-reading is
   cheaper than storing a megabyte, and it avoids generalising the point-dataset
   storage.
6. **Storage keys are parameterised, and the MLP keeps its existing values.** No
   saved network is orphaned by this phase.
7. **Weights are namespaced by example id, and the MLP keeps its existing address.**
   The two examples cannot overwrite each other, and no saved weights are orphaned
   by the change — the same principle applied to the storage keys in §6.
8. **The kernel bound stays strict under `'same'` padding.** One rule is easier to
   teach than two, and an oversized kernel is not worth supporting.
9. **`NetworkStore` takes its initial network.** Otherwise the CNN page's Reset
   would produce the MLP's network.
10. **The image conversion joins `data/tensors.ts`.** It is the same job as the
    point conversion, and it avoids a sixth TensorFlow.js import site.
