# VisNet — Reshape layer and shape-based Output block

Date: 2026-09-17
Status: approved (pending written-spec review)

## 1. Overview

This phase generalises the network's endpoints so a network can produce and
declare an **image-shaped** output, not only a flat list of class scores. It adds
a **Reshape** block and changes the Output block from a class count (`units`) to
a **shape**.

It is the prerequisite for the autoencoder example (a separate spec), whose
decoder must reshape a flat vector back into an image.

## 2. Goals

- Add a `reshape` block that rearranges a tensor into a declared shape.
- Let the Output block declare any output shape (e.g. `[2]` or `[28,28,1]`).
- Keep classification networks working unchanged in behaviour.
- Load networks saved by the previous build.

## 3. Non-goals

- The autoencoder example itself (separate spec).
- Arbitrary DAGs, branching, or multi-input blocks. The chain stays linear.
- A general task/objective concept on the `Network`; that belongs to the
  autoencoder spec.

## 4. Domain model

`src/lib/network/types.ts`:

```ts
export interface ReshapeBlock extends BlockBase {
  kind: 'reshape';
  shape: number[]; // the target shape, e.g. [28, 28, 1]
}

export interface OutputBlock extends BlockBase {
  kind: 'output';
  shape: number[]; // the shape the network should produce, e.g. [2]
}
```

`reshape` joins `BlockKind` and `BLOCK_KINDS` after `flatten`. The Output block
loses `units`.

Defaults (`factory.ts`):

| Block     | Default          |
| --------- | ---------------- |
| `reshape` | `shape: [1]`     |
| `output`  | `shape: [2]`     |

`createEmptyNetwork` keeps an Output of `shape: [2]`.

## 5. Shape inference

`inferShapes.ts`:

- `reshape` → `inShape ? [...block.shape] : null`. It contributes no parameters.
- `output` is unchanged: it is a marker that passes its input shape through.

## 6. Validation

`problems.ts`:

**Reshape size mismatch (error).** When the incoming shape is known and the
element counts differ:

- title: `Reshape size does not match`
- message: `This Reshape layer receives {n} numbers, but the shape {shape} holds {m}.`
- fix: `Change the Reshape shape so it holds {n} numbers, for example [{n}].`

**Output shape mismatch (error).** Replaces the old "Output must be a list of
scores" and "Last layer size does not match the Output block" rules. When the
last real layer has a known shape that differs from `output.shape`:

- title: `Last layer shape does not match the Output block`
- message: `The last layer produces {shape}, but the Output block expects {outputShape}.`
- fix: `Set the last layer to produce {outputShape}, or change the Output block to {shape}.`

**Cross-entropy needs a flat output (warning).** Because a rank-1 requirement is
no longer implied by the Output block, warn when the loss is cross-entropy and
the last real layer's shape is not rank 1:

- title: `Cross-entropy needs a flat output`
- message: `Cross-entropy compares a flat list of scores with the labels, but the last layer produces {shape}.`
- fix: `End the network with a Flatten or Linear layer, or switch the loss to mean squared error.`

**Expected classes (warning).** Unchanged in spirit, but reads
`output.shape[0]` and only fires when `output.shape` is rank 1.

## 7. Model building

`tf/buildModel.ts`:

- `reshape` → `tf.layers.reshape({ targetShape: block.shape, inputShape })`.
- `output` remains skipped.

## 8. Persistence

`src/lib/persist/networkCodec.ts`:

- `output` requires a numeric `shape` array; `reshape` likewise.
- **Compatibility shim:** before validating, a block with `kind: 'output'` and no
  `shape` but a numeric `units` is rewritten to `shape: [units]`. Networks saved
  by the previous build therefore still load.

## 9. Inspector

`InspectorPanel.svelte`:

- The Input, Output, and Reshape blocks share one **shape** text field (positive
  whole numbers separated by commas), reusing the existing input-shape parsing
  and error handling.
- The Linear block keeps its `units` number field; the Output block no longer
  has one.
- The Reshape block's shape field is accompanied by the incoming element count
  so the user knows what the shape must multiply to.

## 10. Descriptions

`descriptions.ts`:

- `reshape`: "Rearranges the numbers into a different shape without changing them."
- `output`: "Declares the shape the network predicts."
- New parameter descriptions: `outputShape` ("The shape the network should
  produce, for example 2 numbers or a 28 by 28 image.") and `reshapeShape`
  ("The new shape for the same numbers.").

## 11. Palettes and examples

- `reshape` is added to the CNN palette so it is reachable in the UI.
- `examples/cnn/example.ts` builds its Output with `shape: [10]`.
- The MLP example is unchanged (it uses the default `[2]` Output).

## 12. Testing

- `types.test.ts`, `factory.test.ts`: `reshape` kind and defaults; Output shape.
- `inferShapes.test.ts`: reshape output shape; reshape null propagation.
- `problems.test.ts`: reshape mismatch, output shape mismatch, cross-entropy
  rank warning, expected-classes via `shape[0]`; the rule catalog is updated.
- `networkCodec.test.ts`: reshape and output shape validation; the `units` shim
  loads a legacy Output.
- `descriptions.test.ts`: every kind and parameter is described.
- `InspectorPanel.test.ts`: the shared shape field edits Input, Output, and
  Reshape; Linear keeps `units`.
- Fixture updates: every test that builds an Output block switches from
  `units: n` to `shape: [n]` (`probe`, `chain`, `flow`, `placement`,
  `networkStore`, `buildModel`, `cnn/example`).
- `buildModel.test.ts`: a reshape layer builds and the network output shape
  matches.

## 13. Files

- Modify: `src/lib/network/types.ts`, `factory.ts`, `inferShapes.ts`,
  `problems.ts`, `descriptions.ts`
- Modify: `src/lib/persist/networkCodec.ts`
- Modify: `src/lib/tf/buildModel.ts`
- Modify: `src/lib/components/InspectorPanel.svelte`
- Modify: `src/lib/examples/cnn/example.ts`
- Modify: `README.md`
- Modify the tests listed in section 12.

## 14. Manual verification

1. Add a Reshape block; set its shape to the incoming element count — no error.
2. Set a wrong shape — the "Reshape size does not match" error appears with a fix.
3. Change the Output block's shape and watch the last-layer error appear and clear.
4. Build a network ending in Reshape `[28,28,1]` with an Output of `[28,28,1]` —
   no errors.
5. Reload a network saved before this change — it still loads (the `units` shim).
6. A classification network behaves exactly as before.
