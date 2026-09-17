# VisNet MaxPool2D design

## 1. Goal

Add a `maxpool2d` block kind: a spatial layer that shrinks an image by keeping the
largest value in each small window. It joins `conv2d` as the second spatial layer, so a
learner can build the textbook `conv -> relu -> pool -> flatten -> dense` pipeline.

**In scope:** the block kind end to end — type, factory, shape inference, clamping,
validation, serialization, the TensorFlow.js layer, the inspector controls, the CNN
palette, tests, and documentation.

**Out of scope:** global and average pooling, pooling for rank-1 input, changing the
default CNN network, and any change to the MLP example. `maxpool2d` is available to drag
in; the shipped default networks are untouched.

## 2. The block kind

`src/lib/network/types.ts`:

```ts
export interface MaxPool2dBlock extends BlockBase {
  kind: 'maxpool2d';
  poolSize: number;
  stride: number;
  padding: 'same' | 'valid';
}
```

`maxpool2d` is added to the `BlockKind` union, to the `Block` union, and to
`BLOCK_KINDS` immediately after `conv2d`, keeping the two spatial layers adjacent. The
order of `BLOCK_KINDS` is asserted by `types.test.ts`, so that test's expected array
gains the new kind in the same position, and the fixture network in the same test gains
a `maxpool2d` block so that every kind remains represented.

`src/lib/network/factory.ts` gains:

```ts
case 'maxpool2d':
  return { id: newBlockId(), kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' };
```

The default is the textbook halving: a 2×2 window moving two steps. Note that this
defaults to `valid`, unlike `conv2d`'s `same`. That difference is deliberate: a
convolution often wants to preserve its input's size so layers can be stacked, whereas
pooling exists precisely to shrink, and `valid` is the conventional pooling default.

## 3. Shape inference

`src/lib/network/inferShapes.ts` renames `convOutputSize` to `spatialOutputSize` and
keeps its body unchanged, because the pooling formula is identical to the convolution
one: `ceil(size / stride)` for `same`, and `floor((size - window) / stride) + 1` for
`valid`. Its second parameter is renamed from `kernelSize` to `window`, since it now
stands for a convolution kernel or a pooling window. The rename touches the two call
sites inside this module, the import in `InspectorPanel.svelte`, and the tests that
import it.

A new case in `outputShapeFor`:

```ts
case 'maxpool2d': {
  if (!inShape || inShape.length !== 3) return null;
  const [height, width, channels] = inShape;
  const outHeight = spatialOutputSize(height, block.poolSize, block.stride, block.padding);
  const outWidth = spatialOutputSize(width, block.poolSize, block.stride, block.padding);
  if (outHeight <= 0 || outWidth <= 0) return null;
  return [outHeight, outWidth, channels];
}
```

Pooling passes the channel count through unchanged, unlike `conv2d`, which replaces it
with `filters`. Pooling has no trainable weights, so `paramCountFor` needs no new case:
its default branch already returns 0, and a MaxPool node therefore reports "0
parameters" in the node, the tooltip, and the shape table.

## 4. Parameter clamping

`src/lib/network/constraints.ts`. `parameterBounds(inShape)` is already kind-agnostic —
it keys off an incoming shape of rank 3 rather than off `conv2d` — so it is reused
unchanged, giving Pool size and Stride the same 1..min(H, W) choices as the convolution
kernel and stride.

`clampBlockPatch` gains a `poolSize` case that clamps like `kernelSize` does, with the
label `Pool size` and the existing dimension message: `Pool size changed from 7 to 4
because the incoming data is 4×4.` `stride` is already handled by the shared branch and
needs nothing new.

`clampNetwork` currently skips every block that is not `conv2d`. It is generalised so
that `conv2d` clamps `{ kernelSize, stride }` and `maxpool2d` clamps `{ poolSize, stride
}`, and every other kind is skipped. Without this, a pool dragged onto a small image
would be left oversized and broken instead of being corrected with an announcement.

## 5. Validation

`src/lib/network/validate.ts` gains two error rules, mirroring the convolution ones.

| Condition                                                       | Title                                | Message                                                                                           | Fix                                                                               |
| --------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `maxpool2d` on input that is not rank 3                         | Pooling layer needs image data       | `This Pooling layer receives {shape}. It expects image data shaped [height, width, channels].`    | Give the Input block a 3D shape such as [28, 28, 1], or remove the Pooling layer. |
| `maxpool2d` on rank-3 input whose output dimension would be ≤ 0 | Pool window is larger than the image | `A {poolSize}×{poolSize} pool with stride {stride} leaves no room to slide over a {H}×{W} image.` | Use a smaller pool or stride, or set padding to 'same'.                           |

Both carry the offending `blockId`. Every issue keeps a non-empty `title`, `message`, and
`fix`, and both rules are added to the message-contract table in `validate.test.ts`.

The existing `hasConvolution` check stays keyed to `conv2d` only. A network whose image
input flows through pooling but no convolution still receives the "Image input without a
Convolution layer" warning, which is correct: pooling selects and shrinks, but it has no
weights, so it cannot learn anything about the image.

## 6. TensorFlow.js mapping

`src/lib/tf/buildModel.ts` gains:

```ts
case 'maxpool2d':
  return tf.layers.maxPooling2d({
    poolSize: block.poolSize,
    strides: block.stride,
    padding: block.padding,
    inputShape
  });
```

`inputShape` is passed on the first layer exactly as the other cases do. Because
`buildModel` is covered by tests that construct and run a real model, this case is
verified against the library rather than assumed.

## 7. The inspector

`src/lib/components/InspectorPanel.svelte` gains a `maxpool2d` branch with three
controls, matching the convolution controls in layout and behaviour:

- **Pool size** — a select over the same `1..min(H, W)` choices, labelled `2×2` and so on.
- **Stride** — a select over the same choices.
- **Padding** — the existing `same` / `valid` select, each option annotated with the
  resulting size, for example `valid — becomes 14×14`.

Three pieces of existing code currently test `block.kind === 'conv2d'` and must be
generalised to cover both spatial kinds: the `kernelChoices` and `strideChoices`
deriveds, and `outputSizeFor`, which must read `poolSize` for a pooling block and
`kernelSize` for a convolution.

## 8. The palette

`src/lib/examples/cnn/example.ts` adds `'maxpool2d'` to `CNN_PALETTE`, directly after
`'conv2d'`. The MLP palette is unchanged: its input is a flat list of two numbers, so a
pooling block would always be invalid there. The default CNN network is unchanged, so
MaxPool is something a learner drags in rather than something the example ships with.

## 9. Serialization

`src/lib/network/serialize.ts` adds a `maxpool2d` case to `isBlock`, requiring a finite
positive `poolSize`, a finite positive `stride`, and a `padding` of `same` or `valid`.

This case is load-bearing. A saved network is rejected whole if any block fails
`isBlock`, so without it a learner who saved a network containing a pool block would
reload to find their work discarded and the page reporting that the saved network could
not be read.

## 10. Testing

Unit tests, colocated with each module:

- `types.test.ts` — the `BLOCK_KINDS` array and the every-kind fixture.
- `factory.test.ts` — a `maxpool2d` default with a unique id (the existing `it.each`
  over `BLOCK_KINDS` covers this once the kind exists).
- `descriptions.test.ts` — the existing every-kind description check.
- `inferShapes.test.ts` — `spatialOutputSize` for both paddings; pooling output shapes
  including channel pass-through, stride 2, and `valid` shrinking; a null output when
  the window does not fit; and a parameter count of 0.
- `constraints.test.ts` — clamping an oversized `poolSize` and `stride`, and
  `clampNetwork` correcting a pooling block rather than only a convolution.
- `validate.test.ts` — both new rules, and their rows in the message-contract table.
- `serialize.test.ts` — a round trip through a network containing `maxpool2d`, and
  rejection of a malformed pool block.
- `buildModel.test.ts` — building a model that contains `maxPooling2d` and confirming its
  output shape matches the inferred shape.
- `InspectorPanel.test.ts` — the pool controls render, and changing one patches the
  block.
- `examples/cnn/example.test.ts` — `CNN_PALETTE` contains `maxpool2d`.

## 11. Documentation

A new spec, this document, is the authoritative record for the feature. In addition, the
master design document `docs/superpowers/specs/2026-09-16-visnet-design.md` enumerates
the block set in several engine tables — the block-kind union, the defaults table, the
block-description table, the shape-inference table, the validation-rule table, the
parameter-bounds list, and the TensorFlow.js mapping table. Those tables are the closest
thing the project has to an engine reference, so each gains a `maxpool2d` row rather than
being left to describe a block set that no longer matches the code.

## 12. Manual verification checklist

1. On `/examples/cnn`, drag a `maxpool2d` block onto the canvas and confirm it lands and
   shows `[28 × 28 × 8] → [14 × 14 × 8]` and `0 parameters`.
2. Select it and confirm the inspector shows Pool size, Stride, and Padding, with the
   padding options annotated with the resulting size.
3. Add a pool with a large pool size, then change the Input shape to something smaller
   such as `[4, 4, 1]` and confirm the pool size is clamped with an announcement rather
   than left broken. (The Pool size select only offers sizes that fit, so an oversized
   pool arises from the image shrinking underneath it, not from the select.)
4. Set padding to `same` and confirm the output size updates to match.
5. Train a network containing the pool block and confirm the loss falls, so the layer is
   wired into the real model rather than only into the shape maths.
6. Reload and confirm the network containing the pool block is restored intact.
7. Drag a `maxpool2d` onto a network whose Input is flat, such as `[2]`, and confirm the
   "Pooling layer needs image data" error appears.

## 13. Decisions

1. **`poolSize`, `stride`, and `padding` are all exposed.** This mirrors `conv2d`, so the
   inspector, the clamping bounds, and the descriptions are reused rather than
   reinvented, and padding stays a visible teaching point on both spatial layers.
2. **Pooling defaults to `valid`, unlike convolution's `same`.** Pooling exists to
   shrink, and `valid` is the conventional default for it.
3. **The spatial helpers are shared rather than duplicated.** The output-size formula is
   genuinely the same for both layers, and `parameterBounds` was already written in terms
   of rank-3 input rather than conv2d, so `maxpool2d` becomes a peer of `conv2d` instead
   of a parallel copy that can drift.
4. **Pooling does not count as a Convolution layer** for the existing image-input
   warning, because it has no trainable weights and cannot learn.
5. **The block is named `maxpool2d`**, mirroring `conv2d`, so the two spatial layers read
   as a pair wherever the kind is shown verbatim.
