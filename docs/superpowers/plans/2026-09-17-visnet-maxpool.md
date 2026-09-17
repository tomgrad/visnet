# MaxPool2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `maxpool2d` block kind so a learner can build the textbook `conv -> relu -> pool -> flatten -> dense` pipeline.

**Architecture:** `maxpool2d` becomes a peer of `conv2d`, not a copy of it. The pooling output-size formula is identical to the convolution one, and `parameterBounds()` in `constraints.ts` already keys off "incoming shape is rank 3" rather than off conv2d, so the two layers share the spatial helpers. Pooling has no trainable weights, so it contributes 0 parameters and channels pass through untouched.

**Tech Stack:** SvelteKit 2, Svelte 5 runes, TypeScript strict, Vitest (`engine` and `ui` projects), `@tensorflow/tfjs`.

**Spec:** `docs/superpowers/specs/2026-09-17-visnet-maxpool-design.md`

## Global Constraints

- Package manager is **npm**. Do not use pnpm, yarn, or bun.
- TypeScript strict mode. `npm run check` must report 0 errors and 0 warnings.
- Svelte 5 runes only.
- Do not add code comments unless a non-obvious constraint requires one.
- No `any` in production code.
- `src/lib/network/**` is pure: no Svelte, no TensorFlow.js, no DOM. It must run in plain Node.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, `src/lib/persist/weights.ts`, and the test files colocated with those modules.
- Every validation issue carries a non-empty `title`, `message`, and `fix`.
- `npm run lint`, `npm run check`, `npm test`, and `npm run build` must all pass before each commit.
- Do NOT commit anything under `.superpowers/`.
- Plain CSS with existing tokens only.
- The block is named exactly `maxpool2d`, and its parameters are exactly `poolSize`, `stride`, and `padding`.

---

### Task 1: The block kind, its defaults, and its persistence

**Files:**
- Modify: `src/lib/network/types.ts`
- Modify: `src/lib/network/factory.ts`
- Modify: `src/lib/network/descriptions.ts`
- Modify: `src/lib/network/serialize.ts`
- Test: `src/lib/network/types.test.ts`
- Test: `src/lib/network/factory.test.ts`
- Test: `src/lib/network/descriptions.test.ts`
- Test: `src/lib/network/serialize.test.ts`

**Interfaces:**
- Produces: `MaxPool2dBlock { kind: 'maxpool2d'; poolSize: number; stride: number; padding: 'same' | 'valid' }`, added to `BlockKind`, `BLOCK_KINDS`, and the `Block` union. `createBlock('maxpool2d')` returns `{ poolSize: 2, stride: 2, padding: 'valid' }`. `BLOCK_DESCRIPTIONS.maxpool2d` and `PARAM_DESCRIPTIONS.poolSize` exist. `isBlock` in `serialize.ts` accepts the kind.

Adding `maxpool2d` to the `BlockKind` union makes `createBlock` non-exhaustive and makes `BLOCK_DESCRIPTIONS` (typed `Record<BlockKind, string>`) incomplete, so all three of `types.ts`, `factory.ts`, and `descriptions.ts` must change together for the project to type-check.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/types.test.ts`, change the first test's name and its expected array, and add a fixture block. The `BLOCK_KINDS` array is asserted in order, so `maxpool2d` goes immediately after `conv2d`:

```ts
describe('BLOCK_KINDS', () => {
  it('lists the nine block kinds in pipeline order', () => {
    expect([...BLOCK_KINDS]).toEqual([
      'input',
      'linear',
      'conv2d',
      'maxpool2d',
      'flatten',
      'relu',
      'sigmoid',
      'softmax',
      'output'
    ]);
  });
```

In the `domain types` describe, insert a pooling block into the fixture so every kind stays represented:

```ts
      { id: 'c', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'i', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      { id: 'd', kind: 'flatten' },
```

In `src/lib/network/factory.test.ts`, extend the `applies the documented defaults` test:

```ts
    expect(createBlock('maxpool2d')).toMatchObject({
      kind: 'maxpool2d',
      poolSize: 2,
      stride: 2,
      padding: 'valid'
    });
```

In `src/lib/network/serialize.test.ts`, add a fixture beside the existing `INPUT`, `OUTPUT`, and `LINEAR` constants and three tests inside the `fromJSON` describe:

```ts
const IMAGE = { id: 'img', kind: 'input', shape: [28, 28, 1] };
const MAXPOOL = { id: 'p', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' };
```

```ts
  it('round-trips a maxpool2d block', () => {
    const restored = fromJSON(JSON.stringify(envelope([IMAGE, MAXPOOL, OUTPUT])));
    expect(restored?.blocks[1]).toEqual(MAXPOOL);
  });

  it('rejects a maxpool2d block with an invalid padding', () => {
    const pool = { id: 'p', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'half' };
    expect(fromJSON(JSON.stringify(envelope([IMAGE, pool, OUTPUT])))).toBeNull();
  });

  it('rejects a maxpool2d block with no pool size', () => {
    const pool = { id: 'p', kind: 'maxpool2d', stride: 2, padding: 'valid' };
    expect(fromJSON(JSON.stringify(envelope([IMAGE, pool, OUTPUT])))).toBeNull();
  });
```

In `src/lib/network/descriptions.test.ts`, the `covers every tunable parameter` test asserts the
exhaustive `PARAM_DESCRIPTIONS` key list, so `'poolSize'` joins it in alphabetical position:

```ts
        'padding',
        'poolSize',
        'stride',
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run --project engine src/lib/network/types.test.ts src/lib/network/factory.test.ts src/lib/network/descriptions.test.ts src/lib/network/serialize.test.ts`

Expected: FAIL. `types.test.ts` fails the ordered array, `factory.test.ts` fails because `createBlock('maxpool2d')` returns `undefined`, `descriptions.test.ts` fails because `BLOCK_DESCRIPTIONS.maxpool2d` is missing, and only the *round-trip* serialize test fails. The two negative serialize tests pass even before this task, because a malformed `maxpool2d` block is rejected by `isBlock`'s `default: return false` branch either way; they are regression guards for the explicit case's internals, not RED evidence.

- [ ] **Step 3: Add the block kind**

In `src/lib/network/types.ts`:

```ts
export type BlockKind =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'maxpool2d'
  | 'flatten'
  | 'relu'
  | 'sigmoid'
  | 'softmax'
  | 'output';

export const BLOCK_KINDS: readonly BlockKind[] = [
  'input',
  'linear',
  'conv2d',
  'maxpool2d',
  'flatten',
  'relu',
  'sigmoid',
  'softmax',
  'output'
];
```

Add the interface directly after `Conv2dBlock`:

```ts
export interface MaxPool2dBlock extends BlockBase {
  kind: 'maxpool2d';
  poolSize: number;
  stride: number;
  padding: 'same' | 'valid';
}
```

And extend the union:

```ts
export type Block =
  | InputBlock
  | LinearBlock
  | Conv2dBlock
  | MaxPool2dBlock
  | FlattenBlock
  | ActivationBlock
  | OutputBlock;
```

- [ ] **Step 4: Add the factory default**

In `src/lib/network/factory.ts`, add a case to `createBlock` directly after the `conv2d` case:

```ts
    case 'maxpool2d':
      return { id: newBlockId(), kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' };
```

Note the default is `valid`, deliberately unlike `conv2d`'s `same`: pooling exists to shrink, and `valid` is its conventional default.

- [ ] **Step 5: Add the descriptions**

In `src/lib/network/descriptions.ts`, add to `BLOCK_DESCRIPTIONS` directly after `conv2d`:

```ts
  maxpool2d: 'Shrinks an image by keeping the largest value in each small window.',
```

And add to `PARAM_DESCRIPTIONS`, after `kernelSize`:

```ts
  poolSize: 'How large the window is. The largest value in it survives.',
```

- [ ] **Step 6: Accept the kind in serialization**

In `src/lib/network/serialize.ts`, add a case to `isBlock` directly after the `conv2d` case:

```ts
    case 'maxpool2d':
      return (
        isFinitePositive(value.poolSize) &&
        isFinitePositive(value.stride) &&
        (value.padding === 'same' || value.padding === 'valid')
      );
```

This case is load-bearing: a saved network is rejected whole if any block fails `isBlock`, so without it a learner who saved a network containing a pool would reload to find their work discarded.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `npx vitest run --project engine src/lib/network/types.test.ts src/lib/network/factory.test.ts src/lib/network/descriptions.test.ts src/lib/network/serialize.test.ts`

Expected: PASS.

Then run the whole engine suite, because `descriptions.test.ts` iterates `BLOCK_KINDS` and asserts every kind has a description:

Run: `npm test`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/lib/network/types.ts src/lib/network/factory.ts src/lib/network/descriptions.ts src/lib/network/serialize.ts src/lib/network/types.test.ts src/lib/network/factory.test.ts src/lib/network/descriptions.test.ts src/lib/network/serialize.test.ts
git commit -m "feat: add the maxpool2d block kind, its defaults, and its persistence"
```

---

### Task 2: Shape inference and the spatial rename

**Files:**
- Modify: `src/lib/network/inferShapes.ts`
- Modify: `src/lib/components/InspectorPanel.svelte`
- Test: `src/lib/network/inferShapes.test.ts`

**Interfaces:**
- Consumes: `MaxPool2dBlock` from Task 1.
- Produces: `spatialOutputSize(size, window, stride, padding)` — the renamed `convOutputSize`, same body, second parameter renamed from `kernelSize` to `window`. Pooling output shape is `[outHeight, outWidth, channels]`.

`convOutputSize` is exported and imported by `InspectorPanel.svelte`, so the rename must update that import in the same task to keep `npm run check` green. Task 6 rewrites the rest of that file.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/inferShapes.test.ts`, update the import and rename the existing describe:

```ts
import { inferShapes, spatialOutputSize } from './inferShapes';
```

```ts
describe('spatialOutputSize', () => {
  it('keeps the size with same padding', () => {
    expect(spatialOutputSize(28, 3, 1, 'same')).toBe(28);
    expect(spatialOutputSize(28, 3, 2, 'same')).toBe(14);
  });

  it('shrinks with valid padding', () => {
    expect(spatialOutputSize(28, 3, 1, 'valid')).toBe(26);
    expect(spatialOutputSize(28, 5, 2, 'valid')).toBe(12);
  });
});
```

Add these describes at the end of the file:

```ts
describe('inferShapes on a pooling chain', () => {
  const result = inferShapes(
    net([
      { id: 'in', kind: 'input', shape: [28, 28, 3] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'out', kind: 'output', units: 10 }
    ])
  );

  it('halves each spatial dimension and passes the channels through', () => {
    expect(result.perBlock[1].outShape).toEqual([14, 14, 3]);
  });

  it('costs no trainable parameters', () => {
    expect(result.perBlock[1].paramCount).toBe(0);
  });

  it('flattens the pooled result', () => {
    expect(result.perBlock[2].outShape).toEqual([14 * 14 * 3]);
  });
});

describe('inferShapes with same-padded pooling', () => {
  it('halves an even size', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'same' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].outShape).toEqual([14, 14, 1]);
  });

  it('rounds up an odd size', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [27, 27, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'same' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].outShape).toEqual([14, 14, 1]);
  });
});

describe('inferShapes with an impossible pool', () => {
  it('returns a null output shape when the window does not fit', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [2, 2, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 3, stride: 3, padding: 'valid' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].inShape).toEqual([2, 2, 1]);
    expect(result.perBlock[1].outShape).toBeNull();
  });

  it('returns a null output shape for flat input', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [784] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].outShape).toBeNull();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run --project engine src/lib/network/inferShapes.test.ts`

Expected: FAIL to even load, because `spatialOutputSize` is not exported yet, and the pooling describes fail because `outputShapeFor` has no `maxpool2d` case.

- [ ] **Step 3: Rename the helper and add the pooling case**

In `src/lib/network/inferShapes.ts`, rename the function and its second parameter:

```ts
export function spatialOutputSize(
  size: number,
  window: number,
  stride: number,
  padding: 'same' | 'valid'
): number {
  if (padding === 'same') return Math.ceil(size / stride);
  return Math.floor((size - window) / stride) + 1;
}
```

Update the two calls inside the `conv2d` case to use the new name, and add a `maxpool2d` case directly after it:

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

Do **not** add a `maxpool2d` case to `paramCountFor`. Its `default` branch already returns 0, which is the correct count for a layer with no weights.

- [ ] **Step 4: Update the inspector's import**

In `src/lib/components/InspectorPanel.svelte`, update the import and the one call site:

```ts
  import { spatialOutputSize } from '../network/inferShapes';
```

```ts
      spatialOutputSize(dimension, kernelSize, stride, padding);
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run --project engine src/lib/network/inferShapes.test.ts`

Expected: PASS.

Run: `npm run check`

Expected: `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 6: Commit**

```bash
git add src/lib/network/inferShapes.ts src/lib/network/inferShapes.test.ts src/lib/components/InspectorPanel.svelte
git commit -m "feat: infer maxpool2d output shapes from the shared spatial helper"
```

---

### Task 3: Parameter clamping

**Files:**
- Modify: `src/lib/network/constraints.ts`
- Test: `src/lib/network/constraints.test.ts`

**Interfaces:**
- Consumes: `MaxPool2dBlock` from Task 1, `spatialOutputSize` from Task 2 (not used directly here, but the same bounds apply).
- Produces: `clampBlockPatch` clamps `poolSize` with the label `Pool size`; `clampNetwork` corrects pooling blocks as well as convolutions.

`parameterBounds(inShape)` needs no change: it keys off an incoming shape of rank 3, so it already returns the 1..min(H, W) choices that pooling needs.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/constraints.test.ts`, add a fixture beside `IMAGE_NETWORK` and `FLAT_NETWORK`:

```ts
const POOL_NETWORK = net([
  { id: 'in', kind: 'input', shape: [28, 28, 1] },
  { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
  { id: 'flat', kind: 'flatten' },
  { id: 'dense', kind: 'linear', units: 10 },
  { id: 'out', kind: 'output', units: 10 }
]);
```

Add these tests to the `clampBlockPatch` describe:

```ts
  it('leaves a valid pool untouched and says nothing', () => {
    expect(clampBlockPatch(POOL_NETWORK, 'pool', { poolSize: 2, stride: 2 })).toEqual({
      patch: { poolSize: 2, stride: 2 },
      announcement: null
    });
  });

  it('clamps a pool window larger than the image and names the dimensions', () => {
    const result = clampBlockPatch(POOL_NETWORK, 'pool', { poolSize: 40 });
    expect(result.patch).toEqual({ poolSize: 28 });
    expect(result.announcement).toBe(
      'Pool size changed from 40 to 28 because the incoming data is 28×28.'
    );
  });
```

Add this test to the `clampNetwork` describe:

```ts
  it('re-clamps a downstream pool when the input shape shrinks', () => {
    const shrunk = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 28, stride: 1, padding: 'valid' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'out', kind: 'output', units: 10 }
    ]);

    const result = clampNetwork(shrunk);

    expect(result.network.blocks[1]).toMatchObject({ poolSize: 4, stride: 1 });
    expect(result.announcements).toEqual([
      'Pool size changed from 28 to 4 because the incoming data is 4×4.'
    ]);
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run --project engine src/lib/network/constraints.test.ts`

Expected: FAIL. `clampBlockPatch` ignores `poolSize`, so the patch comes back as `{ poolSize: 40 }` with a null announcement, and `clampNetwork` skips the pooling block entirely.

- [ ] **Step 3: Clamp the pool size in a patch**

In `src/lib/network/constraints.ts`, inside `clampBlockPatch`, add a `poolSize` branch between the existing `kernelSize` and `stride` branches, so the first-reported correction keeps the documented order:

```ts
    if ('poolSize' in patch && typeof patch.poolSize === 'number') {
      const next = clampDimension(patch.poolSize, 'Pool size', limit, inShape);
      Object.assign(result, { poolSize: next.value });
      if (announcement === null) announcement = next.announcement;
    }
```

- [ ] **Step 4: Correct pooling blocks on the way in**

In `src/lib/network/constraints.ts`, replace the body of the `forEach` inside `clampNetwork` so it handles both spatial kinds:

```ts
    current.blocks.forEach((block, index) => {
      const patch: Partial<Block> | null =
        block.kind === 'conv2d'
          ? { kernelSize: block.kernelSize, stride: block.stride }
          : block.kind === 'maxpool2d'
            ? { poolSize: block.poolSize, stride: block.stride }
            : null;
      if (!patch) return;

      const bounds = parameterBounds(perBlock[index].inShape);
      if (!bounds) return;

      const corrected = clampBlockPatch(current, block.id, patch);
      if (!corrected.announcement) return;

      current = replaceBlock(current, block.id, corrected.patch);
      announcements.push(corrected.announcement);
      changed = true;
    });
```

`Partial<Block>` is already imported in this file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run --project engine src/lib/network/constraints.test.ts`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/network/constraints.ts src/lib/network/constraints.test.ts
git commit -m "feat: clamp maxpool2d pool sizes like convolution kernels"
```

---

### Task 4: Validation rules

**Files:**
- Modify: `src/lib/network/validate.ts`
- Test: `src/lib/network/validate.test.ts`

**Interfaces:**
- Consumes: `MaxPool2dBlock` from Task 1; the null output shape Task 2 produces for an impossible pool.
- Produces: two error rules, `Pooling layer needs image data` and `Pool window is larger than the image`.

`validate.test.ts` has a message-contract table whose `has a case for every rule and no case without a rule` test asserts that the set of titles produced across all cases equals the set of declared rule titles. Adding a rule without adding a case fails that test, and adding a case that produces an undeclared title fails it too.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/validate.test.ts`, add these tests to the `validate errors` describe, after the existing convolution tests:

```ts
  it('reports a pooling layer receiving flat data', () => {
    const network = net([
      INPUT,
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Pooling layer needs image data');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('pool');
    expect(issue?.message).toContain('[2]');
  });

  it('reports a pool window larger than the image', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [2, 2, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Pool window is larger than the image');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('pool');
    expect(issue?.message).toContain('3');
  });
```

Add both titles to `ERROR_RULE_TITLES`, after `'Kernel is larger than the image'`:

```ts
  'Pooling layer needs image data',
  'Pool window is larger than the image',
```

Add both cases to `RULE_CASES`, after the `'Kernel is larger than the image'` entry:

```ts
  {
    title: 'Pooling layer needs image data',
    severity: 'error',
    network: net([
      INPUT,
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      OUTPUT
    ])
  },
  {
    title: 'Pool window is larger than the image',
    severity: 'error',
    network: net([
      { id: 'small', kind: 'input', shape: [2, 2, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ])
  },
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run --project engine src/lib/network/validate.test.ts`

Expected: FAIL. Both new tests find no matching issue, and the contract test fails because `ERROR_RULE_TITLES` declares two rules that no case produces.

- [ ] **Step 3: Add the rules**

In `src/lib/network/validate.ts`, inside the `net.blocks.forEach` block, add both rules directly after the existing `conv2d` output-dimension rule:

```ts
    if (block.kind === 'maxpool2d' && info.inShape && info.inShape.length !== 3) {
      issues.push({
        severity: 'error',
        title: 'Pooling layer needs image data',
        message: `This Pooling layer receives ${shapeText(info.inShape)}. It expects image data shaped [height, width, channels].`,
        fix: 'Give the Input block a 3D shape such as [28, 28, 1], or remove the Pooling layer.',
        blockId: block.id
      });
    }

    if (
      block.kind === 'maxpool2d' &&
      info.inShape &&
      info.inShape.length === 3 &&
      info.outShape === null
    ) {
      const [height, width] = info.inShape;
      issues.push({
        severity: 'error',
        title: 'Pool window is larger than the image',
        message: `A ${block.poolSize}×${block.poolSize} pool with stride ${block.stride} leaves no room to slide over a ${height}×${width} image.`,
        fix: "Use a smaller pool or stride, or set padding to 'same'.",
        blockId: block.id
      });
    }
```

Leave `hasConvolution` keyed to `conv2d` alone. Pooling has no weights, so a network of pooling without convolution should still receive the "Image input without a Convolution layer" warning.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run --project engine src/lib/network/validate.test.ts`

Expected: PASS, including `has a case for every rule and no case without a rule`.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/validate.ts src/lib/network/validate.test.ts
git commit -m "feat: validate maxpool2d input rank and window size"
```

---

### Task 5: The TensorFlow.js layer

**Files:**
- Modify: `src/lib/tf/buildModel.ts`
- Test: `src/lib/tf/buildModel.test.ts`

**Interfaces:**
- Consumes: `MaxPool2dBlock` from Task 1.
- Produces: a `maxpool2d` block builds a `tf.layers.maxPooling2d`.

- [ ] **Step 1: Write the failing test**

In `src/lib/tf/buildModel.test.ts`, add this test to the `buildModel` describe, after `builds a convolutional chain with flattening`:

```ts
  it('builds a pooling chain and halves the spatial dimensions', () => {
    const network: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', units: 10 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('MaxPooling2D');
    expect(model.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(model.layers[0].outputShape).toEqual([null, 14, 14, 1]);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  it('runs a forward pass through a pooling layer', () => {
    const network: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', units: 10 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const model = build(network);
    const prediction = model.predict(tf.ones([1, 28, 28, 1])) as tf.Tensor;
    expect(prediction.shape).toEqual([1, 10]);
    prediction.dispose();
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run --project engine src/lib/tf/buildModel.test.ts`

Expected: FAIL with `Cannot build a layer for block kind maxpool2d`.

- [ ] **Step 3: Add the layer**

In `src/lib/tf/buildModel.ts`, add a case to `layerFor` directly after the `conv2d` case:

```ts
    case 'maxpool2d':
      return tf.layers.maxPooling2d({
        poolSize: block.poolSize,
        strides: block.stride,
        padding: block.padding,
        inputShape
      });
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run --project engine src/lib/tf/buildModel.test.ts`

Expected: PASS.

If `getClassName()` does not return `MaxPooling2D`, report the actual value rather than deleting the assertion — it is the evidence that the block maps to a real pooling layer.

- [ ] **Step 5: Commit**

```bash
git add src/lib/tf/buildModel.ts src/lib/tf/buildModel.test.ts
git commit -m "feat: build a max pooling layer for maxpool2d blocks"
```

---

### Task 6: The CNN palette and the inspector controls

**Files:**
- Modify: `src/lib/examples/cnn/example.ts`
- Modify: `src/lib/components/InspectorPanel.svelte`
- Test: `src/lib/examples/cnn/example.test.ts`
- Test: `src/lib/components/InspectorPanel.test.ts`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `CNN_PALETTE` includes `maxpool2d`; the inspector renders `param-pool-size`, `param-stride`, and `param-padding` for a pooling block.

The convolution and pooling controls differ only in their size field, so they share one snippet rather than duplicating three labels. The existing test ids `param-kernel-size`, `param-stride`, and `param-padding` must keep working for `conv2d`.

- [ ] **Step 1: Write the failing tests**

In `src/lib/examples/cnn/example.test.ts`, replace the `CNN_PALETTE` test:

```ts
describe('CNN_PALETTE', () => {
  it('offers convolution, pooling and flatten, and no pinned blocks', () => {
    expect(CNN_PALETTE).toEqual([
      'conv2d',
      'maxpool2d',
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
```

In `src/lib/components/InspectorPanel.test.ts`, add these tests at the end of the describe:

```ts
  it('offers pool size choices limited to the incoming image', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('maxpool2d', 1);
    store.updateBlock(store.network.blocks[1].id, { poolSize: 3 });
    render(InspectorPanel, { props: { store } });

    const options = Array.from(
      screen.getByTestId('param-pool-size').querySelectorAll('option')
    ).map((option) => option.getAttribute('value'));
    expect(options).toEqual(['1', '2', '3', '4']);
  });

  it('changes the pool size', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('maxpool2d', 1);
    store.select(store.network.blocks[1].id);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-pool-size'), { target: { value: '4' } });
    expect(store.network.blocks[1]).toMatchObject({ poolSize: 4 });
  });

  it('describes what each padding choice does to a pooled image', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    store.addBlock('maxpool2d', 1);
    store.updateBlock(store.network.blocks[1].id, { poolSize: 3 });
    render(InspectorPanel, { props: { store } });

    const labels = Array.from(screen.getByTestId('param-padding').querySelectorAll('option')).map(
      (option) => option.textContent ?? ''
    );
    expect(labels.some((label) => label.includes('14×14'))).toBe(true);
    expect(labels.some((label) => label.includes('13×13'))).toBe(true);
  });
```

A 3×3 pool with stride 2 over 28×28 gives 14 with `same` padding and 13 with `valid`, which is why this test sets the pool size before rendering.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run --project ui src/lib/examples/cnn/example.test.ts src/lib/components/InspectorPanel.test.ts`

Expected: FAIL. The palette test fails on the array, and the inspector tests fail because no `param-pool-size` element exists.

- [ ] **Step 3: Add the palette entry**

In `src/lib/examples/cnn/example.ts`:

```ts
export const CNN_PALETTE: BlockKind[] = [
  'conv2d',
  'maxpool2d',
  'flatten',
  'linear',
  'relu',
  'sigmoid',
  'softmax'
];
```

- [ ] **Step 4: Generalise the inspector's derived state**

In `src/lib/components/InspectorPanel.svelte`, replace the `kernelChoices` and `strideChoices` deriveds with:

```ts
  const spatialSize = $derived(
    block?.kind === 'conv2d' ? block.kernelSize : block?.kind === 'maxpool2d' ? block.poolSize : null
  );
  const spatialStride = $derived(
    block?.kind === 'conv2d' || block?.kind === 'maxpool2d' ? block.stride : null
  );
  const paddingValue = $derived(
    block?.kind === 'conv2d' || block?.kind === 'maxpool2d' ? block.padding : 'valid'
  );
  const sizeChoices = $derived(
    spatialSize === null ? [] : bounds ? bounds.kernelSize : [spatialSize]
  );
  const strideChoices = $derived(
    spatialStride === null ? [] : bounds ? bounds.stride : [spatialStride]
  );
```

Replace `outputSizeFor` with:

```ts
  function outputSizeFor(padding: 'same' | 'valid'): string {
    if (!inShape || inShape.length !== 3 || spatialSize === null || spatialStride === null) {
      return '';
    }
    const [height, width] = inShape;
    const compute = (dimension: number): number =>
      spatialOutputSize(dimension, spatialSize, spatialStride, padding);
    const result = `${compute(height)}×${compute(width)}`;
    if (padding === 'same' && result === `${height}×${width}`) return `stays ${result}`;
    return `becomes ${result}`;
  }
```

- [ ] **Step 5: Share the spatial controls**

In `src/lib/components/InspectorPanel.svelte`, declare this snippet at the top level of the markup, before the `{#if !block}` block:

```svelte
{#snippet spatialControls(
  testId: string,
  label: string,
  key: 'kernelSize' | 'poolSize',
  value: number,
  description: string
)}
  <label>
    <span>{label}</span>
    <select
      data-testid={testId}
      title={description}
      value={String(value)}
      onchange={(event) => patch({ [key]: Number(event.currentTarget.value) } as Partial<Block>)}
    >
      {#each sizeChoices as choice (choice)}
        <option value={String(choice)}>{choice}×{choice}</option>
      {/each}
    </select>
    <small>{description}</small>
  </label>

  <label>
    <span>Stride</span>
    <select
      data-testid="param-stride"
      title={PARAM_DESCRIPTIONS.stride}
      value={String(spatialStride)}
      onchange={(event) => patch({ stride: Number(event.currentTarget.value) } as Partial<Block>)}
    >
      {#each strideChoices as choice (choice)}
        <option value={String(choice)}>{choice}</option>
      {/each}
    </select>
    <small>{PARAM_DESCRIPTIONS.stride}</small>
  </label>

  <label>
    <span>Padding</span>
    <select
      data-testid="param-padding"
      title={PARAM_DESCRIPTIONS.padding}
      value={paddingValue}
      onchange={(event) =>
        patch({ padding: event.currentTarget.value as 'same' | 'valid' } as Partial<Block>)}
    >
      <option value="same">same — {outputSizeFor('same')}</option>
      <option value="valid">valid — {outputSizeFor('valid')}</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.padding}</small>
  </label>
{/snippet}
```

- [ ] **Step 6: Render the controls for both spatial kinds**

In `src/lib/components/InspectorPanel.svelte`, replace the whole existing `{#if block.kind === 'conv2d'} … {/if}` block with the filters field, guarded to convolution only, followed by the shared snippet call:

```svelte
    {#if block.kind === 'conv2d'}
      <label>
        <span>Filters</span>
        <input
          type="number"
          min="1"
          data-testid="param-filters"
          title={PARAM_DESCRIPTIONS.filters}
          value={block.filters}
          onchange={(event) => commitNumber(event, 'filters')}
        />
        <small>{PARAM_DESCRIPTIONS.filters}</small>
      </label>
    {/if}

    {#if block.kind === 'conv2d'}
      {@render spatialControls(
        'param-kernel-size',
        'Kernel size',
        'kernelSize',
        block.kernelSize,
        PARAM_DESCRIPTIONS.kernelSize
      )}
    {:else if block.kind === 'maxpool2d'}
      {@render spatialControls(
        'param-pool-size',
        'Pool size',
        'poolSize',
        block.poolSize,
        PARAM_DESCRIPTIONS.poolSize
      )}
    {/if}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `npx vitest run --project ui src/lib/examples/cnn/example.test.ts src/lib/components/InspectorPanel.test.ts`

Expected: PASS, including the pre-existing convolution tests that use `param-kernel-size` and `param-padding`.

Run: `npm run check`

Expected: `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 8: Commit**

```bash
git add src/lib/examples/cnn/example.ts src/lib/components/InspectorPanel.svelte src/lib/examples/cnn/example.test.ts src/lib/components/InspectorPanel.test.ts
git commit -m "feat: offer maxpool2d in the CNN palette with pool size and stride controls"
```

---

### Task 7: Documentation and full verification

**Files:**
- Modify: `docs/superpowers/specs/2026-09-16-visnet-design.md`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: a master design document whose engine tables match the code, and a green full verification.

- [ ] **Step 1: Update the type listing**

In `docs/superpowers/specs/2026-09-16-visnet-design.md`, in the `types.ts` listing, extend the `BlockKind` union and add the interface:

```ts
export type BlockKind =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'maxpool2d'
  | 'flatten'
  | 'relu'
  | 'sigmoid'
  | 'softmax'
  | 'output';
```

```ts
export interface MaxPool2dBlock extends BlockBase {
  kind: 'maxpool2d';
  poolSize: number;
  stride: number;
  padding: 'same' | 'valid';
}
```

And extend the `Block` union to `InputBlock | LinearBlock | Conv2dBlock | MaxPool2dBlock | FlattenBlock | ActivationBlock | OutputBlock`.

- [ ] **Step 2: Update the Defaults table**

Add a row after the `conv2d` row:

```
| `maxpool2d`                                | `poolSize: 2, stride: 2, padding: 'valid'`              |
```

- [ ] **Step 3: Update the Block explanations table**

Add a row after the `conv2d` row:

```
| `maxpool2d` | "Shrinks an image by keeping the largest value in each small window."                             |
```

And extend the parameter-description sentence below the table to mention `poolSize`: "How large the window is. The largest value in it survives."

- [ ] **Step 4: Update the Shape rules table**

Add a row after the `conv2d` row:

```
| `maxpool2d`                    | `[ceil(H/stride), ceil(W/stride), channels]` for `same`; `[floor((H - poolSize)/stride) + 1, …]` for `valid` |
```

- [ ] **Step 5: Update the Message catalog**

Add two rows to the Errors table, after the `conv2d` output-dimension row:

```
| `maxpool2d` on non-rank-3 input | "Pooling layer needs image data"    | "This Pooling layer receives {shape}. It expects image data shaped [height, width, channels]."            | "Give the Input block a 3D shape such as [28, 28, 1], or remove the Pooling layer." |
| `maxpool2d` output dimension ≤ 0 | "Pool window is larger than the image" | "A {poolSize}×{poolSize} pool with stride {stride} leaves no room to slide over a {H}×{W} image." | "Use a smaller pool or stride, or set padding to 'same'."                          |
```

- [ ] **Step 6: Update the parameter bounds list**

In the inspector section, after the three `conv2d` bullets, add:

```
  - `maxpool2d` `poolSize`: choices from 1 to `min(H, W)`
  - `maxpool2d` `stride`: choices from 1 to `min(H, W)`
  - `maxpool2d` `padding`: `same` and `valid`, each shown with the output size it
    would produce
```

- [ ] **Step 7: Update the TensorFlow.js mapping table**

Add a row after the `conv2d` row:

```
| `maxpool2d`                    | `tf.layers.maxPooling2d({ poolSize, strides, padding })`                            |
```

- [ ] **Step 8: Reformat the tables**

Run: `npm run format`

Expected: the modified markdown tables are re-aligned by Prettier. No other file changes, because the repository is already Prettier-clean.

- [ ] **Step 9: Run the full verification**

Run: `npm run lint`

Expected: no errors.

Run: `npm run check`

Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`

Expected: both projects green, 0 failures.

Run: `npm run build`

Expected: build succeeds, and `build/index.html`, `build/examples/mlp.html`, and `build/examples/cnn.html` all exist.

Run: `git status --short`

Expected: only the design document is modified.

- [ ] **Step 10: Commit**

```bash
git add docs/superpowers/specs/2026-09-16-visnet-design.md
git commit -m "docs: record maxpool2d in the engine reference tables"
```

---

## Manual verification

These need a browser and are not covered by any automated test. Run `npm run dev` and open `/examples/cnn`.

1. Drag a `maxpool2d` block onto the canvas and confirm it lands and shows `[28 × 28 × 8] → [14 × 14 × 8]` and `0 parameters`.
2. Select it and confirm the inspector shows Pool size, Stride, and Padding, with the padding options annotated with the resulting size.
3. Add a pool with a large pool size, then change the Input shape to `[4, 4, 1]` and confirm the pool size is clamped with an announcement rather than left broken. (The Pool size select only offers sizes that fit, so an oversized pool arises from the image shrinking underneath it.)
4. Set padding to `same` and confirm the output size updates to match.
5. Train a network containing the pool block and confirm the loss falls.
6. Reload and confirm the network containing the pool block is restored intact.
7. Drag a `maxpool2d` onto a network whose Input is flat, such as `[2]`, and confirm the "Pooling layer needs image data" error appears.
