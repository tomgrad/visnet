# Reshape Layer and Shape-Based Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `reshape` block and let the Output block declare any output shape, so a network can produce an image.

**Architecture:** Extend the pure domain (new block kind, Output `shape`), update shape inference, validation, the model builder, the codec (with a `units`→`shape` shim), and the inspector's shared shape field.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@tensorflow/tfjs`, Vitest (`engine` in Node, `ui` in jsdom).

## Global Constraints

- `src/lib/network/**` is pure: no Svelte, no TF.js, no DOM.
- `@tensorflow/tfjs` runtime imports confined to the allow-listed modules; other modules may use `import type`.
- No block stores its input dimension; shapes derive from the previous block.
- Every validation problem carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.
- Verify with `npm test`, `npm run check`, `npm run lint`, `npm run build` after every task.

## Design reference

`docs/superpowers/specs/2026-09-17-visnet-reshape-output-shape-design.md`

---

### Task 1: The reshape block

**Files:**
- Modify: `src/lib/network/types.ts`, `src/lib/network/factory.ts`, `src/lib/network/inferShapes.ts`, `src/lib/tf/buildModel.ts`, `src/lib/network/descriptions.ts`, `src/lib/network/problems.ts`, `src/lib/persist/networkCodec.ts`, `src/lib/examples/cnn/example.ts`
- Modify tests: `src/lib/network/types.test.ts`, `src/lib/network/factory.test.ts`, `src/lib/network/inferShapes.test.ts`, `src/lib/network/problems.test.ts`, `src/lib/network/descriptions.test.ts`, `src/lib/persist/networkCodec.test.ts`, `src/lib/tf/buildModel.test.ts`, `src/lib/examples/cnn/example.test.ts`

**Interfaces:**
- Produces: `ReshapeBlock`, `reshape` in `BlockKind`/`BLOCK_KINDS`, `createBlock('reshape')`, reshape shape inference, reshape validation, `tf.layers.reshape` building.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/types.test.ts`, add `'reshape'` after `'flatten'` in the expected `BLOCK_KINDS` list (now twelve kinds) and add `{ id: 'r', kind: 'reshape', shape: [4] }` to the domain-types block list.

In `src/lib/network/factory.test.ts`, the `it.each(BLOCK_KINDS)` test covers `reshape`; add an explicit default assertion inside `applies the documented defaults`:

```ts
    expect(createBlock('reshape')).toMatchObject({ kind: 'reshape', shape: [1] });
```

In `src/lib/network/inferShapes.test.ts`, add:

```ts
describe('inferShapes on a reshape', () => {
  it('adopts the declared shape', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [4, 4, 1] },
        { id: 'r', kind: 'reshape', shape: [16] },
        { id: 'out', kind: 'output', units: 16 }
      ])
    );
    expect(result.perBlock[1].outShape).toEqual([16]);
    expect(result.perBlock[1].paramCount).toBe(0);
  });

  it('returns a null output shape when the input is unknown', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        { id: 'r', kind: 'reshape', shape: [8] },
        { id: 'out', kind: 'output', units: 8 }
      ])
    );
    expect(result.perBlock[2].outShape).toBeNull();
  });
});
```

In `src/lib/network/problems.test.ts`, add a focused test and a catalog entry:

```ts
  it('reports a reshape whose element count does not match', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'r', kind: 'reshape', shape: [10] },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Reshape size does not match');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('r');
    expect(issue?.message).toContain('16');
    expect(issue?.message).toContain('10');
  });
```

Add `'Reshape size does not match'` to `ERROR_RULE_TITLES`, and a `RULE_CASES` entry:

```ts
  {
    title: 'Reshape size does not match',
    severity: 'error',
    network: net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'r', kind: 'reshape', shape: [10] },
      OUTPUT
    ])
  },
```

In `src/lib/network/descriptions.test.ts`, the `it.each(BLOCK_KINDS)` covers `reshape`; no explicit change is needed.

In `src/lib/persist/networkCodec.test.ts`, add:

```ts
  it('rejects a reshape block without a numeric shape', () => {
    const training = createEmptyNetwork().training;
    expect(
      decodeNetwork(JSON.stringify({ blocks: [{ id: 'r', kind: 'reshape' }], training }))
    ).toBeNull();
  });
```

In `src/lib/tf/buildModel.test.ts`, add:

```ts
  it('builds a reshape layer that changes the tensor shape', () => {
    const network: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [4, 4, 1] },
        { id: 'r', kind: 'reshape', shape: [16] },
        { id: 'out', kind: 'output', units: 16 }
      ],
      training: { loss: 'mse', optimizer: 'sgd', learningRate: 0.1, batchSize: 4 },
      positions: {}
    };
    const model = build(network);
    expect(model.layers[0].getClassName()).toBe('Reshape');
    expect(model.outputs[0].shape).toEqual([null, 16]);
  });
```

In `src/lib/examples/cnn/example.test.ts`, add `'reshape'` to the expected `CNN_PALETTE` after `'flatten'`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/network src/lib/persist src/lib/tf src/lib/examples/cnn`
Expected: FAIL on the new reshape assertions and the palette.

- [ ] **Step 3: Implement**

`types.ts`: add `| 'reshape'` after `'flatten'` in `BlockKind`; add `'reshape',` after `'flatten',` in `BLOCK_KINDS`; add:

```ts
export interface ReshapeBlock extends BlockBase {
  kind: 'reshape';
  shape: number[];
}
```

and add `| ReshapeBlock` to the `Block` union after `FlattenBlock`.

`factory.ts`, in `createBlock` after the `flatten` case:

```ts
    case 'reshape':
      return { id: newBlockId(), kind: 'reshape', shape: [1] };
```

`inferShapes.ts`, in `outputShapeFor` after the `flatten` case:

```ts
    case 'reshape':
      return inShape ? [...block.shape] : null;
```

`buildModel.ts`, in `layerFor` after the `flatten` case:

```ts
    case 'reshape':
      return tf.layers.reshape({ targetShape: block.shape, inputShape });
```

`descriptions.ts`: add `reshape: 'Rearranges the numbers into a different shape without changing them.',` after the `flatten` entry.

`problems.ts`: add a `product` helper and, inside the per-block `forEach` after the flatten rule, add:

```ts
    if (block.kind === 'reshape' && info.inShape) {
      const incoming = product(info.inShape);
      const target = product(block.shape);
      if (incoming !== target) {
        problems.push(
          error({
            title: 'Reshape size does not match',
            message: `This Reshape layer receives ${incoming} numbers, but the shape ${shapeText(block.shape)} holds ${target}.`,
            fix: `Change the Reshape shape so it holds ${incoming} numbers, for example [${incoming}].`,
            blockId: block.id
          })
        );
      }
    }
```

with:

```ts
function product(shape: number[]): number {
  return shape.reduce((total, size) => total * size, 1);
}
```

`networkCodec.ts`, in `isBlock`:

```ts
    case 'reshape':
      return Array.isArray(value.shape) && value.shape.every((n) => typeof n === 'number');
```

`examples/cnn/example.ts`: add `'reshape'` to `CNN_PALETTE` after `'flatten'`.

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src
git commit -m "feat: add a reshape layer"
```

---

### Task 2: Shape-based Output block

**Files:**
- Modify: `src/lib/network/types.ts`, `src/lib/network/factory.ts`, `src/lib/network/problems.ts`, `src/lib/persist/networkCodec.ts`, `src/lib/network/descriptions.ts`, `src/lib/examples/cnn/example.ts`
- Modify tests: every file in `src` that builds an Output block (see Step 3), plus `src/lib/network/types.test.ts`, `problems.test.ts`, `networkCodec.test.ts`, `descriptions.test.ts`, `factory.test.ts`, `src/lib/examples/cnn/example.test.ts`

**Interfaces:**
- Consumes: Task 1's reshape.
- Produces: `OutputBlock { kind: 'output'; shape: number[] }`; shape-equality validation; the `units`→`shape` codec shim.

- [ ] **Step 1: Write the failing tests**

In `src/lib/network/types.test.ts`, change the domain-types Output block to `{ id: 'h', kind: 'output', shape: [2] }`.

In `src/lib/network/problems.test.ts`, replace the `'Output must be a list of scores'` and `'Last layer size does not match the Output block'` titles in `ERROR_RULE_TITLES` with `'Last layer shape does not match the Output block'`, and replace their `RULE_CASES` entries with:

```ts
  {
    title: 'Last layer shape does not match the Output block',
    severity: 'error',
    network: net([
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'dense', kind: 'linear', units: 4 },
      { id: 'out', kind: 'output', shape: [2] }
    ])
  },
```

Add `'Cross-entropy needs a flat output'` to `WARNING_RULE_TITLES` and a `RULE_CASES` entry:

```ts
  {
    title: 'Cross-entropy needs a flat output',
    severity: 'warning',
    network: net([
      { id: 'in', kind: 'input', shape: [4, 4, 1] },
      { id: 'pool', kind: 'maxpool2d', poolSize: 2, stride: 2, padding: 'valid' },
      { id: 'out', kind: 'output', shape: [2, 2, 1] }
    ])
  },
```

In `src/lib/persist/networkCodec.test.ts`, add:

```ts
  it('loads a legacy output block that stores units', () => {
    const net = createEmptyNetwork();
    const blocks = net.blocks.map((block) =>
      block.kind === 'output' ? { id: block.id, kind: 'output', units: 2 } : block
    );
    const restored = decodeNetwork(JSON.stringify({ blocks, training: net.training }));
    expect(restored?.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [2] });
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/network src/lib/persist`
Expected: FAIL on the new Output expectations.

- [ ] **Step 3: Implement**

`types.ts`: change `OutputBlock` to:

```ts
export interface OutputBlock extends BlockBase {
  kind: 'output';
  shape: number[];
}
```

`factory.ts`: change the `output` case to `return { id: newBlockId(), kind: 'output', shape: [2] };`.

`descriptions.ts`: change the `output` description to `'Declares the shape the network predicts.'` and add parameter descriptions:

```ts
  outputShape: 'The shape the network should produce, for example 2 numbers or a 28 by 28 image.',
  reshapeShape: 'The new shape for the same numbers.',
```

In `src/lib/network/descriptions.test.ts`, add `'outputShape'` and `'reshapeShape'` to the expected `PARAM_DESCRIPTIONS` key list.

`problems.ts`: add a `sameShape` helper:

```ts
function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((size, index) => size === b[index]);
}
```

Replace the two output rules with:

```ts
  if (outputBlock && outputBlock.kind === 'output' && lastRealIndex >= 0) {
    const lastShape = perBlock[lastRealIndex].outShape;
    if (lastShape && !sameShape(lastShape, outputBlock.shape)) {
      problems.push(
        error({
          title: 'Last layer shape does not match the Output block',
          message: `The last layer produces ${shapeText(lastShape)}, but the Output block expects ${shapeText(outputBlock.shape)}.`,
          fix: `Set the last layer to produce ${shapeText(outputBlock.shape)}, or change the Output block to ${shapeText(lastShape)}.`,
          blockId: net.blocks[lastRealIndex].id
        })
      );
    }
  }
```

Add the cross-entropy rank warning after the softmax warnings:

```ts
  const finalShape = lastRealIndex >= 0 ? perBlock[lastRealIndex].outShape : null;
  if (net.training.loss === 'crossEntropy' && finalShape && finalShape.length !== 1) {
    problems.push(
      warning({
        title: 'Cross-entropy needs a flat output',
        message: `Cross-entropy compares a flat list of scores with the labels, but the last layer produces ${shapeText(finalShape)}.`,
        fix: 'End the network with a Flatten or Linear layer, or switch the loss to mean squared error.'
      })
    );
  }
```

Change the expected-classes warning to:

```ts
  if (
    options.expectedClasses !== undefined &&
    outputBlock &&
    outputBlock.kind === 'output' &&
    outputBlock.shape.length === 1 &&
    outputBlock.shape[0] !== options.expectedClasses
  ) {
    problems.push(
      warning({
        title: 'Output size does not match the data',
        message: `The Output block says ${outputBlock.shape[0]} classes, but the dataset has ${options.expectedClasses}.`,
        fix: `Make the last layer produce ${options.expectedClasses} numbers, and set the Output block to ${options.expectedClasses}.`,
        blockId: outputBlock.id
      })
    );
  }
```

`networkCodec.ts`: split the `linear`/`output` case and add a shim:

```ts
    case 'linear':
      return typeof value.units === 'number';
    case 'output':
      return Array.isArray(value.shape) && value.shape.every((n) => typeof n === 'number');
```

and, in `decodeNetwork`, before validating:

```ts
    const blocks = parsed.blocks.map((block) =>
      isRecord(block) &&
      block.kind === 'output' &&
      block.shape === undefined &&
      typeof block.units === 'number'
        ? { ...block, shape: [block.units] }
        : block
    );
    if (!blocks.every(isBlock)) return null;
```

then use `blocks` for the returned network.

`examples/cnn/example.ts`: change the output block to `shape: [10]`.

**Fixture updates:** update every remaining Output block in `src` from `units` to `shape`:

```bash
grep -rn "kind: 'output'" src
```

Replace each `{ id: …, kind: 'output', units: N }` with `{ id: …, kind: 'output', shape: [N] }`. This touches `probe.test.ts`, `chain.test.ts`, `flow.test.ts`, `placement.test.ts`, `networkStore.svelte.test.ts`, `buildModel.test.ts`, `inferShapes.test.ts`, `problems.test.ts`, `cnn/example.test.ts`, and any others the grep reports.

- [ ] **Step 4: Run the tests**

Run: `npm test && npm run check`
Expected: PASS, 0 errors.

- [ ] **Step 5: Commit**

```bash
git add -A src
git commit -m "feat: let the Output block declare a shape"
```

---

### Task 3: Shared shape field in the inspector

**Files:**
- Modify: `src/lib/components/InspectorPanel.svelte`
- Modify tests: `src/lib/components/InspectorPanel.test.ts`

**Interfaces:**
- Consumes: Task 2's `OutputBlock.shape`; Task 1's `ReshapeBlock.shape`.
- Produces: one shape text field for Input, Output, and Reshape; Linear keeps `units`.

- [ ] **Step 1: Write the failing tests**

In `src/lib/components/InspectorPanel.test.ts`, add:

```ts
  it('edits the output block shape', async () => {
    const store = new NetworkStore();
    const id = store.network.blocks.at(-1)!.id;
    store.select(id);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-shape'), { target: { value: '3' } });
    expect(store.network.blocks.at(-1)).toMatchObject({ kind: 'output', shape: [3] });
  });

  it('edits the reshape shape and shows the incoming count', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [4, 4, 1] });
    store.addBlock('reshape', 1);
    store.select(store.network.blocks[1].id);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-shape'), { target: { value: '16' } });
    expect(store.network.blocks[1]).toMatchObject({ kind: 'reshape', shape: [16] });
    expect(screen.getByTestId('inspector').textContent).toContain('16');
  });

  it('shows no units field for the output block', () => {
    const store = new NetworkStore();
    store.select(store.network.blocks.at(-1)!.id);
    render(InspectorPanel, { props: { store } });
    expect(screen.queryByTestId('param-units')).toBeNull();
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/InspectorPanel.test.ts`
Expected: FAIL — the Output block has a units field and no shape field.

- [ ] **Step 3: Implement**

In `InspectorPanel.svelte`, generalise the shape commit to any block with a `shape`:

```ts
  function commitShape(event: Event): void {
    if (!block || !('shape' in block)) return;
    const input = event.currentTarget as HTMLInputElement;
    const parts = input.value.split(',').map((part) => part.trim());
    const parsed = parts.map((part) => Number(part));
    const valid =
      parts.length > 0 &&
      parts.every((part) => part !== '') &&
      parsed.every((value) => Number.isInteger(value) && value > 0);

    if (!valid) {
      input.value = block.shape.join(', ');
      paramError = 'Enter positive whole numbers, separated by commas.';
      return;
    }
    paramError = null;
    patch({ shape: parsed } as Partial<Block>);
  }
```

Remove the output branch from `currentNumber` and the units condition:

```ts
  function currentNumber(key: 'units' | 'filters' | 'size'): number | null {
    if (!block) return null;
    if (key === 'units' && block.kind === 'linear') return block.units;
    if (key === 'filters' && block.kind === 'conv2d') return block.filters;
    if (key === 'size' && block.kind === 'upsampling2d') return block.size;
    return null;
  }
```

Add a shape-field snippet and render it for the three kinds:

```svelte
{#snippet shapeField(value: number[], label: string, description: string)}
  <label>
    <span>{label}</span>
    <input
      type="text"
      data-testid="param-shape"
      title={description}
      value={value.join(', ')}
      onchange={commitShape}
    />
    <small>{description}</small>
  </label>
{/snippet}
```

Replace the Input-only shape block with:

```svelte
    {#if block.kind === 'input'}
      {@render shapeField(block.shape, 'Shape', PARAM_DESCRIPTIONS.inputShape)}
    {:else if block.kind === 'output'}
      {@render shapeField(block.shape, 'Output shape', PARAM_DESCRIPTIONS.outputShape)}
    {:else if block.kind === 'reshape'}
      {@render shapeField(block.shape, 'New shape', PARAM_DESCRIPTIONS.reshapeShape)}
      <p class="incoming" data-testid="reshape-count">
        The incoming data has {inShape ? inShape.reduce((total, size) => total * size, 1) : '—'}{' '}
        numbers.
      </p>
    {/if}
```

Change the units block condition from `block.kind === 'linear' || block.kind === 'output'` to `block.kind === 'linear'`.

- [ ] **Step 4: Run the tests, check, lint and build**

Run: `npm test && npm run check && npm run lint && npm run build`
Expected: PASS, 0 errors, clean lint, prerender succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A src
git commit -m "feat: share one shape field across input, output and reshape"
```

---

### Task 4: Update the docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the module list**

Add `- reshape` after `- flatten` in `README.md`'s "Available modules" list.

- [ ] **Step 2: Verify**

Run: `npm test && npm run check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: list the reshape layer"
```

---

## Self-review

- **Spec coverage:** Task 1 is spec §4 (reshape), §5, §6 (reshape rule), §7, §10, §11; Task 2 is spec §4 (output), §6 (output + cross-entropy + expected classes), §8, §11; Task 3 is spec §9; Task 4 is spec §11.
- **Placeholder scan:** No "TBD"/"handle edge cases"; code is given for every changed function, and the fixture update is a concrete grep-and-replace.
- **Type consistency:** `ReshapeBlock`/`OutputBlock.shape` defined in Tasks 1–2 and consumed in Task 3. `outputShape`/`reshapeShape` descriptions defined in Task 2 and used in Task 3.
- **Known risk:** Task 2 is broad because the Output type change touches every fixture; the grep step makes the set exhaustive, and `npm test` is the gate.
