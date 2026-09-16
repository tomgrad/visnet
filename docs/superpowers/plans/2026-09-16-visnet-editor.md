# VisNet Editor and MLP Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the VisNet editor UI, the 2D point-classification example, and browser persistence on top of the finished engine, so a beginner can drag blocks, train, and watch a live decision boundary.

**Architecture:** The engine (`src/lib/network/`, `src/lib/editor/flow.ts`, `src/lib/tf/`, `src/lib/training/`, `src/lib/data/`, `src/lib/render/`) is complete and must not be restructured. This plan adds a reactive runes store over it, a Svelte Flow canvas that is a pure projection of the network, inspector and shape-table panels that make the data shape visible at every point in the pipeline, the MLP example page that owns the dataset and the training controls, and a persistence layer. Example pages compose the editor; the editor knows nothing about datasets or training.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, `@xyflow/svelte` 1.6, `@tensorflow/tfjs` 4.22, Vitest 5 with `test.projects` (Node for engine tests, jsdom for component tests), `@testing-library/svelte`, plain CSS with design tokens.

## Global Constraints

- Package manager is **npm**. Do not use pnpm, yarn, or bun.
- TypeScript strict mode is on; `npm run check` must report 0 errors and 0 warnings.
- `src/lib/network/**` stays pure: no Svelte, no `@tensorflow/tfjs`, no DOM.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, and (added by this plan) `src/lib/persist/weights.ts`. No other production module may import it.
- `src/lib/editor/flow.ts` and `src/lib/editor/history.ts` stay free of Svelte, TensorFlow.js, DOM, and `@xyflow/svelte`.
- No backend, no server code. All routes prerendered; the build output is static files.
- **Every TensorFlow.js and Svelte Flow usage must be browser-only**, so `npm run build` (which prerenders with SSR on) succeeds. Guard with `browser` from `$app/environment` and dynamic `import()`; never import TensorFlow.js or Svelte Flow at module scope of a component that is prerendered.
- Use Svelte 5 runes only. Do not use `svelte/store` writable/readable, and do not use `export let`.
- Every validation issue must carry a non-empty `title`, `message`, and `fix` (engine rule, already enforced).
- Beginner-friendliness is a product requirement: every block and parameter has a plain-language description from `src/lib/network/descriptions.ts`, every error shows its fix, and every automatic parameter correction is announced inline.
- Styling is plain CSS with the tokens in `src/lib/styles/tokens.css`. No CSS framework. Do not add new tokens without adding them to `tokens.css` and `tokens.test.ts`.
- Do not add code comments unless a non-obvious constraint requires one.
- Commit at the end of every task with the exact message shown in that task.
- Do not commit anything under `.superpowers/`.
- Reference: `docs/superpowers/specs/2026-09-16-visnet-design.md`, especially sections 2, 7, 8, 9, 14, 15, 16, 17.

## File Structure

| Path | Responsibility |
| --- | --- |
| `vite.config.ts` | Add `test.projects`: a `node` project for engine tests, a `ui` project (jsdom) for component tests |
| `src/lib/editor/placement.ts` | Pure index math: where a palette click or a canvas drop inserts a block |
| `src/lib/network/constraints.ts` | Pure parameter bounds and clamping derived from the incoming shape, with announcement text |
| `src/lib/editor/networkStore.svelte.ts` | Runes store: the network, selection, derived issues/shapes, mutators, history, announcements |
| `src/lib/components/BlockNode.svelte` | Custom Svelte Flow node: kind, key parameter, in→out shape badge, delete |
| `src/lib/components/BlockPalette.svelte` | The blocks this example permits, each with its plain-language description |
| `src/lib/components/BlockCanvas.svelte` | Svelte Flow wrapper: the only component that imports `@xyflow/svelte` |
| `src/lib/components/EditorToolbar.svelte` | Undo/redo, reset, fit view, save/load model |
| `src/lib/components/InspectorPanel.svelte` | Selected block's parameters, constrained and explained, plus incoming shape |
| `src/lib/components/ShapeTable.svelte` | The whole pipeline: position, block, in shape, out shape, parameters |
| `src/lib/components/IssuesPanel.svelte` | Errors and warnings with their fixes |
| `src/lib/components/NetworkEditor.svelte` | The embeddable unit: composes toolbar, palette, canvas, inspector, shape table, issues |
| `src/lib/components/LossChart.svelte` | Hand-rolled SVG loss curve |
| `src/lib/components/StatsReadout.svelte` | Epoch, loss, accuracy with one-line explanations |
| `src/lib/components/TrainingPanel.svelte` | Loss, optimizer, learning rate, batch size, play/pause/step/reset |
| `src/lib/components/DecisionBoundary.svelte` | The 2D canvas: boundary, points, click-to-add |
| `src/lib/components/ExampleLayout.svelte` | Page shell: editor on one side, experiment on the other |
| `src/lib/examples/mlp/datasetStore.svelte.ts` | The MLP dataset state, generators, click-to-add, tensor lifecycle |
| `src/lib/examples/mlp/example.ts` | Pure example configuration: allowed palette, generator labels, defaults |
| `src/lib/persist/storage.ts` | Injectable `Storage` wrapper for the network and the dataset, with versioning |
| `src/lib/persist/weights.ts` | IndexedDB weight save/load through TensorFlow.js IO, plus pure shape comparison |
| `src/routes/+page.svelte` | Landing page |
| `src/routes/examples/mlp/+page.svelte` | The MLP example |
| `src/routes/examples/cnn/+page.svelte` | CNN placeholder |

---

### Task 1: Component-test harness

**Files:**
- Modify: `package.json` (devDependencies)
- Modify: `vite.config.ts`
- Create: `src/lib/components/TestHarness.svelte` (temporary, deleted in this task)
- Test: `src/lib/components/TestHarness.test.ts` (temporary, deleted in this task)

**Interfaces:**
- Consumes: nothing.
- Produces: a Vitest setup in which `src/lib/components/**/*.test.ts` runs in jsdom with Svelte 5 components renderable, and every other `src/**/*.test.ts` keeps running in Node exactly as it does today. Later tasks rely on `render` from `@testing-library/svelte` working.

This task exists because component tests are the only automated verification available for the UI. If the jsdom + SvelteKit + Vitest combination cannot be made to work, **stop and report BLOCKED** with the exact error and what you tried — do not silently skip component testing.

- [ ] **Step 1: Install the test dependencies**

```bash
npm install -D @testing-library/svelte @testing-library/user-event jsdom
```

- [ ] **Step 2: Write the failing test**

Create `src/lib/components/TestHarness.svelte`:

```svelte
<script lang="ts">
  let { label = 'harness' }: { label?: string } = $props();
</script>

<p data-testid="label">{label}</p>
```

Create `src/lib/components/TestHarness.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import TestHarness from './TestHarness.svelte';

describe('component test harness', () => {
  it('renders a Svelte 5 component in jsdom', () => {
    render(TestHarness, { props: { label: 'rendered' } });
    expect(screen.getByTestId('label').textContent).toBe('rendered');
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/TestHarness.test.ts`
Expected: FAIL. The likely failures are a missing jsdom environment, or a Svelte server build being resolved instead of the browser build (`Cannot read properties of undefined`, or a component that renders as an empty string). Record the actual error.

- [ ] **Step 4: Configure the two Vitest projects**

Replace the `test` block in `vite.config.ts` with:

```ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit()],
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'engine',
          environment: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.svelte.test.ts', 'src/lib/components/**/*.test.ts']
        }
      },
      {
        extends: true,
        resolve: { conditions: ['browser'] },
        test: {
          name: 'ui',
          environment: 'jsdom',
          include: ['src/**/*.svelte.test.ts', 'src/lib/components/**/*.test.ts']
        }
      }
    ]
  }
});
```

The `ui` project owns two filename patterns: `*.test.ts` inside `src/lib/components/`, and any `*.svelte.test.ts` anywhere. The second exists so that tests of runes modules (files named `*.svelte.ts`, such as the editor store in Task 4) run under the browser condition set, where runes behave as they do in the app. Engine tests keep running in Node.

- [ ] **Step 5: Iterate until both projects are green**

Run: `npx vitest run src/lib/components/TestHarness.test.ts`
Expected: PASS.

If the ui project still resolves the server build of Svelte, add to the `ui` project:

```ts
        server: { deps: { inline: ['@xyflow/svelte', '@testing-library/svelte'] } },
```

Re-run until green. Then run the full suite:

Run: `npm test`
Expected: the engine project reports the same test files and test count as before this task (16 files, 222 tests at the time of writing), the ui project reports 1 file and 1 test, and there are no failures. If the engine project's count changed, stop and investigate — the engine tests must keep running in Node.

- [ ] **Step 6: Delete the harness files**

```bash
rm src/lib/components/TestHarness.svelte src/lib/components/TestHarness.test.ts
```

- [ ] **Step 7: Verify the toolchain**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm run lint`
Expected: no errors.

Run: `npm test`
Expected: engine project green with its full count, ui project reports `0` test files (the harness is gone), no failures.

Run: `npm run build`
Expected: build succeeds and `build/index.html` exists.

- [ ] **Step 8: Commit**

```bash
git add package.json package-lock.json vite.config.ts
git commit -m "test: add a jsdom project for component tests"
```

---

### Task 2: Block placement index math

**Files:**
- Create: `src/lib/editor/placement.ts`
- Test: `src/lib/editor/placement.test.ts`

**Interfaces:**
- Consumes: `Network` from `../network/types`.
- Produces:

```ts
export function insertionIndexFor(net: Network, selectedBlockId: string | null): number;
export function dropIndexFor(flowX: number, blockCount: number, nodeWidth: number, gap: number): number;
```

- `insertionIndexFor` returns the index a palette click should insert at: one past the selected block, or just before the output block when nothing is selected or the selection is unknown. The result is always in the interior range `[1, blocks.length - 1]`, which is exactly what `insertAt` accepts, so inserting never lands before the input or after the output.
- `dropIndexFor` converts a canvas x coordinate into the same interior range. Blocks sit at `x = i * (nodeWidth + gap)` with their centre at `i * (nodeWidth + gap) + nodeWidth / 2`. The rule is **nearest gap**: the result is `1` when the drop is left of the second block's centre, and increases by one each time the drop passes an interior block's centre, clamped to `[1, blockCount - 1]`. Interior blocks are indices `1` through `blockCount - 2`, so the input and the output are never drop targets.
  - Concretely, with `nodeWidth` 200 and `gap` 80 the step is 280 and the interior centres are at 380, 660, 940, …. So for a five-block chain: `x < 380 → 1`, `380 ≤ x < 660 → 2`, `660 ≤ x < 940 → 3`, `x ≥ 940 → 4`.
  - For a three-block chain `[input, X, output]` there are two interior slots: `x < 380 → 1` and `x ≥ 380 → 2`. A far-right drop must land in the **last** interior slot, not the first.
  - `blockCount` below 3 leaves a single interior slot, so the result is always `1`.
- Both are pure and have no Svelte, DOM, TensorFlow.js, or `@xyflow/svelte` imports.

- [ ] **Step 1: Write the failing test**

`src/lib/editor/placement.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import type { Network } from '../network/types';
import { dropIndexFor, insertionIndexFor } from './placement';

function net(): Network {
  return {
    version: 1,
    blocks: [
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'a', kind: 'linear', units: 8 },
      { id: 'b', kind: 'relu' },
      { id: 'c', kind: 'linear', units: 2 },
      { id: 'out', kind: 'output', units: 2 }
    ],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

describe('insertionIndexFor', () => {
  it('inserts after the selected block', () => {
    expect(insertionIndexFor(net(), 'a')).toBe(2);
    expect(insertionIndexFor(net(), 'b')).toBe(3);
  });

  it('inserts just before the output when the selection is the last interior block', () => {
    expect(insertionIndexFor(net(), 'c')).toBe(4);
  });

  it('inserts just before the output when nothing is selected', () => {
    expect(insertionIndexFor(net(), null)).toBe(4);
  });

  it('falls back to the output slot for an unknown selection', () => {
    expect(insertionIndexFor(net(), 'missing')).toBe(4);
  });

  it('never returns an index outside the interior range', () => {
    const network = net();
    for (const id of ['in', 'a', 'b', 'c', 'out', null]) {
      const index = insertionIndexFor(network, id);
      expect(index).toBeGreaterThanOrEqual(1);
      expect(index).toBeLessThanOrEqual(network.blocks.length - 1);
    }
  });
});

describe('dropIndexFor', () => {
  const width = 200;
  const gap = 80;

  it('drops before the first interior block when left of it', () => {
    expect(dropIndexFor(0, 5, width, gap)).toBe(1);
    expect(dropIndexFor(200, 5, width, gap)).toBe(1);
  });

  it('advances one slot per interior block centre passed', () => {
    expect(dropIndexFor(379, 5, width, gap)).toBe(1);
    expect(dropIndexFor(380, 5, width, gap)).toBe(2);
    expect(dropIndexFor(659, 5, width, gap)).toBe(2);
    expect(dropIndexFor(660, 5, width, gap)).toBe(3);
    expect(dropIndexFor(940, 5, width, gap)).toBe(4);
  });

  it('clamps to the last interior slot when dropped past the end', () => {
    expect(dropIndexFor(5000, 5, width, gap)).toBe(4);
  });

  it('uses both interior slots for a three-block network', () => {
    expect(dropIndexFor(0, 3, width, gap)).toBe(1);
    expect(dropIndexFor(9999, 3, width, gap)).toBe(2);
  });

  it('stays inside the interior range for any coordinate', () => {
    for (let x = -500; x <= 3000; x += 37) {
      const index = dropIndexFor(x, 6, width, gap);
      expect(index).toBeGreaterThanOrEqual(1);
      expect(index).toBeLessThanOrEqual(5);
    }
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/editor/placement.test.ts`
Expected: FAIL — `Failed to resolve import "./placement"`.

- [ ] **Step 3: Write the implementation**

`src/lib/editor/placement.ts`:

```ts
import type { Network } from '../network/types';

const FIRST_INTERIOR = 1;

export function insertionIndexFor(net: Network, selectedBlockId: string | null): number {
  const last = net.blocks.length - 1;
  const index = net.blocks.findIndex((block) => block.id === selectedBlockId);
  if (index === -1) return last;
  return Math.min(index + 1, last);
}

export function dropIndexFor(
  flowX: number,
  blockCount: number,
  nodeWidth: number,
  gap: number
): number {
  const last = blockCount - 1;
  if (last < FIRST_INTERIOR) return FIRST_INTERIOR;

  const step = nodeWidth + gap;
  let index = FIRST_INTERIOR;
  for (let i = FIRST_INTERIOR; i < last; i++) {
    const centre = i * step + nodeWidth / 2;
    if (flowX >= centre) index = i + 1;
  }
  return Math.min(Math.max(index, FIRST_INTERIOR), last);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/editor/placement.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/editor/placement.ts src/lib/editor/placement.test.ts
git commit -m "feat: add block placement index math"
```

---

### Task 3: Parameter bounds and clamping

**Files:**
- Create: `src/lib/network/constraints.ts`
- Test: `src/lib/network/constraints.test.ts`

**Interfaces:**
- Consumes: `Network`, `Block` from `./types`; `inferShapes` from `./inferShapes`.
- Produces:

```ts
export interface ParameterBounds {
  kernelSize: number[];
  stride: number[];
}

export function parameterBounds(inShape: number[] | null): ParameterBounds | null;

export interface ClampResult {
  patch: Partial<Block>;
  announcement: string | null;
}

export function clampBlockPatch(net: Network, id: string, patch: Partial<Block>): ClampResult;
```

- `parameterBounds` returns the valid choices for a convolution's kernel size and stride given the incoming shape. It returns `null` unless the incoming shape is rank 3, because only image data has spatial dimensions to slide over. Otherwise both lists are `[1 .. min(height, width)]`.
- `clampBlockPatch` enforces the app's rule that no automatic correction is ever silent. It:
  - clamps `units` and `filters` to integers of at least 1, announcing `Units changed from {from} to {to}. A layer must produce at least one number.` or the same sentence with `Filters`;
  - clamps `kernelSize` and `stride` into `parameterBounds(inShape)` when bounds exist, announcing `Kernel size changed from {from} to {to} because the incoming data is {height}×{width}.` or the same with `Stride`;
  - returns the patch untouched with `announcement: null` when the block id is unknown, and when nothing needed correcting;
  - when the incoming shape is unknown or is not rank 3, still clamps `units` and `filters` — they do not depend on the shape — but leaves `kernelSize` and `stride` alone, because there are no spatial dimensions to bound them;
  - when several parameters in one patch are corrected, the **first** correction's announcement is the one returned.
- The announcement uses `×` (U+00D7) between the dimensions, matching `shapeLabel`'s style.
- This module is pure and lives under `src/lib/network/`, so it must not import Svelte, TensorFlow.js, or DOM.

- [ ] **Step 1: Write the failing test**

`src/lib/network/constraints.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { clampBlockPatch, parameterBounds } from './constraints';
import type { Network } from './types';

function net(blocks: Network['blocks']): Network {
  return {
    version: 1,
    blocks,
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

const IMAGE_NETWORK = net([
  { id: 'in', kind: 'input', shape: [28, 28, 1] },
  { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
  { id: 'flat', kind: 'flatten' },
  { id: 'dense', kind: 'linear', units: 10 },
  { id: 'out', kind: 'output', units: 10 }
]);

const FLAT_NETWORK = net([
  { id: 'in', kind: 'input', shape: [2] },
  { id: 'dense', kind: 'linear', units: 8 },
  { id: 'out', kind: 'output', units: 2 }
]);

describe('parameterBounds', () => {
  it('returns no bounds for flat input', () => {
    expect(parameterBounds([2])).toBeNull();
    expect(parameterBounds(null)).toBeNull();
    expect(parameterBounds([4, 4])).toBeNull();
  });

  it('offers kernel and stride choices up to the smaller image dimension', () => {
    expect(parameterBounds([28, 28, 1])).toEqual({
      kernelSize: Array.from({ length: 28 }, (_, i) => i + 1),
      stride: Array.from({ length: 28 }, (_, i) => i + 1)
    });
    expect(parameterBounds([6, 4, 3])?.kernelSize).toEqual([1, 2, 3, 4]);
  });

  it('returns no bounds when a spatial dimension is not positive', () => {
    expect(parameterBounds([0, 4, 1])).toBeNull();
  });
});

describe('clampBlockPatch', () => {
  it('leaves a valid patch untouched and says nothing', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'dense', { units: 16 })).toEqual({
      patch: { units: 16 },
      announcement: null
    });
  });

  it('raises units to at least one and announces it', () => {
    const result = clampBlockPatch(FLAT_NETWORK, 'dense', { units: 0 });
    expect(result.patch).toEqual({ units: 1 });
    expect(result.announcement).toBe(
      'Units changed from 0 to 1. A layer must produce at least one number.'
    );
  });

  it('rounds units down to an integer and announces it', () => {
    const result = clampBlockPatch(FLAT_NETWORK, 'dense', { units: 7.6 });
    expect(result.patch).toEqual({ units: 7 });
    expect(result.announcement).toBe(
      'Units changed from 7.6 to 7. A layer must produce at least one number.'
    );
  });

  it('clamps filters and announces it with the filters wording', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { filters: 0 });
    expect(result.patch).toEqual({ filters: 1 });
    expect(result.announcement).toBe(
      'Filters changed from 0 to 1. A layer must produce at least one number.'
    );
  });

  it('clamps a kernel larger than the image and names the dimensions', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { kernelSize: 40 });
    expect(result.patch).toEqual({ kernelSize: 28 });
    expect(result.announcement).toBe(
      'Kernel size changed from 40 to 28 because the incoming data is 28×28.'
    );
  });

  it('clamps a stride larger than the image', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { stride: 99 });
    expect(result.patch).toEqual({ stride: 28 });
    expect(result.announcement).toBe(
      'Stride changed from 99 to 28 because the incoming data is 28×28.'
    );
  });

  it('leaves a kernel alone when the incoming data is not an image', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'dense', { units: 4 }).announcement).toBeNull();
  });

  it('says nothing for an unknown block', () => {
    expect(clampBlockPatch(FLAT_NETWORK, 'missing', { units: 0 })).toEqual({
      patch: { units: 0 },
      announcement: null
    });
  });

  it('clamps several parameters in one patch and reports the first correction', () => {
    const result = clampBlockPatch(IMAGE_NETWORK, 'conv', { kernelSize: 40, stride: 50 });
    expect(result.patch).toEqual({ kernelSize: 28, stride: 28 });
    expect(result.announcement).toBe(
      'Kernel size changed from 40 to 28 because the incoming data is 28×28.'
    );
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/constraints.test.ts`
Expected: FAIL — `Failed to resolve import "./constraints"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/constraints.ts`:

```ts
import { inferShapes } from './inferShapes';
import type { Block, Network } from './types';

export interface ParameterBounds {
  kernelSize: number[];
  stride: number[];
}

export interface ClampResult {
  patch: Partial<Block>;
  announcement: string | null;
}

const MINIMUM_COUNT = 1;

export function parameterBounds(inShape: number[] | null): ParameterBounds | null {
  if (!inShape || inShape.length !== 3) return null;
  const limit = Math.min(inShape[0], inShape[1]);
  if (!Number.isFinite(limit) || limit < MINIMUM_COUNT) return null;
  const choices = Array.from({ length: Math.floor(limit) }, (_, index) => index + MINIMUM_COUNT);
  return { kernelSize: choices, stride: choices };
}

function clampCount(
  value: number,
  label: string
): { value: number; announcement: string | null } {
  const rounded = Number.isFinite(value) ? Math.max(MINIMUM_COUNT, Math.floor(value)) : MINIMUM_COUNT;
  if (rounded === value) return { value, announcement: null };
  return {
    value: rounded,
    announcement: `${label} changed from ${value} to ${rounded}. A layer must produce at least one number.`
  };
}

function clampDimension(
  value: number,
  label: string,
  limit: number,
  inShape: number[]
): { value: number; announcement: string | null } {
  const rounded = Number.isFinite(value)
    ? Math.min(Math.max(MINIMUM_COUNT, Math.floor(value)), limit)
    : MINIMUM_COUNT;
  if (rounded === value) return { value, announcement: null };
  const [height, width] = inShape;
  return {
    value: rounded,
    announcement: `${label} changed from ${value} to ${rounded} because the incoming data is ${height}×${width}.`
  };
}

export function clampBlockPatch(net: Network, id: string, patch: Partial<Block>): ClampResult {
  const index = net.blocks.findIndex((block) => block.id === id);
  if (index === -1) return { patch, announcement: null };

  const inShape = inferShapes(net).perBlock[index].inShape;
  const bounds = parameterBounds(inShape);
  const result: Partial<Block> = { ...patch };
  let announcement: string | null = null;

  const record = (next: { value: number; announcement: string | null }, key: string): void => {
    (result as Record<string, unknown>)[key] = next.value;
    if (announcement === null) announcement = next.announcement;
  };

  if (typeof patch.units === 'number') {
    record(clampCount(patch.units, 'Units'), 'units');
  }

  if (typeof patch.filters === 'number') {
    record(clampCount(patch.filters, 'Filters'), 'filters');
  }

  if (bounds && inShape) {
    const limit = Math.max(...bounds.kernelSize);
    if (typeof patch.kernelSize === 'number') {
      record(clampDimension(patch.kernelSize, 'Kernel size', limit, inShape), 'kernelSize');
    }
    if (typeof patch.stride === 'number') {
      record(clampDimension(patch.stride, 'Stride', limit, inShape), 'stride');
    }
  }

  return { patch: result, announcement };
}
```

Note on TypeScript: `patch` is a `Partial<Block>`, a union of partials, so assigning `(result as Record<string, unknown>)[key]` is the narrowest way to write a dynamic key without `any`. If `svelte-check` rejects the cast, use a `switch` on the four parameter names instead — it is longer but fully typed, and the tests will not change.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/constraints.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/constraints.ts src/lib/network/constraints.test.ts
git commit -m "feat: add parameter bounds and announced clamping"
```

---

### Task 4: Editor store

**Files:**
- Create: `src/lib/editor/networkStore.svelte.ts`
- Test: `src/lib/editor/networkStore.svelte.test.ts`

**Interfaces:**
- Consumes: `createBlock`, `createEmptyNetwork` from `../network/factory`; `insertAt`, `moveBlock`, `removeBlock`, `replaceBlock` from `../network/chain`; `clampBlockPatch` from `../network/constraints`; `inferShapes` from `../network/inferShapes`; `validate` from `../network/validate`; `History` from `./history`; `insertionIndexFor` from `./placement`; `Block`, `BlockKind`, `Network`, `TrainingConfig` from `../network/types`.
- Produces: `class NetworkStore` with
  - state: `network`, `selectedBlockId`, `expectedClasses`, `announcements`, `canUndo`, `canRedo`
  - derived: `issues`, `errors`, `warnings`, `shapes`, `isValid`, `paramCount`, `selectedBlock`
  - methods: `select(id)`, `addBlock(kind, index?)`, `removeBlock(id)`, `moveBlock(from, to)`, `moveSelectedBy(offset)`, `updateBlock(id, patch)`, `updateTraining(patch)`, `undo()`, `redo()`, `load(net)`, `reset()`, `announce(message)`, `dismissAnnouncements()`

**Behaviour that matters:**
- Every mutation goes through one private `#commit` that pushes the previous network onto the history stack and then replaces it. A mutation that returns the same `Network` reference (a refused delete, a no-op move) records nothing, which is why the engine's refusal paths return the same reference.
- `addBlock` selects the newly created block, so the inspector immediately shows it.
- `updateBlock` runs the patch through `clampBlockPatch` and pushes any announcement into `announcements`, so an automatic correction is never silent.
- `updateTraining` changes only the training config, so the model can be recompiled without rebuilding it and losing weights.
- `load` clears history and selection; it is what the persistence layer calls on startup.
- `expectedClasses` is settable by the embedding page and drives the output-units warning.
- The store holds no rules of its own: everything delegates to the pure modules.

- [ ] **Step 1: Write the failing test**

`src/lib/editor/networkStore.svelte.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { NetworkStore } from './networkStore.svelte';

function store(): NetworkStore {
  const instance = new NetworkStore();
  instance.expectedClasses = 2;
  return instance;
}

const ids = (instance: NetworkStore) => instance.network.blocks.map((block) => block.id);
const kinds = (instance: NetworkStore) => instance.network.blocks.map((block) => block.kind);

describe('selection', () => {
  it('selects and clears', () => {
    const instance = store();
    instance.select('x');
    expect(instance.selectedBlockId).toBe('x');
    instance.select(null);
    expect(instance.selectedBlockId).toBeNull();
  });
});

describe('addBlock', () => {
  it('inserts after the selection and selects the new block', () => {
    const instance = store();
    const anchor = instance.network.blocks[1].id;
    instance.select(anchor);
    const created = instance.addBlock('sigmoid');
    expect(kinds(instance)).toEqual([
      'input',
      'linear',
      'sigmoid',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
    expect(instance.network.blocks[2].id).toBe(created);
    expect(instance.selectedBlockId).toBe(created);
  });

  it('inserts before the output when nothing is selected', () => {
    const instance = store();
    instance.addBlock('flatten');
    expect(kinds(instance)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'flatten',
      'output'
    ]);
  });

  it('honours an explicit index', () => {
    const instance = store();
    instance.addBlock('sigmoid', 1);
    expect(kinds(instance)[1]).toBe('sigmoid');
  });
});

describe('removeBlock', () => {
  it('removes an interior block', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.removeBlock(target);
    expect(ids(instance)).not.toContain(target);
  });

  it('refuses to remove the input or output and records no history entry', () => {
    const instance = store();
    instance.removeBlock(instance.network.blocks[0].id);
    instance.removeBlock(instance.network.blocks[instance.network.blocks.length - 1].id);
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
    expect(instance.canUndo).toBe(false);
  });

  it('clears the selection when the selected block is removed', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.select(target);
    instance.removeBlock(target);
    expect(instance.selectedBlockId).toBeNull();
  });
});

describe('moveBlock', () => {
  it('reorders the chain', () => {
    const instance = store();
    const moving = instance.network.blocks[1].id;
    instance.moveBlock(1, 3);
    expect(ids(instance)[3]).toBe(moving);
  });

  it('records nothing for a no-op move', () => {
    const instance = store();
    instance.moveBlock(1, 1);
    expect(instance.canUndo).toBe(false);
  });

  it('moves the selection by an offset and ignores the ends', () => {
    const instance = store();
    const first = instance.network.blocks[1].id;
    instance.select(first);
    instance.moveSelectedBy(-1);
    expect(ids(instance)[1]).toBe(first);
    expect(instance.canUndo).toBe(false);

    instance.moveSelectedBy(1);
    expect(ids(instance)[2]).toBe(first);
    expect(instance.canUndo).toBe(true);
  });
});

describe('updateBlock', () => {
  it('applies a valid patch with no announcement', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.updateBlock(target, { units: 16 });
    expect(instance.network.blocks[1]).toMatchObject({ units: 16 });
    expect(instance.announcements).toEqual([]);
  });

  it('announces an automatic correction', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.updateBlock(target, { units: 0 });
    expect(instance.network.blocks[1]).toMatchObject({ units: 1 });
    expect(instance.announcements).toEqual([
      'Units changed from 0 to 1. A layer must produce at least one number.'
    ]);
  });

  it('dismisses announcements', () => {
    const instance = store();
    instance.announce('something');
    instance.dismissAnnouncements();
    expect(instance.announcements).toEqual([]);
  });
});

describe('updateTraining', () => {
  it('changes only the training configuration', () => {
    const instance = store();
    const before = ids(instance);
    instance.updateTraining({ optimizer: 'sgd', learningRate: 0.5 });
    expect(instance.network.training).toMatchObject({ optimizer: 'sgd', learningRate: 0.5 });
    expect(ids(instance)).toEqual(before);
  });
});

describe('history', () => {
  it('undoes and redoes a change and reports availability', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;

    expect(instance.canUndo).toBe(false);
    instance.updateBlock(target, { units: 32 });
    expect(instance.canUndo).toBe(true);

    instance.undo();
    expect(instance.network.blocks[1]).toMatchObject({ units: 8 });
    expect(instance.canRedo).toBe(true);

    instance.redo();
    expect(instance.network.blocks[1]).toMatchObject({ units: 32 });
  });

  it('does nothing when there is nothing to undo', () => {
    const instance = store();
    instance.undo();
    instance.redo();
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
  });

  it('clears history and selection on load', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[1].id, { units: 32 });
    instance.select(instance.network.blocks[1].id);
    instance.load(createEmptyNetwork());
    expect(instance.canUndo).toBe(false);
    expect(instance.canRedo).toBe(false);
    expect(instance.selectedBlockId).toBeNull();
    expect(instance.announcements).toEqual([]);
  });

  it('resets to a fresh default network', () => {
    const instance = store();
    instance.removeBlock(instance.network.blocks[1].id);
    instance.reset();
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
    expect(instance.canUndo).toBe(false);
  });
});

describe('derived state', () => {
  it('reflects the network', () => {
    const instance = store();
    expect(instance.isValid).toBe(true);
    expect(instance.errors).toEqual([]);
    expect(instance.warnings).toEqual([]);
    expect(instance.paramCount).toBe(42);

    instance.removeBlock(instance.network.blocks[4].id);
    expect(instance.isValid).toBe(true);
    expect(instance.warnings.map((issue) => issue.title)).toContain(
      'Add a Softmax for probabilities'
    );
  });

  it('exposes the selected block', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    expect(instance.selectedBlock).toBeNull();
    instance.select(target);
    expect(instance.selectedBlock).toMatchObject({ id: target, kind: 'linear' });
  });

  it('reports errors that block training', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[0].id, { shape: [28, 28, 1] });
    expect(instance.isValid).toBe(false);
    expect(instance.errors.map((issue) => issue.title)).toContain(
      'Linear layer needs a flat list'
    );
  });
});
```

If Svelte warns about reading a `$derived` outside a reactive context, wrap the `derived state` assertions in `$effect.root(() => { ... })` imported from `svelte`, or assert through `validate`/`inferShapes` directly, and record which you did in your report. Test output must be free of warnings.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/editor/networkStore.svelte.test.ts`
Expected: FAIL — `Failed to resolve import "./networkStore.svelte"`.

- [ ] **Step 3: Write the implementation**

`src/lib/editor/networkStore.svelte.ts`:

```ts
import { insertAt, moveBlock, removeBlock, replaceBlock } from '../network/chain';
import { clampBlockPatch } from '../network/constraints';
import { createBlock, createEmptyNetwork } from '../network/factory';
import { inferShapes } from '../network/inferShapes';
import type { Block, BlockKind, Network, TrainingConfig } from '../network/types';
import { validate } from '../network/validate';
import { History } from './history';
import { insertionIndexFor } from './placement';

const HISTORY_LIMIT = 50;

export class NetworkStore {
  network = $state<Network>(createEmptyNetwork());
  selectedBlockId = $state<string | null>(null);
  expectedClasses = $state<number | undefined>(undefined);
  announcements = $state<string[]>([]);
  canUndo = $state(false);
  canRedo = $state(false);

  #history = new History<Network>(HISTORY_LIMIT);

  issues = $derived(validate(this.network, { expectedClasses: this.expectedClasses }));
  errors = $derived(this.issues.filter((issue) => issue.severity === 'error'));
  warnings = $derived(this.issues.filter((issue) => issue.severity === 'warning'));
  shapes = $derived(inferShapes(this.network));
  isValid = $derived(this.errors.length === 0);
  paramCount = $derived(this.shapes.totalParamCount);
  selectedBlock = $derived(
    this.network.blocks.find((block) => block.id === this.selectedBlockId) ?? null
  );

  select(id: string | null): void {
    this.selectedBlockId = id;
  }

  addBlock(kind: BlockKind, index?: number): string {
    const block = createBlock(kind);
    const target = index ?? insertionIndexFor(this.network, this.selectedBlockId);
    this.#commit(insertAt(this.network, target, block));
    this.selectedBlockId = block.id;
    return block.id;
  }

  removeBlock(id: string): void {
    const next = removeBlock(this.network, id);
    if (next === this.network) return;
    this.#commit(next);
    if (this.selectedBlockId === id) this.selectedBlockId = null;
  }

  moveBlock(from: number, to: number): void {
    this.#commit(moveBlock(this.network, from, to));
  }

  moveSelectedBy(offset: number): void {
    const index = this.network.blocks.findIndex((block) => block.id === this.selectedBlockId);
    if (index === -1) return;
    this.moveBlock(index, index + offset);
  }

  updateBlock(id: string, patch: Partial<Block>): void {
    const { patch: clamped, announcement } = clampBlockPatch(this.network, id, patch);
    this.#commit(replaceBlock(this.network, id, clamped));
    if (announcement) this.announce(announcement);
  }

  updateTraining(patch: Partial<TrainingConfig>): void {
    this.#commit({ ...this.network, training: { ...this.network.training, ...patch } });
  }

  undo(): void {
    const previous = this.#history.undo(this.network);
    if (previous === null) return;
    this.network = previous;
    this.#syncHistoryFlags();
  }

  redo(): void {
    const next = this.#history.redo(this.network);
    if (next === null) return;
    this.network = next;
    this.#syncHistoryFlags();
  }

  load(net: Network): void {
    this.#history.clear();
    this.network = net;
    this.selectedBlockId = null;
    this.announcements = [];
    this.#syncHistoryFlags();
  }

  reset(): void {
    this.load(createEmptyNetwork());
  }

  announce(message: string): void {
    this.announcements = [...this.announcements, message];
  }

  dismissAnnouncements(): void {
    this.announcements = [];
  }

  #commit(next: Network): void {
    if (next === this.network) return;
    this.#history.push(this.network);
    this.network = next;
    this.#syncHistoryFlags();
  }

  #syncHistoryFlags(): void {
    this.canUndo = this.#history.canUndo;
    this.canRedo = this.#history.canRedo;
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/editor/networkStore.svelte.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/editor/networkStore.svelte.ts src/lib/editor/networkStore.svelte.test.ts
git commit -m "feat: add the reactive editor store"
```

---

### Task 5: Block node and palette

**Files:**
- Create: `src/lib/components/BlockNode.svelte`
- Create: `src/lib/components/BlockPalette.svelte`
- Test: `src/lib/components/BlockPalette.test.ts`

**Interfaces:**
- Consumes: `BLOCK_DESCRIPTIONS` from `../network/descriptions`; `shapeLabel`, `FlowNode` from `../editor/flow`; `BlockKind` from `../network/types`; `Handle`, `Position`, `NodeProps` from `@xyflow/svelte`.
- Produces:
  - `BlockNode.svelte` — a Svelte Flow custom node. Its `data` is `FlowNode['data']` plus `onremove: () => void`, which `BlockCanvas` injects. It renders the block kind, its `in → out` shape badge, its parameter count, a delete button when `removable`, a `title` tooltip containing the block's plain-language description, a target handle on the left unless the block is the input, and a source handle on the right unless the block is the output.
  - `BlockPalette.svelte` — props `{ palette: BlockKind[]; onadd: (kind: BlockKind) => void; ondragstart?: (kind: BlockKind, event: DragEvent) => void }`. It renders one button per kind showing the kind name and its description, calls `onadd` on click, and calls `ondragstart` when a drag begins. Buttons carry `data-testid={`palette-${kind}`}`.

This is the only pair of components in the plan that imports `@xyflow/svelte` besides `BlockCanvas`. `BlockNode` does not render its own children or use the store; all actions arrive through `data` callbacks, which is the idiomatic Svelte Flow pattern and keeps the node a pure function of its props.

- [ ] **Step 1: Write the failing test**

`src/lib/components/BlockPalette.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
import type { BlockKind } from '../network/types';
import BlockPalette from './BlockPalette.svelte';

const PALETTE: BlockKind[] = ['linear', 'relu', 'sigmoid', 'softmax'];

describe('BlockPalette', () => {
  it('renders one button per permitted block', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`)).toBeTruthy();
    }
    expect(screen.queryByTestId('palette-conv2d')).toBeNull();
  });

  it('shows the plain-language description for each block', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`).textContent).toContain(
        BLOCK_DESCRIPTIONS[kind]
      );
    }
  });

  it('reports the chosen kind on click', async () => {
    const onadd = vi.fn();
    render(BlockPalette, { props: { palette: PALETTE, onadd } });
    await userEvent.click(screen.getByTestId('palette-relu'));
    expect(onadd).toHaveBeenCalledWith('relu');
  });

  it('makes every entry draggable', () => {
    render(BlockPalette, { props: { palette: PALETTE, onadd: () => {} } });
    for (const kind of PALETTE) {
      expect(screen.getByTestId(`palette-${kind}`).getAttribute('draggable')).toBe('true');
    }
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/BlockPalette.test.ts`
Expected: FAIL — `Failed to resolve import "./BlockPalette.svelte"`.

- [ ] **Step 3: Write `BlockPalette.svelte`**

```svelte
<script lang="ts">
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockKind } from '../network/types';

  let {
    palette,
    onadd,
    ondragstart
  }: {
    palette: BlockKind[];
    onadd: (kind: BlockKind) => void;
    ondragstart?: (kind: BlockKind, event: DragEvent) => void;
  } = $props();
</script>

<div class="palette">
  <h2>Blocks</h2>
  <ul>
    {#each palette as kind (kind)}
      <li>
        <button
          type="button"
          draggable="true"
          data-testid={`palette-${kind}`}
          title={BLOCK_DESCRIPTIONS[kind]}
          onclick={() => onadd(kind)}
          ondragstart={(event) => ondragstart?.(kind, event)}
        >
          <span class="kind">{kind}</span>
          <span class="description">{BLOCK_DESCRIPTIONS[kind]}</span>
        </button>
      </li>
    {/each}
  </ul>
</div>

<style>
  .palette h2 {
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
    margin: 0 0 var(--space-2);
  }

  .palette ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }

  .palette button {
    width: 100%;
    text-align: left;
    display: grid;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: grab;
  }

  .palette button:hover {
    border-color: var(--color-accent);
  }

  .kind {
    font-weight: 600;
  }

  .description {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
```

- [ ] **Step 4: Write `BlockNode.svelte`**

```svelte
<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { shapeLabel } from '../editor/flow';
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockKind } from '../network/types';

  interface NodeData {
    kind: BlockKind;
    inShape: number[] | null;
    outShape: number[] | null;
    paramCount: number | null;
    index: number;
    removable: boolean;
    onremove?: () => void;
  }

  let { data, selected }: NodeProps = $props();
  const info = $derived(data as unknown as NodeData);
  const inLabel = $derived(shapeLabel(info.inShape) ?? '—');
  const outLabel = $derived(shapeLabel(info.outShape) ?? '—');
  const tooltip = $derived(
    `${BLOCK_DESCRIPTIONS[info.kind]} Input ${inLabel}, output ${outLabel}, ${info.paramCount ?? 0} trainable numbers.`
  );
</script>

<div class="block" class:selected data-testid="block-node" title={tooltip}>
  {#if info.kind !== 'input'}
    <Handle type="target" position={Position.Left} />
  {/if}

  <header>
    <span class="kind">{info.kind}</span>
    {#if info.removable}
      <button
        type="button"
        class="remove"
        aria-label={`Remove ${info.kind}`}
        data-testid="block-remove"
        onclick={() => info.onremove?.()}
      >
        ×
      </button>
    {/if}
  </header>

  <p class="shapes">{inLabel} → {outLabel}</p>
  <p class="params">{info.paramCount ?? 0} parameters</p>

  {#if info.kind !== 'output'}
    <Handle type="source" position={Position.Right} />
  {/if}
</div>

<style>
  .block {
    width: 160px;
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .block.selected {
    border-color: var(--color-accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--color-accent) 30%, transparent);
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }

  .kind {
    font-weight: 600;
  }

  .remove {
    border: none;
    background: none;
    color: var(--color-text-muted);
    cursor: pointer;
    font-size: var(--text-base);
    line-height: 1;
    padding: 0 var(--space-1);
  }

  .remove:hover {
    color: var(--color-error);
  }

  .shapes,
  .params {
    margin: var(--space-1) 0 0;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run src/lib/components/BlockPalette.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/components/BlockPalette.svelte src/lib/components/BlockPalette.test.ts src/lib/components/BlockNode.svelte
git commit -m "feat: add the block palette and canvas node"
```

---

### Task 6: Canvas

**Files:**
- Create: `src/lib/components/BlockCanvas.svelte`

**Interfaces:**
- Consumes: `SvelteFlow`, `Background`, `Controls`, `useSvelteFlow`, `MarkerType`, types `Node`, `Edge`, `NodeTypes` from `@xyflow/svelte`; `NODE_WIDTH`, `NODE_GAP`, `toFlow`, `connectionToIntent` from `../editor/flow`; `dropIndexFor` from `../editor/placement`; `NetworkStore` from `../editor/networkStore.svelte`; `BlockNode.svelte`; `browser` from `$app/environment`.
- Produces: `BlockCanvas.svelte` with props `{ store: NetworkStore; palette: BlockKind[]; ondragover: (kind: BlockKind | null) => void }`. It renders the network as a left-to-right pipeline of `BlockNode`s with shape labels on every wire, and turns the two user gestures into store calls:
  - drawing a wire from one block's source handle to another block's target handle runs `connectionToIntent` and applies the result with `store.moveBlock`;
  - dropping a palette block on the canvas computes a flow x coordinate, converts it to an insertion index with `dropIndexFor`, and calls `store.addBlock(kind, index)`.

**This is the highest-risk component in the plan.** It is the only place that touches Svelte Flow's reactive node/edge contract and its coordinate conversion, and neither can be exercised in jsdom. Follow these rules:

- Nodes and edges are **derived from the store on every render** with `$derived`. Never keep a second copy of node or edge state, and never mutate the arrays. If Svelte Flow mutates the arrays it is given, pass a fresh copy (`[...nodes]`).
- Node positions come from `toFlow`, which computes them from the chain index. Do not enable node dragging (`nodesDraggable={false}`); reordering happens through the wire gesture and the inspector's move buttons.
- Register `BlockNode` once in a `nodeTypes` object and give every node `type: 'block'`.
- Inject the delete callback into each node's data: `data: { ...node.data, onremove: () => store.removeBlock(node.id) }`.
- Svelte Flow needs a sized container. Give the wrapper an explicit height (`height: 100%` inside a container the page sizes, with a `min-height`).
- Import the library stylesheet: `import '@xyflow/svelte/dist/style.css';`.
- Guard the whole render with `{#if browser}` so prerendering never renders Svelte Flow.
- **`BlockCanvas` must be loaded with a dynamic `import()`, not a static import.** `NetworkEditor` does this (Task 13), so that `@xyflow/svelte` and `BlockNode` never enter the prerender module graph. A static import plus an `{#if browser}` guard only prevents rendering, not importing.

Verify by `npm run check`, `npm run build`, and the manual checklist in Task 16. If Svelte Flow's reactive props or `useSvelteFlow`'s coordinate helper do not behave as described, consult the installed package's types under `node_modules/@xyflow/svelte/dist/lib/` and adjust — but report any deviation from this task in your report rather than silently changing the interaction model.

Two API details confirmed against the installed 1.6.6 types, so use them exactly: `onnodeclick` receives a single destructured object `{ node, event }`, and `onpaneclick` receives `{ event }`. `useSvelteFlow()` exposes `screenToFlowPosition({ x, y })`.

- [ ] **Step 1: Write `BlockCanvas.svelte`**

```svelte
<script lang="ts">
  import { browser } from '$app/environment';
  import {
    Background,
    Controls,
    MarkerType,
    SvelteFlow,
    useSvelteFlow,
    type Edge,
    type Node,
    type NodeTypes
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { connectionToIntent, NODE_GAP, NODE_WIDTH, toFlow } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { dropIndexFor } from '../editor/placement';
  import type { BlockKind } from '../network/types';
  import BlockNode from './BlockNode.svelte';

  let {
    store,
    palette,
    ondragover
  }: {
    store: NetworkStore;
    palette: BlockKind[];
    ondragover: (kind: BlockKind | null) => void;
  } = $props();

  const nodeTypes: NodeTypes = { block: BlockNode as NodeTypes[string] };

  const flow = $derived(toFlow(store.network, store.shapes));

  const nodes = $derived<Node[]>(
    flow.nodes.map((node) => ({
      id: node.id,
      type: 'block',
      position: node.position,
      selected: node.id === store.selectedBlockId,
      data: { ...node.data, onremove: () => store.removeBlock(node.id) }
    }))
  );

  const edges = $derived<Edge[]>(
    flow.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? undefined,
      markerEnd: MarkerType.ArrowClosed,
      animated: false
    }))
  );

  let wrapper: HTMLDivElement | null = $state(null);
  let viewport = $state<{ screenToFlowPosition: (p: { x: number; y: number }) => { x: number; y: number } } | null>(null);

  function handleConnect(connection: { source: string; target: string }): void {
    const intent = connectionToIntent(connection, store.network);
    if (!intent) return;
    store.moveBlock(intent.from, intent.to);
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    const kind = event.dataTransfer?.getData('application/visnet-block') as BlockKind | undefined;
    ondragover(null);
    if (!kind || !palette.includes(kind)) return;

    const x = viewport && wrapper
      ? viewport.screenToFlowPosition({ x: event.clientX, y: event.clientY }).x
      : 0;
    const index = dropIndexFor(x, store.network.blocks.length, NODE_WIDTH, NODE_GAP);
    store.addBlock(kind, index);
  }
</script>

{#if browser}
  <div
    class="canvas"
    bind:this={wrapper}
    ondragover={(event) => event.preventDefault()}
    ondrop={handleDrop}
    data-testid="canvas"
  >
    <SvelteFlow
      {nodes}
      {edges}
      {nodeTypes}
      fitView
      nodesDraggable={false}
      nodesConnectable
      onconnect={handleConnect}
      onnodeclick={({ node }) => store.select(node.id)}
      onpaneclick={() => store.select(null)}
    >
      <Background />
      <Controls />
      <CanvasViewport bind:viewport />
    </SvelteFlow>
  </div>
{/if}

<style>
  .canvas {
    height: 100%;
    min-height: 320px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg);
  }
</style>
```

`CanvasViewport` is a tiny child component whose only job is to hand the parent Svelte Flow's coordinate helper, because `useSvelteFlow` must be called inside the flow's context:

`src/lib/components/CanvasViewport.svelte`:

```svelte
<script lang="ts">
  import { useSvelteFlow } from '@xyflow/svelte';

  let {
    viewport = $bindable()
  }: {
    viewport: { screenToFlowPosition: (point: { x: number; y: number }) => { x: number; y: number } } | null;
  } = $props();

  const flow = useSvelteFlow();
  $effect(() => {
    viewport = flow;
  });
</script>
```

- [ ] **Step 2: Verify**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`. Type errors against Svelte Flow's prop types are the most likely failure here; resolve them against the installed types rather than by casting to `any`.

Run: `npm run lint`
Expected: no errors.

Run: `npm test`
Expected: unchanged — no new tests in this task.

Run: `npm run build`
Expected: build succeeds and `build/index.html` exists. If the build fails with a `window is not defined` or similar, the browser guard is missing or the import is at module scope; fix the guard rather than disabling prerendering.

- [ ] **Step 3: Commit**

```bash
git add src/lib/components/BlockCanvas.svelte src/lib/components/CanvasViewport.svelte
git commit -m "feat: add the Svelte Flow canvas"
```

---

### Task 7: Inspector panel

**Files:**
- Create: `src/lib/components/InspectorPanel.svelte`
- Test: `src/lib/components/InspectorPanel.test.ts`

**Interfaces:**
- Consumes: `NetworkStore` from `../editor/networkStore.svelte`; `PARAM_DESCRIPTIONS`, `BLOCK_DESCRIPTIONS` from `../network/descriptions`; `parameterBounds` from `../network/constraints`; `shapeLabel` from `../editor/flow`.
- Produces: `InspectorPanel.svelte` with props `{ store: NetworkStore }`. It shows the selected block's plain-language description, its incoming shape as read-only context labelled with the block it came from, and controls for exactly the parameters that block kind has. Every control has a `title` and a visible description from `PARAM_DESCRIPTIONS`.

Control behaviour, which is the app's "automatic input selection" requirement:
- `input` — a text field of comma-separated numbers; on change it parses, drops non-numbers, and requires at least one positive number, otherwise it leaves the shape untouched and shows an inline message.
- `linear` — `units`, a number input, minimum 1.
- `conv2d` — `filters` (number, minimum 1); `kernelSize` and `stride` as `<select>`s limited to `parameterBounds(inShape)` when bounds exist, falling back to the stored value as the only option when they do not; `padding` as a two-option `<select>` where each option's label states the resulting output size, e.g. `same — stays 28×28`, `valid — becomes 26×26`.
- `flatten`, `relu`, `sigmoid`, `softmax` — no parameters; show the description only.
- `output` — `units`, a number input, minimum 1.

Every numeric and text field commits on **`change`**, not on each keystroke. Committing per keystroke would push one history entry per character and would let the clamped write-back fight the field's own value — typing `16` into a field bound to `units` becomes `116` once the first keystroke is clamped and written back. `commitNumber` must ignore an empty or non-finite value and leave the network untouched. That is why the tests below commit with `fireEvent.change` rather than `userEvent.type`.

Also renders the selected block's output shape, its parameter count, and two buttons, `Move left` and `Move right`, which call `store.moveSelectedBy(-1)` and `store.moveSelectedBy(1)`. Both are disabled for the input and output blocks, which cannot move.

Element test ids: `inspector`, `inspector-empty`, `param-units`, `param-filters`, `param-kernel-size`, `param-stride`, `param-padding`, `param-shape`, `move-left`, `move-right`, `inspector-incoming`.

- [ ] **Step 1: Write the failing test**

`src/lib/components/InspectorPanel.test.ts`:

```ts
import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import InspectorPanel from './InspectorPanel.svelte';

function storeWithSelection(index: number): NetworkStore {
  const store = new NetworkStore();
  store.select(store.network.blocks[index].id);
  return store;
}

describe('InspectorPanel', () => {
  it('prompts the user when nothing is selected', () => {
    render(InspectorPanel, { props: { store: new NetworkStore() } });
    expect(screen.getByTestId('inspector-empty')).toBeTruthy();
  });

  it('shows the selected block and its incoming shape', () => {
    render(InspectorPanel, { props: { store: storeWithSelection(1) } });
    expect(screen.getByTestId('inspector').textContent).toContain('linear');
    expect(screen.getByTestId('inspector-incoming').textContent).toContain('[2]');
  });

  it('edits a linear layer unit count', async () => {
    const store = storeWithSelection(1);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-units'), { target: { value: '16' } });
    expect(store.network.blocks[1]).toMatchObject({ units: 16 });
  });

  it('shows no unit control for an activation block', () => {
    render(InspectorPanel, { props: { store: storeWithSelection(2) } });
    expect(screen.queryByTestId('param-units')).toBeNull();
    expect(screen.getByTestId('inspector').textContent).toContain('relu');
  });

  it('limits convolution kernel choices to the incoming image size', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('conv2d', 1);
    store.updateBlock(store.network.blocks[1].id, { kernelSize: 3 });
    render(InspectorPanel, { props: { store } });

    const options = Array.from(
      screen.getByTestId('param-kernel-size').querySelectorAll('option')
    ).map((option) => option.getAttribute('value'));
    expect(options).toEqual(['1', '2', '3', '4']);
    expect(screen.getByTestId('param-kernel-size')).toBeTruthy();
  });

  it('describes what each padding choice does to the image size', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    store.addBlock('conv2d', 1);
    render(InspectorPanel, { props: { store } });

    const labels = Array.from(
      screen.getByTestId('param-padding').querySelectorAll('option')
    ).map((option) => option.textContent ?? '');
    expect(labels.some((label) => label.includes('28×28'))).toBe(true);
    expect(labels.some((label) => label.includes('26×26'))).toBe(true);
  });

  it('moves the selected block and refuses to move the input or output', async () => {
    const store = storeWithSelection(1);
    const moving = store.network.blocks[1].id;
    render(InspectorPanel, { props: { store } });
    await userEvent.click(screen.getByTestId('move-right'));
    expect(store.network.blocks[2].id).toBe(moving);
  });

  it('disables the move buttons for the input and output blocks', () => {
    const store = storeWithSelection(0);
    render(InspectorPanel, { props: { store } });
    expect((screen.getByTestId('move-left') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('move-right') as HTMLButtonElement).disabled).toBe(true);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/InspectorPanel.test.ts`
Expected: FAIL — `Failed to resolve import "./InspectorPanel.svelte"`.

- [ ] **Step 3: Write the implementation**

`src/lib/components/InspectorPanel.svelte`:

```svelte
<script lang="ts">
  import { parameterBounds } from '../network/constraints';
  import { BLOCK_DESCRIPTIONS, PARAM_DESCRIPTIONS } from '../network/descriptions';
  import { shapeLabel } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import type { Block, InputBlock } from '../network/types';

  let { store }: { store: NetworkStore } = $props();

  const block = $derived(store.selectedBlock);
  const index = $derived(
    store.network.blocks.findIndex((candidate) => candidate.id === store.selectedBlockId)
  );
  const info = $derived(index === -1 ? null : store.shapes.perBlock[index]);
  const previousKind = $derived(index > 0 ? store.network.blocks[index - 1].kind : null);
  const inShape = $derived(info?.inShape ?? null);
  const bounds = $derived(parameterBounds(inShape));
  const kernelChoices = $derived(
    bounds && block?.kind === 'conv2d' ? bounds.kernelSize : block?.kind === 'conv2d' ? [block.kernelSize] : []
  );
  const strideChoices = $derived(
    bounds && block?.kind === 'conv2d' ? bounds.stride : block?.kind === 'conv2d' ? [block.stride] : []
  );
  const canMove = $derived(block !== null && block.kind !== 'input' && block.kind !== 'output');
  let shapeError = $state<string | null>(null);

  function patch(next: Partial<Block>): void {
    if (!block) return;
    store.updateBlock(block.id, next);
  }

  function commitShape(event: Event): void {
    if (!block || block.kind !== 'input') return;
    const raw = (event.currentTarget as HTMLInputElement).value;
    const parsed = raw
      .split(',')
      .map((part) => Number(part.trim()))
      .filter((value) => Number.isFinite(value) && value > 0)
      .map((value) => Math.floor(value));

    if (parsed.length === 0) {
      shapeError = 'Enter at least one positive number, separated by commas.';
      return;
    }
    shapeError = null;
    patch({ shape: parsed } as Partial<InputBlock>);
  }

  function commitNumber(event: Event, key: 'units' | 'filters'): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(value)) return;
    patch({ [key]: value } as Partial<Block>);
  }

  function outputSizeFor(padding: 'same' | 'valid'): string {
    if (!inShape || inShape.length !== 3 || !block || block.kind !== 'conv2d') return '';
    const [height, width] = inShape;
    const { kernelSize, stride } = block;
    const compute = (dimension: number): number =>
      padding === 'same'
        ? Math.ceil(dimension / stride)
        : Math.floor((dimension - kernelSize) / stride) + 1;
    const result = `${compute(height)}×${compute(width)}`;
    if (padding === 'same' && result === `${height}×${width}`) return `stays ${result}`;
    return `becomes ${result}`;
  }
</script>

{#if !block}
  <div class="inspector" data-testid="inspector-empty">
    <p>Select a block to see and change its settings.</p>
  </div>
{:else}
  <div class="inspector" data-testid="inspector">
    <h2>{block.kind}</h2>
    <p class="description">{BLOCK_DESCRIPTIONS[block.kind]}</p>

    {#if block.kind !== 'input'}
      <p class="incoming" data-testid="inspector-incoming">
        Input {shapeLabel(inShape) ?? '—'}{previousKind ? ` from ${previousKind}` : ''}
      </p>
    {/if}

    {#if block.kind === 'input'}
      <label>
        <span>Shape</span>
        <input
          type="text"
          data-testid="param-shape"
          title={PARAM_DESCRIPTIONS.inputShape}
          value={block.shape.join(', ')}
          onchange={commitShape}
        />
        <small>{PARAM_DESCRIPTIONS.inputShape}</small>
      </label>
      {#if shapeError}<p class="error">{shapeError}</p>{/if}
    {/if}

    {#if block.kind === 'linear' || block.kind === 'output'}
      <label>
        <span>Units</span>
        <input
          type="number"
          min="1"
          data-testid="param-units"
          title={PARAM_DESCRIPTIONS.units}
          value={block.units}
          onchange={(event) => commitNumber(event, 'units')}
        />
        <small>{PARAM_DESCRIPTIONS.units}</small>
      </label>
    {/if}

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

      <label>
        <span>Kernel size</span>
        <select
          data-testid="param-kernel-size"
          title={PARAM_DESCRIPTIONS.kernelSize}
          value={String(block.kernelSize)}
          onchange={(event) => patch({ kernelSize: Number(event.currentTarget.value) } as Partial<Block>)}
        >
          {#each kernelChoices as choice (choice)}
            <option value={String(choice)}>{choice}×{choice}</option>
          {/each}
        </select>
        <small>{PARAM_DESCRIPTIONS.kernelSize}</small>
      </label>

      <label>
        <span>Stride</span>
        <select
          data-testid="param-stride"
          title={PARAM_DESCRIPTIONS.stride}
          value={String(block.stride)}
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
          value={block.padding}
          onchange={(event) => patch({ padding: event.currentTarget.value as 'same' | 'valid' } as Partial<Block>)}
        >
          <option value="same">same — {outputSizeFor('same')}</option>
          <option value="valid">valid — {outputSizeFor('valid')}</option>
        </select>
        <small>{PARAM_DESCRIPTIONS.padding}</small>
      </label>
    {/if}

    <dl class="facts">
      <dt>Output</dt>
      <dd>{shapeLabel(info?.outShape ?? null) ?? '—'}</dd>
      <dt>Parameters</dt>
      <dd>{info?.paramCount ?? 0}</dd>
    </dl>

    <div class="moves">
      <button
        type="button"
        data-testid="move-left"
        disabled={!canMove}
        onclick={() => store.moveSelectedBy(-1)}
      >
        Move left
      </button>
      <button
        type="button"
        data-testid="move-right"
        disabled={!canMove}
        onclick={() => store.moveSelectedBy(1)}
      >
        Move right
      </button>
    </div>
  </div>
{/if}

<style>
  .inspector {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-base);
  }

  .description,
  .incoming {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  input,
  select {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  .error {
    margin: 0;
    color: var(--color-error);
    font-size: var(--text-xs);
  }

  .facts {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-1) var(--space-2);
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }

  .facts dt {
    color: var(--color-text-muted);
  }

  .facts dd {
    margin: 0;
  }

  .moves {
    display: flex;
    gap: var(--space-2);
  }

  .moves button {
    flex: 1;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }

  .moves button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/components/InspectorPanel.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/InspectorPanel.svelte src/lib/components/InspectorPanel.test.ts
git commit -m "feat: add the block inspector with constrained parameters"
```

---

### Task 8: Shape table and issues panel

**Files:**
- Create: `src/lib/components/ShapeTable.svelte`
- Create: `src/lib/components/IssuesPanel.svelte`
- Test: `src/lib/components/ShapeTable.test.ts`
- Test: `src/lib/components/IssuesPanel.test.ts`

**Interfaces:**
- Consumes: `NetworkStore`; `shapeLabel` from `../editor/flow`; `Issue` from `../network/validate`.
- Produces:
  - `ShapeTable.svelte`, props `{ store: NetworkStore }`. A table with one row per block: position, block kind, input shape, output shape, parameter count, plus a final row with the network's total parameter count. Unknown shapes render `—`. This is the "see the shape of the data at any point in the pipeline" view. Test ids: `shape-table`, `shape-row`.
  - `IssuesPanel.svelte`, props `{ store: NetworkStore }`. Lists every error and warning as its `title`, its `message`, and its `fix`, the fix visually marked as an instruction. Clicking an entry selects the offending block when it has a `blockId`. When there are no issues, it says so plainly. Test ids: `issues-panel`, `issue`, `issue-fix`, `issues-empty`.

- [ ] **Step 1: Write the failing tests**

`src/lib/components/ShapeTable.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import ShapeTable from './ShapeTable.svelte';

describe('ShapeTable', () => {
  it('lists one row per block with its shapes', () => {
    const store = new NetworkStore();
    render(ShapeTable, { props: { store } });

    const rows = screen.getAllByTestId('shape-row');
    expect(rows).toHaveLength(store.network.blocks.length);
    expect(rows[0].textContent).toContain('input');
    expect(rows[1].textContent).toContain('[2]');
    expect(rows[1].textContent).toContain('[8]');
  });

  it('reports the network total parameter count', () => {
    const store = new NetworkStore();
    render(ShapeTable, { props: { store } });
    expect(screen.getByTestId('shape-table').textContent).toContain('42');
  });

  it('shows a dash for shapes it cannot compute', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    render(ShapeTable, { props: { store } });
    const rows = screen.getAllByTestId('shape-row');
    expect(rows[1].textContent).toContain('—');
  });
});
```

`src/lib/components/IssuesPanel.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import IssuesPanel from './IssuesPanel.svelte';

describe('IssuesPanel', () => {
  it('says the network is fine when there are no issues', () => {
    render(IssuesPanel, { props: { store: new NetworkStore() } });
    expect(screen.getByTestId('issues-empty')).toBeTruthy();
  });

  it('shows a warning with its fix', () => {
    const store = new NetworkStore();
    store.removeBlock(store.network.blocks[4].id);
    render(IssuesPanel, { props: { store } });

    const issue = screen.getByTestId('issue');
    expect(issue.textContent).toContain('Add a Softmax for probabilities');
    expect(issue.textContent).toContain('Add a Softmax block after the last Linear layer.');
    expect(screen.getByTestId('issue-fix')).toBeTruthy();
  });

  it('shows an error and selects the offending block when clicked', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    render(IssuesPanel, { props: { store } });

    const issues = screen.getAllByTestId('issue');
    expect(issues.some((issue) => issue.textContent?.includes('Linear layer needs a flat list'))).toBe(
      true
    );

    await userEvent.click(issues[0]);
    expect(store.selectedBlockId).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/lib/components/ShapeTable.test.ts src/lib/components/IssuesPanel.test.ts`
Expected: FAIL — both imports unresolvable.

- [ ] **Step 3: Write the implementations**

`src/lib/components/ShapeTable.svelte`:

```svelte
<script lang="ts">
  import { shapeLabel } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let { store }: { store: NetworkStore } = $props();
</script>

<table class="shapes" data-testid="shape-table">
  <caption>The shape of the data at every step</caption>
  <thead>
    <tr>
      <th scope="col">#</th>
      <th scope="col">Block</th>
      <th scope="col">In</th>
      <th scope="col">Out</th>
      <th scope="col">Parameters</th>
    </tr>
  </thead>
  <tbody>
    {#each store.network.blocks as block, index (block.id)}
      {@const info = store.shapes.perBlock[index]}
      <tr data-testid="shape-row">
        <td>{index + 1}</td>
        <td>{block.kind}</td>
        <td>{shapeLabel(info.inShape) ?? '—'}</td>
        <td>{shapeLabel(info.outShape) ?? '—'}</td>
        <td>{info.paramCount ?? '—'}</td>
      </tr>
    {/each}
  </tbody>
  <tfoot>
    <tr>
      <th scope="row" colspan="4">Total parameters</th>
      <td>{store.shapes.totalParamCount}</td>
    </tr>
  </tfoot>
</table>

<style>
  .shapes {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-xs);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  caption {
    text-align: left;
    padding: var(--space-2) var(--space-3);
    color: var(--color-text-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  th,
  td {
    text-align: left;
    padding: var(--space-1) var(--space-3);
    border-top: 1px solid var(--color-border);
    font-family: var(--font-mono);
  }

  thead th {
    color: var(--color-text-muted);
    font-family: var(--font-sans);
    font-weight: 600;
  }

  tfoot th,
  tfoot td {
    font-weight: 600;
  }
</style>
```

`src/lib/components/IssuesPanel.svelte`:

```svelte
<script lang="ts">
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let { store }: { store: NetworkStore } = $props();
</script>

<div class="issues" data-testid="issues-panel">
  <h2>Problems</h2>

  {#if store.issues.length === 0}
    <p class="ok" data-testid="issues-empty">This network looks good.</p>
  {:else}
    <ul>
      {#each store.issues as issue, index (index)}
        <li
          class="issue"
          class:error={issue.severity === 'error'}
          class:warning={issue.severity === 'warning'}
        >
          <button
            type="button"
            data-testid="issue"
            onclick={() => issue.blockId && store.select(issue.blockId)}
            disabled={!issue.blockId}
          >
            <span class="severity">{issue.severity === 'error' ? 'Blocks training' : 'Heads up'}</span>
            <span class="title">{issue.title}</span>
            <span class="message">{issue.message}</span>
            <span class="fix" data-testid="issue-fix">{issue.fix}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .issues {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .ok {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }

  button {
    width: 100%;
    display: grid;
    gap: var(--space-1);
    text-align: left;
    padding: var(--space-2) var(--space-3);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-left-width: 3px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font: inherit;
  }

  button:disabled {
    cursor: default;
  }

  .error button {
    border-left-color: var(--color-error);
  }

  .warning button {
    border-left-color: var(--color-warning);
  }

  .severity {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .title {
    font-weight: 600;
    font-size: var(--text-sm);
  }

  .message {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  .fix {
    font-size: var(--text-sm);
    color: var(--color-text);
  }
</style>
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/lib/components/ShapeTable.test.ts src/lib/components/IssuesPanel.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/ShapeTable.svelte src/lib/components/ShapeTable.test.ts src/lib/components/IssuesPanel.svelte src/lib/components/IssuesPanel.test.ts
git commit -m "feat: add the shape table and issues panel"
```

---

### Task 9: Loss chart and stats readout

**Files:**
- Create: `src/lib/components/LossChart.svelte`
- Create: `src/lib/components/StatsReadout.svelte`
- Test: `src/lib/components/LossChart.test.ts`
- Test: `src/lib/components/StatsReadout.test.ts`

**Interfaces:**
- Consumes: `TrainStats` from `../training/Trainer`.
- Produces:
  - `LossChart.svelte`, props `{ points: number[]; width?: number; height?: number }` (defaults 320 and 120). Renders an SVG polyline of the per-epoch mean loss over the most recent 200 values, with the minimum and maximum labelled and the latest value shown. With fewer than two points it renders an explanatory placeholder instead of a line. Test ids: `loss-chart`, `loss-chart-empty`, `loss-chart-path`, `loss-chart-latest`.
  - `StatsReadout.svelte`, props `{ stats: TrainStats | null }`. Shows the epoch number, the latest loss, and the training accuracy as a percentage, each with a one-line plain-language explanation. Accuracy is shown only when the last epoch reported one. Test ids: `stats-readout`, `stats-epoch`, `stats-loss`, `stats-accuracy`, `stats-empty`.

- [ ] **Step 1: Write the failing tests**

`src/lib/components/LossChart.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import LossChart from './LossChart.svelte';

describe('LossChart', () => {
  it('explains itself when there is nothing to plot yet', () => {
    render(LossChart, { props: { points: [] } });
    expect(screen.getByTestId('loss-chart-empty')).toBeTruthy();
    expect(screen.queryByTestId('loss-chart-path')).toBeNull();
  });

  it('does not plot a single point', () => {
    render(LossChart, { props: { points: [0.5] } });
    expect(screen.queryByTestId('loss-chart-path')).toBeNull();
    expect(screen.getByTestId('loss-chart-latest').textContent).toContain('0.500');
  });

  it('plots one segment per consecutive pair', () => {
    render(LossChart, { props: { points: [1, 0.5, 0.25] } });
    const path = screen.getByTestId('loss-chart-path').getAttribute('d') ?? '';
    expect(path.split('L')).toHaveLength(3);
    expect(path.startsWith('M')).toBe(true);
  });

  it('shows the latest value', () => {
    render(LossChart, { props: { points: [1, 0.5, 0.25] } });
    expect(screen.getByTestId('loss-chart-latest').textContent).toContain('0.250');
  });

  it('plots only the most recent window', () => {
    const points = Array.from({ length: 300 }, (_, index) => index);
    render(LossChart, { props: { points } });
    const path = screen.getByTestId('loss-chart-path').getAttribute('d') ?? '';
    expect(path.split('L')).toHaveLength(200);
  });
});
```

`src/lib/components/StatsReadout.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import StatsReadout from './StatsReadout.svelte';

describe('StatsReadout', () => {
  it('prompts the user before training starts', () => {
    render(StatsReadout, { props: { stats: null } });
    expect(screen.getByTestId('stats-empty')).toBeTruthy();
  });

  it('shows the epoch and loss mid-epoch', () => {
    render(StatsReadout, {
      props: {
        stats: { epoch: 3, batch: 2, batchLoss: 0.42, epochMeanLoss: null, epochAccuracy: null }
      }
    });
    expect(screen.getByTestId('stats-epoch').textContent).toContain('3');
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.420');
    expect(screen.queryByTestId('stats-accuracy')).toBeNull();
  });

  it('shows accuracy once an epoch has completed', () => {
    render(StatsReadout, {
      props: {
        stats: {
          epoch: 4,
          batch: 0,
          batchLoss: 0.2,
          epochMeanLoss: 0.25,
          epochAccuracy: 0.875
        }
      }
    });
    expect(screen.getByTestId('stats-loss').textContent).toContain('0.250');
    expect(screen.getByTestId('stats-accuracy').textContent).toContain('87.5');
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/lib/components/LossChart.test.ts src/lib/components/StatsReadout.test.ts`
Expected: FAIL — both imports unresolvable.

- [ ] **Step 3: Write the implementations**

`src/lib/components/LossChart.svelte`:

```svelte
<script lang="ts">
  let { points, width = 320, height = 120 }: { points: number[]; width?: number; height?: number } =
    $props();

  const WINDOW = 200;
  const view = $derived(points.slice(-WINDOW));
  const lowest = $derived(view.length ? Math.min(...view) : 0);
  const highest = $derived(view.length ? Math.max(...view) : 1);
  const latest = $derived(view.length ? view[view.length - 1] : null);

  const path = $derived.by(() => {
    if (view.length < 2) return '';
    const span = highest - lowest || 1;
    return view
      .map((value, index) => {
        const x = (index / (view.length - 1)) * width;
        const y = height - ((value - lowest) / span) * height;
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });
</script>

<figure class="chart" data-testid="loss-chart">
  <figcaption>
    Loss per epoch
    {#if latest !== null}
      <span data-testid="loss-chart-latest">latest {latest.toFixed(3)}</span>
    {/if}
  </figcaption>

  {#if view.length < 2}
    <p class="empty" data-testid="loss-chart-empty">
      Press play to start training. The loss curve appears after the first epoch.
    </p>
  {:else}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss per epoch">
      <path data-testid="loss-chart-path" d={path} fill="none" stroke="var(--color-accent)" />
    </svg>
    <div class="axis">
      <span>lowest {lowest.toFixed(3)}</span>
      <span>highest {highest.toFixed(3)}</span>
    </div>
  {/if}
</figure>

<style>
  .chart {
    margin: 0;
    display: grid;
    gap: var(--space-1);
  }

  figcaption {
    display: flex;
    justify-content: space-between;
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .empty {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  svg {
    width: 100%;
    height: auto;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
  }

  .axis {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
```

`src/lib/components/StatsReadout.svelte`:

```svelte
<script lang="ts">
  import type { TrainStats } from '../training/Trainer';

  let { stats }: { stats: TrainStats | null } = $props();

  const loss = $derived(stats ? (stats.epochMeanLoss ?? stats.batchLoss) : null);
  const lossLabel = $derived(stats?.epochMeanLoss !== null && stats ? 'Average loss this epoch' : 'Loss on the last batch');
  const accuracy = $derived(stats?.epochAccuracy ?? null);
</script>

<div class="stats" data-testid="stats-readout">
  {#if !stats}
    <p class="empty" data-testid="stats-empty">
      Not training yet. Press play to start, or step to train one batch.
    </p>
  {:else}
    <dl>
      <div>
        <dt>Epoch</dt>
        <dd data-testid="stats-epoch">{stats.epoch}</dd>
        <small>One full pass over all the points.</small>
      </div>
      <div>
        <dt>Loss</dt>
        <dd data-testid="stats-loss">{loss?.toFixed(3) ?? '—'}</dd>
        <small>{lossLabel}. Lower is better.</small>
      </div>
      {#if accuracy !== null}
        <div>
          <dt>Accuracy</dt>
          <dd data-testid="stats-accuracy">{(accuracy * 100).toFixed(1)}%</dd>
          <small>Share of points the network classifies correctly.</small>
        </div>
      {/if}
    </dl>
  {/if}
</div>

<style>
  .stats {
    font-size: var(--text-sm);
  }

  .empty {
    margin: 0;
    color: var(--color-text-muted);
  }

  dl {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: var(--space-3);
    margin: 0;
  }

  dt {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  dd {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-lg);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }
</style>
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/lib/components/LossChart.test.ts src/lib/components/StatsReadout.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/LossChart.svelte src/lib/components/LossChart.test.ts src/lib/components/StatsReadout.svelte src/lib/components/StatsReadout.test.ts
git commit -m "feat: add the loss chart and stats readout"
```

---

### Task 10: Training panel

**Files:**
- Create: `src/lib/components/TrainingPanel.svelte`
- Test: `src/lib/components/TrainingPanel.test.ts`

**Interfaces:**
- Consumes: `NetworkStore`; `PARAM_DESCRIPTIONS` from `../network/descriptions`.
- Produces: `TrainingPanel.svelte`, props:

```ts
{
  store: NetworkStore;
  playing: boolean;
  disabled: boolean;
  onplay: () => void;
  onpause: () => void;
  onstep: () => void;
  onreset: () => void;
}
```

It renders the four training settings with their descriptions — `loss` and `optimizer` as `<select>`s, `learningRate` and `batchSize` as number inputs — writing through `store.updateTraining`. It renders `Play`, `Pause`, `Step one batch`, and `Reset model` buttons. `Play` is disabled when `disabled` or already `playing`; `Pause` is disabled when not `playing`; `Step` is disabled when `disabled`. When `disabled` it shows a plain-language line explaining that the network has errors to fix first, so the disabled buttons are never unexplained.

Test ids: `training-panel`, `training-loss`, `training-optimizer`, `training-learning-rate`, `training-batch-size`, `training-play`, `training-pause`, `training-step`, `training-reset`, `training-blocked`.

- [ ] **Step 1: Write the failing test**

`src/lib/components/TrainingPanel.test.ts`:

```ts
import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import TrainingPanel from './TrainingPanel.svelte';

function panel(overrides: Partial<Record<string, unknown>> = {}) {
  const store = new NetworkStore();
  const handlers = {
    onplay: vi.fn(),
    onpause: vi.fn(),
    onstep: vi.fn(),
    onreset: vi.fn()
  };
  render(TrainingPanel, {
    props: { store, playing: false, disabled: false, ...handlers, ...overrides }
  });
  return { store, ...handlers };
}

describe('TrainingPanel', () => {
  it('writes a loss change through to the network', async () => {
    const { store } = panel();
    await userEvent.selectOptions(screen.getByTestId('training-loss'), 'mse');
    expect(store.network.training.loss).toBe('mse');
  });

  it('writes an optimizer change through to the network', async () => {
    const { store } = panel();
    await userEvent.selectOptions(screen.getByTestId('training-optimizer'), 'sgd');
    expect(store.network.training.optimizer).toBe('sgd');
  });

  it('writes a learning rate change through to the network', async () => {
    const { store } = panel();
    await fireEvent.change(screen.getByTestId('training-learning-rate'), {
      target: { value: '0.25' }
    });
    expect(store.network.training.learningRate).toBe(0.25);
  });

  it('writes a batch size change through to the network', async () => {
    const { store } = panel();
    await fireEvent.change(screen.getByTestId('training-batch-size'), { target: { value: '64' } });
    expect(store.network.training.batchSize).toBe(64);
  });

  it('leaves the network alone when a numeric field is cleared', async () => {
    const { store } = panel();
    await fireEvent.change(screen.getByTestId('training-batch-size'), { target: { value: '' } });
    expect(store.network.training.batchSize).toBe(32);
  });

  it('describes every setting in plain language', () => {
    panel();
    const text = screen.getByTestId('training-panel').textContent ?? '';
    expect(text).toContain('The number the network tries to make smaller while training.');
    expect(text).toContain('The rule used to update the weights after each batch.');
    expect(text).toContain('How big each learning step is');
    expect(text).toContain('How many examples are used for one weight update.');
  });

  it('calls the control handlers', async () => {
    const { onplay, onpause, onstep, onreset } = panel({ playing: true });
    await userEvent.click(screen.getByTestId('training-play'));
    await userEvent.click(screen.getByTestId('training-pause'));
    await userEvent.click(screen.getByTestId('training-step'));
    await userEvent.click(screen.getByTestId('training-reset'));
    expect(onplay).not.toHaveBeenCalled();
    expect(onpause).toHaveBeenCalledTimes(1);
    expect(onstep).toHaveBeenCalledTimes(1);
    expect(onreset).toHaveBeenCalledTimes(1);
  });

  it('disables training controls and explains why when the network is invalid', () => {
    panel({ disabled: true });
    expect((screen.getByTestId('training-play') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-step') as HTMLButtonElement).disabled).toBe(true);
    const blocked = screen.getByTestId('training-blocked').textContent ?? '';
    expect(blocked).toContain('problems');
    expect(blocked).toContain('training');
  });

  it('disables play while already playing', () => {
    panel({ playing: true });
    expect((screen.getByTestId('training-play') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('training-pause') as HTMLButtonElement).disabled).toBe(false);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/TrainingPanel.test.ts`
Expected: FAIL — `Failed to resolve import "./TrainingPanel.svelte"`.

- [ ] **Step 3: Write the implementation**

`src/lib/components/TrainingPanel.svelte`:

```svelte
<script lang="ts">
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { PARAM_DESCRIPTIONS } from '../network/descriptions';

  let {
    store,
    playing,
    disabled,
    onplay,
    onpause,
    onstep,
    onreset
  }: {
    store: NetworkStore;
    playing: boolean;
    disabled: boolean;
    onplay: () => void;
    onpause: () => void;
    onstep: () => void;
    onreset: () => void;
  } = $props();

  function setNumber(event: Event, key: 'learningRate' | 'batchSize'): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(value) || value <= 0) return;
    store.updateTraining({ [key]: value });
  }
</script>

<div class="training" data-testid="training-panel">
  <h2>Training</h2>

  <label>
    <span>Loss</span>
    <select
      data-testid="training-loss"
      title={PARAM_DESCRIPTIONS.loss}
      disabled={disabled}
      value={store.network.training.loss}
      onchange={(event) =>
        store.updateTraining({ loss: event.currentTarget.value as 'mse' | 'crossEntropy' })}
    >
      <option value="crossEntropy">cross entropy</option>
      <option value="mse">mean squared error</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.loss}</small>
  </label>

  <label>
    <span>Optimizer</span>
    <select
      data-testid="training-optimizer"
      title={PARAM_DESCRIPTIONS.optimizer}
      disabled={disabled}
      value={store.network.training.optimizer}
      onchange={(event) => store.updateTraining({ optimizer: event.currentTarget.value as 'sgd' | 'adam' })}
    >
      <option value="adam">Adam</option>
      <option value="sgd">SGD</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.optimizer}</small>
  </label>

  <label>
    <span>Learning rate</span>
    <input
      type="number"
      step="0.001"
      min="0.0001"
      data-testid="training-learning-rate"
      title={PARAM_DESCRIPTIONS.learningRate}
      disabled={disabled}
      value={store.network.training.learningRate}
      onchange={(event) => setNumber(event, 'learningRate')}
    />
    <small>{PARAM_DESCRIPTIONS.learningRate}</small>
  </label>

  <label>
    <span>Batch size</span>
    <input
      type="number"
      min="1"
      data-testid="training-batch-size"
      title={PARAM_DESCRIPTIONS.batchSize}
      disabled={disabled}
      value={store.network.training.batchSize}
      onchange={(event) => setNumber(event, 'batchSize')}
    />
    <small>{PARAM_DESCRIPTIONS.batchSize}</small>
  </label>

  <div class="controls">
    <button type="button" data-testid="training-play" disabled={disabled || playing} onclick={onplay}>
      Play
    </button>
    <button type="button" data-testid="training-pause" disabled={!playing} onclick={onpause}>
      Pause
    </button>
    <button type="button" data-testid="training-step" disabled={disabled} onclick={onstep}>
      Step one batch
    </button>
    <button type="button" data-testid="training-reset" onclick={onreset}>Reset model</button>
  </div>

  {#if disabled}
    <p class="blocked" data-testid="training-blocked">
      There are problems to fix below before training. The network cannot be built until they are resolved.
    </p>
  {/if}
</div>

<style>
  .training {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  select,
  input {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  .controls button {
    padding: var(--space-1) var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }

  .controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .blocked {
    margin: 0;
    color: var(--color-error);
    font-size: var(--text-sm);
  }
</style>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/components/TrainingPanel.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/TrainingPanel.svelte src/lib/components/TrainingPanel.test.ts
git commit -m "feat: add the training panel"
```

---

### Task 11: Decision boundary canvas

**Files:**
- Create: `src/lib/components/DecisionBoundary.svelte`
- Test: `src/lib/components/DecisionBoundary.test.ts`

**Interfaces:**
- Consumes: `PointDataset`, `Point` from `../data/points`; `CLASS_COLOURS`, `BACKGROUND_RGB` from `../render/palette`; `GRID_SIZE`, `clientToDomain`, `classesToRgba`, `sampleGrid` from `../render/boundary`; `Rect` from `../render/boundary`.
- Produces: `DecisionBoundary.svelte`, props:

```ts
{
  model: tf.LayersModel | null;
  dataset: PointDataset;
  selectedLabel: 0 | 1;
  onaddpoint: (x: number, y: number) => void;
  caption?: string;
}
```

It renders a square canvas showing the decision boundary and the data points, and turns a click into `onaddpoint` with domain coordinates.

**Browser-only rule:** `../render/boundary` imports `@tensorflow/tfjs` at module scope, so it must **not** be imported statically by a prerendered component. Load it with a dynamic `import()` inside `onMount` and keep the loaded module in state. The `model` prop is typed with `import type`, which is erased at compile time and therefore safe.

**Graceful degradation:** when `canvas.getContext('2d')` returns `null` (as in jsdom), the component must not crash; it skips drawing and keeps handling clicks. A short caption under the canvas explains what the user is looking at.

Element test ids: `decision-boundary`, `boundary-canvas`, `boundary-caption`.

- [ ] **Step 1: Write the failing test**

`src/lib/components/DecisionBoundary.test.ts`:

```ts
import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PointDataset } from '../data/points';
import DecisionBoundary from './DecisionBoundary.svelte';

const DATASET: PointDataset = {
  points: [
    { x: 0, y: 0, label: 0 },
    { x: 0.5, y: 0.5, label: 1 }
  ],
  numClasses: 2
};

const RECT = { left: 0, top: 0, width: 200, height: 200 };

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    ...RECT,
    right: RECT.width,
    bottom: RECT.height,
    x: 0,
    y: 0,
    toJSON: () => ({})
  } as DOMRect);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DecisionBoundary', () => {
  it('renders a canvas and a caption', () => {
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {} }
    });
    expect(screen.getByTestId('boundary-canvas')).toBeTruthy();
    expect(screen.getByTestId('boundary-caption').textContent).toBeTruthy();
  });

  it('says the network must be fixed before a boundary can be drawn', () => {
    render(DecisionBoundary, {
      props: {
        model: null,
        dataset: DATASET,
        selectedLabel: 0,
        onaddpoint: () => {},
        caption: 'Fix the network to see the boundary.'
      }
    });
    expect(screen.getByTestId('boundary-caption').textContent).toContain('Fix the network');
  });

  async function waitForBoundary(): Promise<void> {
    await vi.waitFor(() => {
      expect(screen.getByTestId('decision-boundary').getAttribute('data-ready')).toBe('true');
    });
  }

  it('turns a click into domain coordinates', async () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 1, onaddpoint }
    });
    await waitForBoundary();

    fireEvent.click(screen.getByTestId('boundary-canvas'), { clientX: 100, clientY: 100 });
    expect(onaddpoint).toHaveBeenCalledTimes(1);
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(0, 5);
    expect(y).toBeCloseTo(0, 5);
  });

  it('maps the top-right corner to the positive x, positive y corner', async () => {
    const onaddpoint = vi.fn();
    render(DecisionBoundary, {
      props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint }
    });
    await waitForBoundary();

    fireEvent.click(screen.getByTestId('boundary-canvas'), { clientX: 200, clientY: 0 });
    const [x, y] = onaddpoint.mock.calls[0];
    expect(x).toBeCloseTo(1, 5);
    expect(y).toBeCloseTo(1, 5);
  });

  it('does not crash when the canvas has no 2d context', () => {
    expect(() =>
      render(DecisionBoundary, {
        props: { model: null, dataset: DATASET, selectedLabel: 0, onaddpoint: () => {} }
      })
    ).not.toThrow();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/DecisionBoundary.test.ts`
Expected: FAIL — `Failed to resolve import "./DecisionBoundary.svelte"`.

- [ ] **Step 3: Write the implementation**

`src/lib/components/DecisionBoundary.svelte`:

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import type * as tf from '@tensorflow/tfjs';
  import type { PointDataset } from '../data/points';
  import { BACKGROUND_RGB, CLASS_COLOURS } from '../render/palette';

  let {
    model,
    dataset,
    selectedLabel,
    onaddpoint,
    caption = 'Each coloured area is the class the network predicts at that spot. Click to add a point.'
  }: {
    model: tf.LayersModel | null;
    dataset: PointDataset;
    selectedLabel: 0 | 1;
    onaddpoint: (x: number, y: number) => void;
    caption?: string;
  } = $props();

  const SIZE = 320;
  const POINT_RADIUS = 4;
  const COLOURS = CLASS_COLOURS.map((colour) => colour.rgb);

  let canvas: HTMLCanvasElement | null = $state(null);
  let boundary = $state<typeof import('../render/boundary') | null>(null);

  onMount(async () => {
    boundary = await import('../render/boundary');
  });

  $effect(() => {
    const module = boundary;
    const element = canvas;
    if (!module || !element) return;
    const context = element.getContext('2d');
    if (!context) return;

    const { classesToRgba } = module;

    if (model) {
      const classes = module.sampleGrid(model, GRID_SIZE);
      const rgba = classesToRgba(classes, COLOURS, GRID_SIZE);
      const offscreen = document.createElement('canvas');
      offscreen.width = GRID_SIZE;
      offscreen.height = GRID_SIZE;
      const offscreenContext = offscreen.getContext('2d');
      if (offscreenContext) {
        offscreenContext.putImageData(
          new ImageData(new Uint8ClampedArray(rgba), GRID_SIZE, GRID_SIZE),
          0,
          0
        );
        context.imageSmoothingEnabled = true;
        context.clearRect(0, 0, SIZE, SIZE);
        context.drawImage(offscreen, 0, 0, SIZE, SIZE);
      }
    } else {
      context.fillStyle = `rgb(${BACKGROUND_RGB.join(',')})`;
      context.fillRect(0, 0, SIZE, SIZE);
    }

    for (const point of dataset.points) {
      const px = ((point.x + 1) / 2) * SIZE;
      const py = (1 - (point.y + 1) / 2) * SIZE;
      context.beginPath();
      context.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      context.fillStyle = CLASS_COLOURS[point.label].hex;
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = '#ffffff';
      context.stroke();
    }
  });

  function handleClick(event: MouseEvent): void {
    const module = boundary;
    if (!module || !canvas) return;
    const { x, y } = module.clientToDomain(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect()
    );
    onaddpoint(x, y);
  }
</script>

<figure class="boundary" data-testid="decision-boundary" data-ready={boundary ? 'true' : 'false'}>
  <canvas
    bind:this={canvas}
    width={SIZE}
    height={SIZE}
    data-testid="boundary-canvas"
    onclick={handleClick}
  ></canvas>
  <figcaption data-testid="boundary-caption">
    {caption}
    <span class="hint">Adding class {selectedLabel} points.</span>
  </figcaption>
</figure>

<style>
  .boundary {
    margin: 0;
    display: grid;
    gap: var(--space-1);
    justify-items: start;
  }

  canvas {
    width: 100%;
    max-width: 360px;
    aspect-ratio: 1;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    cursor: crosshair;
    background: var(--color-bg);
  }

  figcaption {
    max-width: 360px;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .hint {
    display: block;
    font-weight: 600;
    color: var(--color-text);
  }
</style>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/components/DecisionBoundary.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify the build still prerenders**

Run: `npm run build`
Expected: build succeeds. A failure here means `@tensorflow/tfjs` reached the prerender path; the dynamic import is the thing to check.

- [ ] **Step 6: Commit**

```bash
git add src/lib/components/DecisionBoundary.svelte src/lib/components/DecisionBoundary.test.ts
git commit -m "feat: add the decision boundary canvas"
```

---

### Task 12: MLP example configuration and dataset store

**Files:**
- Create: `src/lib/examples/mlp/example.ts`
- Create: `src/lib/examples/mlp/datasetStore.svelte.ts`
- Test: `src/lib/examples/mlp/datasetStore.svelte.test.ts`

**Interfaces:**
- Consumes: `generate`, `addPoint`, `clearPoints`, `GENERATOR_NAMES`, `GENERATOR_DESCRIPTIONS`, types `PointDataset`, `GeneratorName` from `../../data/points`; `BlockKind` from `../../network/types`.
- Produces:

```ts
// example.ts
export const MLP_PALETTE: BlockKind[];
export const DEFAULT_GENERATOR: GeneratorName;
export const DEFAULT_POINT_COUNT: number;
export const DEFAULT_SEED: number;
export const CLASS_LABELS: [string, string];
export function defaultDataset(): PointDataset;

// datasetStore.svelte.ts
export class DatasetStore {
  dataset = $state<PointDataset>(...);
  generator = $state<GeneratorName>(...);
  pointCount = $state<number>(...);
  seed = $state<number>(...);
  selectedLabel = $state<0 | 1>(0);
  regenerate(): void;
  reseed(): void;
  addPoint(x: number, y: number): void;
  clear(): void;
  selectLabel(label: 0 | 1): void;
}
```

- `MLP_PALETTE` is the blocks the MLP example permits: `['linear', 'relu', 'sigmoid', 'softmax']`. Convolution and flatten belong to the CNN example, so they are not offered here.
- `CLASS_LABELS` names the two classes for the UI, e.g. `['blue', 'pink']`, matching the `--class-0` and `--class-1` tokens.
- `defaultDataset()` returns `generate(DEFAULT_GENERATOR, DEFAULT_POINT_COUNT, DEFAULT_SEED)`.
- `regenerate()` rebuilds the dataset from the current generator, count, and seed.
- `reseed()` advances the seed so the user gets a different sample of the same shape, and rebuilds.
- `addPoint` clamps both coordinates into `[-1, 1]` before appending, so a click near or past the canvas edge still produces a valid point, and uses the currently selected class label.
- `clear()` empties the dataset but keeps the generator settings.
- The dataset store holds no tensors; the page owns those, because tensor lifetime must be tied to the page's effect cleanup.

- [ ] **Step 1: Write the failing test**

`src/lib/examples/mlp/datasetStore.svelte.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { GENERATOR_NAMES } from '../../data/points';
import { DatasetStore } from './datasetStore.svelte';
import { CLASS_LABELS, DEFAULT_GENERATOR, MLP_PALETTE } from './example';

describe('example configuration', () => {
  it('offers only the blocks the MLP example teaches', () => {
    expect(MLP_PALETTE).toEqual(['linear', 'relu', 'sigmoid', 'softmax']);
  });

  it('names both classes', () => {
    expect(CLASS_LABELS).toHaveLength(2);
    expect(CLASS_LABELS[0]).toBeTruthy();
    expect(CLASS_LABELS[1]).toBeTruthy();
  });

  it('starts from a known generator', () => {
    expect(GENERATOR_NAMES).toContain(DEFAULT_GENERATOR);
  });
});

describe('DatasetStore', () => {
  it('starts with a generated dataset', () => {
    const store = new DatasetStore();
    expect(store.dataset.points.length).toBeGreaterThan(0);
    expect(store.dataset.numClasses).toBe(2);
  });

  it('regenerates when the generator changes', () => {
    const store = new DatasetStore();
    store.generator = 'circles';
    store.regenerate();
    for (const point of store.dataset.points) {
      const radius = Math.hypot(point.x, point.y);
      if (point.label === 0) expect(radius).toBeLessThanOrEqual(0.3);
      else expect(radius).toBeGreaterThanOrEqual(0.6);
    }
  });

  it('respects the point count', () => {
    const store = new DatasetStore();
    store.pointCount = 40;
    store.regenerate();
    expect(store.dataset.points).toHaveLength(40);
  });

  it('reseeds to a different sample of the same size', () => {
    const store = new DatasetStore();
    const before = store.dataset;
    store.reseed();
    expect(store.dataset.points).toHaveLength(before.points.length);
    expect(store.dataset).not.toEqual(before);
  });

  it('adds a point with the selected label', () => {
    const store = new DatasetStore();
    store.clear();
    store.selectLabel(1);
    store.addPoint(0.25, -0.5);
    expect(store.dataset.points).toEqual([{ x: 0.25, y: -0.5, label: 1 }]);
  });

  it('clamps a click outside the canvas into the domain', () => {
    const store = new DatasetStore();
    store.clear();
    store.addPoint(4, -9);
    expect(store.dataset.points[0]).toEqual({ x: 1, y: -1, label: 0 });
  });

  it('clears every point but keeps the settings', () => {
    const store = new DatasetStore();
    store.generator = 'xor';
    store.pointCount = 25;
    store.clear();
    expect(store.dataset.points).toEqual([]);
    expect(store.dataset.numClasses).toBe(2);
    expect(store.generator).toBe('xor');
    expect(store.pointCount).toBe(25);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/examples/mlp/datasetStore.svelte.test.ts`
Expected: FAIL — `Failed to resolve import "./datasetStore.svelte"`.

- [ ] **Step 3: Write `example.ts`**

```ts
import { generate, type GeneratorName, type PointDataset } from '../../data/points';
import type { BlockKind } from '../../network/types';

export const MLP_PALETTE: BlockKind[] = ['linear', 'relu', 'sigmoid', 'softmax'];
export const DEFAULT_GENERATOR: GeneratorName = 'twoGaussians';
export const DEFAULT_POINT_COUNT = 200;
export const DEFAULT_SEED = 1;
export const CLASS_LABELS: [string, string] = ['blue', 'pink'];

export function defaultDataset(): PointDataset {
  return generate(DEFAULT_GENERATOR, DEFAULT_POINT_COUNT, DEFAULT_SEED);
}
```

- [ ] **Step 4: Write `datasetStore.svelte.ts`**

```ts
import { addPoint, clearPoints, generate, type GeneratorName, type PointDataset } from '../../data/points';
import {
  DEFAULT_GENERATOR,
  DEFAULT_POINT_COUNT,
  DEFAULT_SEED,
  defaultDataset
} from './example';

export class DatasetStore {
  dataset = $state<PointDataset>(defaultDataset());
  generator = $state<GeneratorName>(DEFAULT_GENERATOR);
  pointCount = $state<number>(DEFAULT_POINT_COUNT);
  seed = $state<number>(DEFAULT_SEED);
  selectedLabel = $state<0 | 1>(0);

  regenerate(): void {
    this.dataset = generate(this.generator, this.pointCount, this.seed);
  }

  reseed(): void {
    this.seed += 1;
    this.regenerate();
  }

  addPoint(x: number, y: number): void {
    const clamp = (value: number): number => Math.max(-1, Math.min(1, value));
    this.dataset = addPoint(this.dataset, clamp(x), clamp(y), this.selectedLabel);
  }

  clear(): void {
    this.dataset = clearPoints(this.dataset);
  }

  selectLabel(label: 0 | 1): void {
    this.selectedLabel = label;
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run src/lib/examples/mlp/datasetStore.svelte.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/examples/mlp/example.ts src/lib/examples/mlp/datasetStore.svelte.ts src/lib/examples/mlp/datasetStore.svelte.test.ts
git commit -m "feat: add the MLP example configuration and dataset store"
```

---

### Task 13: Editor toolbar and the embeddable editor

**Files:**
- Create: `src/lib/components/EditorToolbar.svelte`
- Create: `src/lib/components/NetworkEditor.svelte`
- Test: `src/lib/components/NetworkEditor.test.ts`

**Interfaces:**
- Consumes: `NetworkStore`; `BlockPalette`, `BlockCanvas`, `InspectorPanel`, `ShapeTable`, `IssuesPanel`, `EditorToolbar`; `BlockKind` from `../network/types`; `BLOCK_DESCRIPTIONS` from `../network/descriptions`.
- Produces:
  - `EditorToolbar.svelte`, props `{ store: NetworkStore; onfit?: () => void; onsave?: () => void; onload?: () => void; saving?: boolean }`. Renders `Undo` (disabled when `store.canUndo` is false), `Redo` (disabled when `store.canRedo` is false), `Reset network`, and, when the page supplies them, `Save model` and `Load model`. It also wires the keyboard shortcuts `Ctrl/Cmd+Z` for undo and `Ctrl/Cmd+Shift+Z` for redo on the window while the component is mounted. Test ids: `undo`, `redo`, `reset-network`, `save-model`, `load-model`.
  - `NetworkEditor.svelte` — **the embedding contract**, with exactly these props:

```ts
{
  store: NetworkStore;
  palette: BlockKind[];
  onsave?: () => void;
  onload?: () => void;
  saving?: boolean;
}
```

It composes the toolbar, the palette, the canvas, the inspector, the shape table, and the issues panel, and shows any `store.announcements` as a dismissible list so automatic corrections are visible. It knows nothing about datasets, training, or persistence beyond the two optional model callbacks. This is the component example pages embed.

- [ ] **Step 1: Write the failing test**

`src/lib/components/NetworkEditor.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import NetworkEditor from './NetworkEditor.svelte';

vi.mock('./BlockCanvas.svelte', () => ({
  default: () => null
}));

function editor() {
  const store = new NetworkStore();
  render(NetworkEditor, {
    props: { store, palette: ['linear', 'relu', 'sigmoid', 'softmax'] }
  });
  return store;
}

describe('NetworkEditor', () => {
  it('shows the palette, the shape table, and the issues panel', () => {
    editor();
    expect(screen.getByTestId('palette-linear')).toBeTruthy();
    expect(screen.getByTestId('shape-table')).toBeTruthy();
    expect(screen.getByTestId('issues-panel')).toBeTruthy();
  });

  it('adds a block when a palette entry is clicked', async () => {
    const store = editor();
    const before = store.network.blocks.length;
    await userEvent.click(screen.getByTestId('palette-relu'));
    expect(store.network.blocks.length).toBe(before + 1);
  });

  it('undoes and redoes through the toolbar', async () => {
    const store = editor();
    const before = store.network.blocks.length;
    await userEvent.click(screen.getByTestId('palette-relu'));
    await userEvent.click(screen.getByTestId('undo'));
    expect(store.network.blocks.length).toBe(before);
    await userEvent.click(screen.getByTestId('redo'));
    expect(store.network.blocks.length).toBe(before + 1);
  });

  it('resets the network', async () => {
    const store = editor();
    await userEvent.click(screen.getByTestId('palette-relu'));
    await userEvent.click(screen.getByTestId('reset-network'));
    expect(store.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
  });

  it('surfaces an automatic correction as a dismissible announcement', async () => {
    const store = editor();
    store.select(store.network.blocks[1].id);
    store.updateBlock(store.network.blocks[1].id, { units: 0 });
    await Promise.resolve();
    expect(screen.getByTestId('announcements').textContent).toContain('Units changed from 0 to 1');
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/components/NetworkEditor.test.ts`
Expected: FAIL — `Failed to resolve import "./NetworkEditor.svelte"`.

- [ ] **Step 3: Write `EditorToolbar.svelte`**

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let {
    store,
    onfit,
    onsave,
    onload,
    saving = false
  }: {
    store: NetworkStore;
    onfit?: () => void;
    onsave?: () => void;
    onload?: () => void;
    saving?: boolean;
  } = $props();

  function handleKeydown(event: KeyboardEvent): void {
    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier || event.key.toLowerCase() !== 'z') return;
    event.preventDefault();
    if (event.shiftKey) store.redo();
    else store.undo();
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  });
</script>

<div class="toolbar">
  <button type="button" data-testid="undo" disabled={!store.canUndo} onclick={() => store.undo()}>
    Undo
  </button>
  <button type="button" data-testid="redo" disabled={!store.canRedo} onclick={() => store.redo()}>
    Redo
  </button>
  <button type="button" data-testid="reset-network" onclick={() => store.reset()}>
    Reset network
  </button>

  {#if onfit}
    <button type="button" data-testid="fit-view" onclick={onfit}>Fit view</button>
  {/if}

  {#if onsave}
    <button type="button" data-testid="save-model" disabled={saving} onclick={onsave}>
      Save model
    </button>
  {/if}

  {#if onload}
    <button type="button" data-testid="load-model" disabled={saving} onclick={onload}>
      Load model
    </button>
  {/if}
</div>

<style>
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  button {
    padding: var(--space-1) var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
    font-size: var(--text-sm);
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
```

- [ ] **Step 4: Write `NetworkEditor.svelte`**

```svelte
<script lang="ts">
  import { onMount, type Component } from 'svelte';
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockKind } from '../network/types';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import BlockPalette from './BlockPalette.svelte';
  import EditorToolbar from './EditorToolbar.svelte';
  import InspectorPanel from './InspectorPanel.svelte';
  import IssuesPanel from './IssuesPanel.svelte';
  import ShapeTable from './ShapeTable.svelte';

  type CanvasProps = {
    store: NetworkStore;
    palette: BlockKind[];
    ondragover: (kind: BlockKind | null) => void;
  };

  let {
    store,
    palette,
    onsave,
    onload,
    saving = false
  }: {
    store: NetworkStore;
    palette: BlockKind[];
    onsave?: () => void;
    onload?: () => void;
    saving?: boolean;
  } = $props();

  let dragging = $state<BlockKind | null>(null);
  let Canvas = $state<Component<CanvasProps> | null>(null);

  onMount(async () => {
    const module = await import('./BlockCanvas.svelte');
    Canvas = module.default as Component<CanvasProps>;
  });
</script>

<section class="editor" data-testid="network-editor">
  <EditorToolbar {store} {onsave} {onload} {saving} />

  {#if store.announcements.length > 0}
    <div class="announcements" data-testid="announcements">
      <ul>
        {#each store.announcements as announcement, index (index)}
          <li>{announcement}</li>
        {/each}
      </ul>
      <button type="button" onclick={() => store.dismissAnnouncements()}>Dismiss</button>
    </div>
  {/if}

  <div class="body">
    <aside class="left">
      <BlockPalette
        {palette}
        onadd={(kind) => store.addBlock(kind)}
        ondragstart={(kind, event) => {
          dragging = kind;
          event.dataTransfer?.setData('application/visnet-block', kind);
        }}
      />
      <p class="drag-hint">
        {dragging
          ? `Drop ${dragging} onto the canvas to place it.`
          : `Click a block to add it, or drag it onto the canvas. ${BLOCK_DESCRIPTIONS.linear}`}
      </p>
    </aside>

    <div class="middle">
      {#if Canvas}
        <Canvas {store} {palette} ondragover={(kind) => (dragging = kind)} />
      {/if}
    </div>

    <aside class="right">
      <InspectorPanel {store} />
      <ShapeTable {store} />
    </aside>
  </div>

  <IssuesPanel {store} />
</section>

<style>
  .editor {
    display: grid;
    gap: var(--space-3);
  }

  .body {
    display: grid;
    gap: var(--space-3);
    grid-template-columns: minmax(180px, 220px) minmax(320px, 1fr) minmax(240px, 300px);
    align-items: start;
  }

  @media (max-width: 1100px) {
    .body {
      grid-template-columns: 1fr;
    }
  }

  .left,
  .right {
    display: grid;
    gap: var(--space-3);
  }

  .middle {
    min-height: 360px;
    height: 480px;
  }

  .drag-hint {
    margin: 0;
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .announcements {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-accent) 12%, var(--color-surface));
    border: 1px solid var(--color-accent);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .announcements ul {
    margin: 0;
    padding-left: var(--space-4);
  }

  .announcements button {
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    cursor: pointer;
  }
</style>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run src/lib/components/NetworkEditor.test.ts`
Expected: PASS. If mocking `BlockCanvas.svelte` does not work with the installed Svelte plugin, replace the mock with a real render and assert on the canvas test id instead — and say which you did in your report.

- [ ] **Step 6: Verify**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 7: Commit**

```bash
git add src/lib/components/EditorToolbar.svelte src/lib/components/NetworkEditor.svelte src/lib/components/NetworkEditor.test.ts
git commit -m "feat: add the editor toolbar and the embeddable network editor"
```

---

### Task 14: Persistence

**Files:**
- Create: `src/lib/persist/storage.ts`
- Create: `src/lib/persist/weights.ts`
- Test: `src/lib/persist/storage.test.ts`
- Test: `src/lib/persist/weights.test.ts`

**Interfaces:**
- Consumes: `toJSON`, `fromJSON` from `../network/serialize`; `createEmptyNetwork` from `../network/factory`; `Network`, `TrainingConfig` from `../network/types`; `PointDataset` from `../data/points`; `@tensorflow/tfjs`.
- Produces:

```ts
// storage.ts
export const NETWORK_KEY = 'visnet:network:v1';
export const DATASET_KEY = 'visnet:mlp:dataset:v1';

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface NetworkStorage {
  saveNetwork(net: Network): void;
  loadNetwork(): Network | null;
  saveDataset(dataset: PointDataset): void;
  loadDataset(): PointDataset | null;
  clear(): void;
}

export function createStorage(backing: KeyValueStore): NetworkStorage;
export function createBrowserStorage(): NetworkStorage | null;

// weights.ts
export const WEIGHTS_URL = 'indexeddb://visnet/weights/main';
export function weightShapes(model: tf.LayersModel): number[][];
export function shapesMatch(a: number[][], b: number[][]): boolean;
export function saveWeights(model: tf.LayersModel): Promise<void>;
export function loadWeightsInto(model: tf.LayersModel): Promise<boolean>;
```

**Behaviour that matters:**
- `createStorage` takes an injected `KeyValueStore` so it is testable in Node with an in-memory fake. `createBrowserStorage` returns `null` when `localStorage` is unavailable or throws on write (private browsing, quota), so the page can degrade to in-memory state and tell the user rather than crashing.
- `loadNetwork` returns `null` for a missing key, corrupt JSON, or an unsupported version, by delegating to `fromJSON`. The caller falls back to `createEmptyNetwork()` and tells the user the saved network could not be read.
- `saveDataset` and `loadDataset` are defensive in the same way: a dataset that is not an array of `{x, y, label}` points with finite coordinates and labels `0` or `1` is treated as absent.
- `weightShapes` returns each weight tensor's shape as a plain array, and `shapesMatch` compares two such lists. Together they answer "can these saved weights be loaded into this model?" without touching IndexedDB, which is what makes the decision testable in Node.
- `saveWeights` writes to IndexedDB through `model.save(WEIGHTS_URL)` and rethrows nothing: it resolves on success and rejects on failure so the caller can show a notice. `loadWeightsInto` returns `false` when no saved weights exist, when the shapes do not match, or when loading fails, and `true` when it has successfully applied them with `model.setWeights`. It must dispose any model it loads for comparison.
- `src/lib/persist/weights.ts` is an approved fifth TensorFlow.js import site. Record in your report that `AGENTS.md`'s allow-list needs it (Task 16 adds it).

- [ ] **Step 1: Write the failing tests**

`src/lib/persist/storage.test.ts`:

```ts
import { beforeEach, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { PointDataset } from '../data/points';
import { createStorage, DATASET_KEY, NETWORK_KEY, type KeyValueStore } from './storage';

function fakeStore(): KeyValueStore & { entries: Map<string, string> } {
  const entries = new Map<string, string>();
  return {
    entries,
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => void entries.set(key, value),
    removeItem: (key) => void entries.delete(key)
  };
}

let backing: ReturnType<typeof fakeStore>;

beforeEach(() => {
  backing = fakeStore();
});

describe('network storage', () => {
  it('round-trips a network', () => {
    const storage = createStorage(backing);
    const network = createEmptyNetwork();
    storage.saveNetwork(network);
    expect(storage.loadNetwork()).toEqual(network);
    expect(backing.entries.has(NETWORK_KEY)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });

  it('returns null for corrupt data', () => {
    backing.setItem(NETWORK_KEY, '{not json');
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });

  it('returns null for an unsupported version', () => {
    backing.setItem(NETWORK_KEY, JSON.stringify({ version: 99, network: createEmptyNetwork() }));
    expect(createStorage(backing).loadNetwork()).toBeNull();
  });
});

describe('dataset storage', () => {
  const dataset: PointDataset = {
    points: [
      { x: 0.1, y: -0.2, label: 0 },
      { x: -0.3, y: 0.4, label: 1 }
    ],
    numClasses: 2
  };

  it('round-trips a dataset', () => {
    const storage = createStorage(backing);
    storage.saveDataset(dataset);
    expect(storage.loadDataset()).toEqual(dataset);
    expect(backing.entries.has(DATASET_KEY)).toBe(true);
  });

  it('returns null when nothing is saved', () => {
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects a payload that is not a list of points', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: 'a', y: 0, label: 0 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects a label that is not 0 or 1', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: 0, y: 0, label: 7 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('rejects non-finite coordinates', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [{ x: null, y: 0, label: 0 }] }));
    expect(createStorage(backing).loadDataset()).toBeNull();
  });

  it('accepts an empty dataset', () => {
    backing.setItem(DATASET_KEY, JSON.stringify({ points: [] }));
    expect(createStorage(backing).loadDataset()).toEqual({ points: [], numClasses: 2 });
  });
});

describe('clear', () => {
  it('removes both entries', () => {
    const storage = createStorage(backing);
    storage.saveNetwork(createEmptyNetwork());
    storage.saveDataset({ points: [], numClasses: 2 });
    storage.clear();
    expect(backing.entries.size).toBe(0);
  });
});
```

`src/lib/persist/weights.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { shapesMatch, weightShapes } from './weights';

let models: tf.Sequential[] = [];

function model(units = 8): tf.Sequential {
  const net = createEmptyNetwork();
  const hidden = net.blocks[1];
  if (hidden.kind === 'linear') hidden.units = units;
  const built = buildModel(net);
  models.push(built);
  return built;
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((built) => built.dispose());
  models = [];
});

describe('weightShapes', () => {
  it('reports every weight tensor shape', () => {
    expect(weightShapes(model())).toEqual([
      [2, 8],
      [8],
      [8, 2],
      [2]
    ]);
  });

  it('changes when the architecture changes', () => {
    expect(weightShapes(model(16))[0]).toEqual([2, 16]);
  });
});

describe('shapesMatch', () => {
  it('accepts identical shapes', () => {
    expect(shapesMatch(weightShapes(model()), weightShapes(model()))).toBe(true);
  });

  it('rejects different shapes', () => {
    expect(shapesMatch(weightShapes(model()), weightShapes(model(16)))).toBe(false);
  });

  it('rejects different lengths', () => {
    expect(shapesMatch([[2, 8]], [])).toBe(false);
  });

  it('rejects a mismatched dimension inside a tensor', () => {
    expect(shapesMatch([[2, 8]], [[2, 9]])).toBe(false);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/lib/persist`
Expected: FAIL — both imports unresolvable.

- [ ] **Step 3: Write `storage.ts`**

```ts
import type { PointDataset } from '../data/points';
import type { Network } from '../network/types';
import { fromJSON, toJSON } from '../network/serialize';

export const NETWORK_KEY = 'visnet:network:v1';
export const DATASET_KEY = 'visnet:mlp:dataset:v1';

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface NetworkStorage {
  saveNetwork(net: Network): void;
  loadNetwork(): Network | null;
  saveDataset(dataset: PointDataset): void;
  loadDataset(): PointDataset | null;
  clear(): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isPoint(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (typeof value.x !== 'number' || !Number.isFinite(value.x)) return false;
  if (typeof value.y !== 'number' || !Number.isFinite(value.y)) return false;
  return value.label === 0 || value.label === 1;
}

function parseDataset(raw: string): PointDataset | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || !Array.isArray(parsed.points)) return null;
    if (!parsed.points.every(isPoint)) return null;
    return { points: parsed.points as PointDataset['points'], numClasses: 2 };
  } catch {
    return null;
  }
}

export function createStorage(backing: KeyValueStore): NetworkStorage {
  return {
    saveNetwork(net) {
      backing.setItem(NETWORK_KEY, toJSON(net));
    },
    loadNetwork() {
      const raw = backing.getItem(NETWORK_KEY);
      return raw === null ? null : fromJSON(raw);
    },
    saveDataset(dataset) {
      backing.setItem(DATASET_KEY, JSON.stringify({ points: dataset.points }));
    },
    loadDataset() {
      const raw = backing.getItem(DATASET_KEY);
      return raw === null ? null : parseDataset(raw);
    },
    clear() {
      backing.removeItem(NETWORK_KEY);
      backing.removeItem(DATASET_KEY);
    }
  };
}

export function createBrowserStorage(): NetworkStorage | null {
  try {
    const probe = '__visnet_probe__';
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return createStorage(window.localStorage);
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Write `weights.ts`**

```ts
import * as tf from '@tensorflow/tfjs';

export const WEIGHTS_URL = 'indexeddb://visnet/weights/main';

export function weightShapes(model: tf.LayersModel): number[][] {
  return model.getWeights().map((tensor) => [...tensor.shape]);
}

export function shapesMatch(a: number[][], b: number[][]): boolean {
  if (a.length !== b.length) return false;
  return a.every((shape, index) => {
    const other = b[index];
    if (shape.length !== other.length) return false;
    return shape.every((dimension, position) => dimension === other[position]);
  });
}

export async function saveWeights(model: tf.LayersModel): Promise<void> {
  await model.save(WEIGHTS_URL);
}

export async function loadWeightsInto(model: tf.LayersModel): Promise<boolean> {
  let saved: tf.LayersModel | null = null;
  try {
    saved = await tf.loadLayersModel(WEIGHTS_URL);
    if (!shapesMatch(weightShapes(model), weightShapes(saved))) return false;
    model.setWeights(saved.getWeights());
    return true;
  } catch {
    return false;
  } finally {
    saved?.dispose();
  }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/lib/persist`
Expected: PASS.

- [ ] **Step 6: Verify and commit**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: all projects green.

```bash
git add src/lib/persist
git commit -m "feat: add browser persistence for the network, dataset, and weights"
```

---

### Task 15: Example page and routes

**Files:**
- Create: `src/lib/examples/mlp/runtime.ts`
- Create: `src/lib/components/ExampleLayout.svelte`
- Create: `src/routes/examples/mlp/+page.svelte`
- Create: `src/routes/examples/cnn/+page.svelte`
- Modify: `src/routes/+page.svelte`

**Interfaces:**
- Consumes: `NetworkStore`, `DatasetStore`, `NetworkEditor`, `TrainingPanel`, `LossChart`, `StatsReadout`, `DecisionBoundary`, `createBrowserStorage`, `TrainStats`, `GENERATOR_NAMES`, `GENERATOR_DESCRIPTIONS`, `CLASS_LABELS`, `MLP_PALETTE`.
- Produces:
  - `runtime.ts` — a prerender-safe facade that loads TensorFlow.js and the engine's TensorFlow.js modules with dynamic `import()`, exactly once, after mount. All types come from `import type`, which is erased. Exports:

```ts
export type Model = tf.LayersModel;
export type ModelData = { xs: tf.Tensor2D; ys: tf.Tensor2D };
export type TrainerHandle = import('../../training/Trainer').Trainer;

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
}

export async function loadRuntime(): Promise<Runtime>;
```

The facade's methods are **synchronous** (except the two weight calls, which touch IndexedDB). That is the point: it lets the page's effects be plain synchronous functions, which is what makes the lifecycle correct.

  - `ExampleLayout.svelte` — props `{ title: string; intro: string; editor: Snippet; experiment: Snippet }`, rendering a two-column shell (editor on the left, experiment on the right) that collapses to one column on narrow screens.
  - `/examples/mlp` — the complete experience.
  - `/examples/cnn` — a placeholder naming what is coming.
  - `/` — a landing page listing the examples.

**The MLP page's model and trainer lifecycle** is the heart of this task:
- The page must not import TensorFlow.js or `render/boundary` statically; go through `runtime.ts` and `DecisionBoundary`.
- An `$effect` rebuilds the model whenever the **architecture** changes. Detect architecture change with a signature: `JSON.stringify(store.network.blocks)`. On rebuild, dispose the previous model and the previous trainer, reset the loss points and stats, and clear `playing`.
- A second `$effect` recompiles, without rebuilding, whenever only the **training** configuration changes: `JSON.stringify(store.network.training)`. This preserves the learned weights.
- A third `$effect` rebuilds the dataset tensors whenever `datasetStore.dataset` changes, disposing the previous tensors in its cleanup.
- The model is only built when `store.isValid`; otherwise it stays `null` and the page passes a caption to the boundary explaining that the network needs fixing first.
- `Trainer` is created when both a model and tensors exist, and disposed in the effect cleanup. `onStats` appends the epoch mean loss to a rolling list of at most 200 values and stores the latest stats. `onError` sets a plain-language banner.
- Play, pause, step, and reset are wired to the trainer; reset disposes and rebuilds the model, which resets the weights.
- Persistence: on mount, `createBrowserStorage()`; load the saved network and dataset when present, otherwise keep the defaults; if the saved network could not be read, announce it. Autosave the network and dataset on change, debounced by 500 ms. If storage is unavailable, announce that changes will not be saved.
- The page owns `expectedClasses = 2` on the store and passes `MLP_PALETTE` to the editor.

- [ ] **Step 1: Write `runtime.ts`**

```ts
import type { PointDataset } from '../../data/points';
import type { Network, TrainingConfig } from '../../network/types';
import type { TrainStats } from '../../training/Trainer';
import type * as tf from '@tensorflow/tfjs';

export type Model = tf.LayersModel;
export type ModelData = {
  xs: tf.Tensor2D;
  ys: tf.Tensor2D;
};
export type TrainerHandle = import('../../training/Trainer').Trainer;

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
}

export async function loadRuntime(): Promise<Runtime> {
  const [builder, tensors, trainerModule, weights] = await Promise.all([
    import('../../tf/buildModel'),
    import('../../data/tensors'),
    import('../../training/Trainer'),
    import('../../persist/weights')
  ]);

  return {
    buildModel: (net) => builder.buildModel(net),
    compileModel: (model, training) => builder.compileModel(model, training),
    toTensors: (dataset) => tensors.toTensors(dataset),
    disposeData: (data) => {
      data?.xs.dispose();
      data?.ys.dispose();
    },
    disposeModel: (model) => {
      model?.dispose();
    },
    createTrainer: (model, data, batchSize, onStats, onError) =>
      new trainerModule.Trainer(model, data, batchSize, onStats, onError),
    saveWeights: (model) => weights.saveWeights(model),
    loadWeightsInto: (model) => weights.loadWeightsInto(model)
  };
}
```

Every module that touches TensorFlow.js is loaded **once**, after mount, and the facade's methods are synchronous. This matters: it lets the page's `$effect`s be plain synchronous functions with no `await` and no cancellation flags, which removes a whole class of ordering bugs where a cleanup cancels an in-flight build. Do not reintroduce async work inside the page's effects.

- [ ] **Step 2: Write `ExampleLayout.svelte`**

```svelte
<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    title,
    intro,
    editor,
    experiment
  }: {
    title: string;
    intro: string;
    editor: Snippet;
    experiment: Snippet;
  } = $props();
</script>

<main class="example">
  <header>
    <h1>{title}</h1>
    <p>{intro}</p>
  </header>

  <div class="columns">
    <section class="editor">{@render editor()}</section>
    <section class="experiment">{@render experiment()}</section>
  </div>
</main>

<style>
  .example {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-5);
    max-width: 1500px;
    margin: 0 auto;
  }

  header h1 {
    margin: 0 0 var(--space-2);
    font-size: var(--text-xl);
  }

  header p {
    margin: 0;
    color: var(--color-text-muted);
    max-width: 70ch;
  }

  .columns {
    display: grid;
    gap: var(--space-4);
    grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
    align-items: start;
  }

  @media (max-width: 1200px) {
    .columns {
      grid-template-columns: 1fr;
    }
  }

  .experiment {
    display: grid;
    gap: var(--space-3);
    position: sticky;
    top: var(--space-4);
  }
</style>
```

- [ ] **Step 3: Write `/examples/mlp/+page.svelte`**

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import DecisionBoundary from '$lib/components/DecisionBoundary.svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import StatsReadout from '$lib/components/StatsReadout.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import { GENERATOR_DESCRIPTIONS, GENERATOR_NAMES } from '$lib/data/points';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import { DatasetStore } from '$lib/examples/mlp/datasetStore.svelte';
  import { CLASS_LABELS, MLP_PALETTE } from '$lib/examples/mlp/example';
  import {
    loadRuntime,
    type Model,
    type ModelData,
    type Runtime,
    type TrainerHandle
  } from '$lib/examples/mlp/runtime';
  import { createBrowserStorage, type NetworkStorage } from '$lib/persist/storage';
  import type { TrainStats } from '$lib/training/Trainer';

  const store = new NetworkStore();
  const datasetStore = new DatasetStore();
  store.expectedClasses = 2;

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
  let playing = $state(false);
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let builtSignature = $state('');

  let currentModel: Model | null = null;
  let data: ModelData | null = null;
  let trainer: TrainerHandle | null = null;
  let storage: NetworkStorage | null = null;
  let compiledTraining = '';

  const architecture = $derived(JSON.stringify(store.network.blocks));
  const trainingSignature = $derived(JSON.stringify(store.network.training));

  function handleStats(next: TrainStats): void {
    stats = next;
    if (next.epochMeanLoss !== null) {
      lossPoints = [...lossPoints, next.epochMeanLoss].slice(-200);
    }
  }

  function handleError(error: unknown): void {
    playing = false;
    banner = `Training stopped: ${error instanceof Error ? error.message : 'unknown error'}. Try fixing the network or resetting the model.`;
  }

  function releaseTrainer(): void {
    trainer?.dispose();
    trainer = null;
    playing = false;
  }

  onMount(async () => {
    runtime = await loadRuntime();
    storage = createBrowserStorage();
    if (!storage) {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
      return;
    }
    const savedNetwork = storage.loadNetwork();
    if (savedNetwork) store.load(savedNetwork);
    const savedDataset = storage.loadDataset();
    if (savedDataset) datasetStore.dataset = savedDataset;
  });

  $effect(() => {
    const api = runtime;
    if (!api) return;
    if (architecture === builtSignature) return;
    builtSignature = architecture;
    compiledTraining = trainingSignature;

    releaseTrainer();
    stats = null;
    lossPoints = [];
    api.disposeModel(currentModel);
    currentModel = null;
    model = null;

    if (!store.isValid) return;
    const built = api.buildModel(store.network);
    currentModel = built;
    model = built;
  });

  $effect(() => {
    const api = runtime;
    const training = trainingSignature;
    if (!api || !currentModel || training === compiledTraining) return;
    compiledTraining = training;
    api.compileModel(currentModel, store.network.training);
  });

  $effect(() => {
    const api = runtime;
    const dataset = datasetStore.dataset;
    const currentModelRef = model;
    if (!api) return;

    const next = api.toTensors(dataset);
    api.disposeData(data);
    data = next;
    releaseTrainer();

    if (!currentModelRef || !store.isValid || next.xs.shape[0] === 0) return;
    trainer = api.createTrainer(
      currentModelRef,
      next,
      store.network.training.batchSize,
      handleStats,
      handleError
    );
  });

  $effect(() => {
    const net = store.network;
    const dataset = datasetStore.dataset;
    const currentStorage = storage;
    if (!currentStorage) return;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(net);
        currentStorage.saveDataset(dataset);
      } catch {
        banner = 'Your work could not be saved. The browser storage may be full.';
      }
    }, 500);
    return () => clearTimeout(timer);
  });

  async function play(): Promise<void> {
    if (!trainer) return;
    playing = true;
    await trainer.play();
    playing = false;
  }

  function pause(): void {
    trainer?.pause();
    playing = false;
  }

  function step(): void {
    void trainer?.step();
  }

  function resetModel(): void {
    releaseTrainer();
    runtime?.disposeModel(currentModel);
    currentModel = null;
    model = null;
    stats = null;
    lossPoints = [];
    builtSignature = '';
  }

  async function save(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      await api.saveWeights(currentModel);
      banner = 'Model saved in this browser.';
    } catch {
      banner = 'The model could not be saved in this browser.';
    } finally {
      saving = false;
    }
  }

  async function load(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      const loaded = await api.loadWeightsInto(currentModel);
      banner = loaded
        ? 'Saved weights loaded.'
        : 'No saved weights match this network. Train and save again.';
    } finally {
      saving = false;
    }
  }
</script>

<ExampleLayout
  title="Points in 2D"
  intro="Build a small network, train it on coloured points, and watch the boundary between the two classes take shape."
>
  {#snippet editor()}
    <NetworkEditor {store} palette={MLP_PALETTE} onsave={save} onload={load} {saving} />
  {/snippet}

  {#snippet experiment()}
    {#if banner}
      <p class="banner" role="status">{banner}</p>
    {/if}

    <div class="dataset">
      <h2>Data</h2>
      <label>
        <span>Pattern</span>
        <select
          data-testid="dataset-generator"
          value={datasetStore.generator}
          onchange={(event) => {
            datasetStore.generator = event.currentTarget.value as (typeof GENERATOR_NAMES)[number];
            datasetStore.regenerate();
          }}
        >
          {#each GENERATOR_NAMES as name (name)}
            <option value={name}>{name}</option>
          {/each}
        </select>
        <small>{GENERATOR_DESCRIPTIONS[datasetStore.generator]}</small>
      </label>

      <label>
        <span>Points</span>
        <input
          type="number"
          min="10"
          max="2000"
          data-testid="dataset-count"
          value={datasetStore.pointCount}
          onchange={(event) => {
            const value = Number(event.currentTarget.value);
            if (Number.isFinite(value) && value >= 10) {
              datasetStore.pointCount = value;
              datasetStore.regenerate();
            }
          }}
        />
      </label>

      <div class="dataset-actions">
        <button type="button" onclick={() => datasetStore.reseed()}>New sample</button>
        <button type="button" onclick={() => datasetStore.clear()}>Clear points</button>
      </div>

      <fieldset>
        <legend>Point to add when clicking</legend>
        {#each [0, 1] as label (label)}
          <label class="inline">
            <input
              type="radio"
              name="label"
              value={label}
              checked={datasetStore.selectedLabel === label}
              onchange={() => datasetStore.selectLabel(label as 0 | 1)}
            />
            <span>{CLASS_LABELS[label]}</span>
          </label>
        {/each}
      </fieldset>
    </div>

    <DecisionBoundary
      {model}
      dataset={datasetStore.dataset}
      selectedLabel={datasetStore.selectedLabel}
      onaddpoint={(x, y) => datasetStore.addPoint(x, y)}
      caption={store.isValid
        ? 'Each coloured area is the class the network predicts at that spot. Click to add a point.'
        : 'Fix the problems listed in the editor before the boundary can be drawn.'}
    />

    <TrainingPanel
      {store}
      {playing}
      disabled={!store.isValid}
      onplay={play}
      onpause={pause}
      onstep={step}
      onreset={resetModel}
    />

    <LossChart points={lossPoints} />
    <StatsReadout {stats} />
  {/snippet}
</ExampleLayout>

<style>
  .banner {
    margin: 0;
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-warning) 12%, var(--color-surface));
    border: 1px solid var(--color-warning);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .dataset {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  label.inline {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  select,
  input[type='number'] {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  fieldset {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2);
    display: flex;
    gap: var(--space-3);
    margin: 0;
  }

  legend {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .dataset-actions {
    display: flex;
    gap: var(--space-2);
  }

  .dataset-actions button {
    flex: 1;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }
</style>
```

Three points about this script that are load-bearing, so do not "simplify" them away:

1. `runtime` and `model` use `$state.raw`, not `$state`. Plain `$state` deep-proxies its value, which would wrap the TensorFlow.js model and tensor objects in a Proxy and break them.
2. `builtSignature` is `$state` (not a plain variable) so that `resetModel` can force a rebuild by clearing it. `currentModel`, `data`, `trainer`, `storage`, and `compiledTraining` are deliberately plain variables: they must not become effect dependencies, or the effects would loop.
3. The effects are synchronous and have no cleanup, so nothing can cancel an in-flight build. All asynchronous work lives in `onMount`, in the event handlers, and in the `Runtime` facade.

- [ ] **Step 4: Write `/examples/cnn/+page.svelte`**

```svelte
<main>
  <h1>Handwritten digits (coming next)</h1>
  <p>
    This example will train a small convolutional network on MNIST-style digit images. The engine
    already supports convolution, flatten, and the shape inference the editor needs; what is missing
    is this page.
  </p>
  <p><a href="/examples/mlp">Try the 2D points example instead</a></p>
</main>

<style>
  main {
    max-width: 70ch;
    margin: 0 auto;
    padding: var(--space-5);
  }

  h1 {
    font-size: var(--text-xl);
  }

  p {
    color: var(--color-text-muted);
  }
</style>
```

- [ ] **Step 5: Write `/`**

```svelte
<main>
  <h1>VisNet</h1>
  <p>Build neural networks from visual blocks and watch them learn.</p>

  <ul>
    <li>
      <a href="/examples/mlp">Points in 2D</a>
      <span>Classify coloured points and see the decision boundary.</span>
    </li>
    <li>
      <a href="/examples/cnn">Handwritten digits</a>
      <span>Coming next: convolutional networks on digit images.</span>
    </li>
  </ul>
</main>

<style>
  main {
    max-width: 70ch;
    margin: 0 auto;
    padding: var(--space-5);
  }

  h1 {
    font-size: var(--text-xl);
  }

  p {
    color: var(--color-text-muted);
  }

  ul {
    list-style: none;
    padding: 0;
    display: grid;
    gap: var(--space-3);
  }

  li {
    display: grid;
    gap: var(--space-1);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  a {
    color: var(--color-accent);
    font-weight: 600;
  }

  li span {
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
</style>
```

- [ ] **Step 6: Verify**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: all projects green.

Run: `npm run build`
Expected: build succeeds and all three routes are prerendered. Confirm `build/examples/mlp/index.html` exists.

Run: `npm run dev`, then open `/examples/mlp` and work through the manual checklist in Task 16.

- [ ] **Step 7: Commit**

```bash
git add src/lib/examples/mlp/runtime.ts src/lib/components/ExampleLayout.svelte src/routes
git commit -m "feat: add the MLP example page and the example routes"
```

---

### Task 16: Documentation and final verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything built in Tasks 1-15.
- Produces: documentation that matches the shipped app, and a full green verification run including the manual checklist.

- [ ] **Step 1: Update `AGENTS.md`**

Update the `## Status` section to:

```markdown
## Status

Engine complete and unit-tested, and the editor UI and the 2D points example are
built on top of it. A user can drag blocks onto a canvas, edit them, train, and
watch a live decision boundary, and their network and dataset are restored on
reload. The convolutional (MNIST) example page is a placeholder.

Component tests run in jsdom; engine tests run in Node. See `npm test`.
```

Update the `## Architecture rules` list so that the TensorFlow.js allow-list reads:

```markdown
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`,
  `src/lib/training/**`, `src/lib/data/tensors.ts`,
  `src/lib/render/boundary.ts` (the only render module that runs a forward pass),
  `src/lib/persist/weights.ts`, and the test files colocated with those modules.
  No other production module may import it.
```

Add these two rules after it:

```markdown
- TensorFlow.js and `@xyflow/svelte` must only be reached from the browser. Import
  them dynamically (or inside an `onMount`/`$effect`) so `npm run build` can
  prerender every route with SSR on.
- `src/lib/editor/flow.ts` and `src/lib/editor/history.ts` stay free of Svelte,
  TensorFlow.js, DOM, and `@xyflow/svelte`.
```

Update the `## Commands` section to add:

```markdown
- `npm run test:watch` — Vitest in watch mode
```

Confirm every command listed exists in `package.json`, and that no script is listed that does not exist.

- [ ] **Step 2: Update `README.md`**

Replace the `## Architecture` and `## Development` sections with:

```markdown
## What works today

- Build a network from blocks on a canvas: add, delete, reorder, and connect them.
- See the tensor shape at every point in the pipeline, on each block and on each wire,
  with a table of the whole network and its parameter counts.
- Train on generated 2D points or points you click onto the canvas, and watch the
  decision boundary change as the loss falls.
- Undo and redo every edit.
- Your network, dataset, and saved weights are restored when you reload.

The convolutional (MNIST) example is not built yet.

## Architecture

The engine is framework-independent. `src/lib/network/` defines networks, infers
the tensor shape flowing through every block, and reports validation problems as
plain-language errors with a suggested fix. TensorFlow.js is confined to
`src/lib/tf/`, `src/lib/training/`, `src/lib/data/tensors.ts`,
`src/lib/render/boundary.ts`, and `src/lib/persist/weights.ts`.

The editor is a projection of that model, never a second copy of it: the canvas
draws the network and turns gestures back into operations on it.

## Development

npm run dev         # development server
npm run build       # static build
npm run check       # type checking
npm run lint        # linting
npm test            # unit and component tests
```

- [ ] **Step 3: Run the full verification suite**

Run: `npm run lint`
Expected: no errors.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: both projects green, 0 failures, no warnings in the output.

Run: `npm run build`
Expected: build succeeds; `build/index.html`, `build/examples/mlp/index.html`, and `build/examples/cnn/index.html` all exist.

If any command fails, fix the cause. Do not weaken or delete a test to make it pass.

- [ ] **Step 4: Work through the manual checklist**

Run `npm run dev` and open `/examples/mlp`. Record the outcome of each item in your report, quoting any console output.

1. The default network renders as a left-to-right pipeline with `in → out` shape badges on every block and shape labels on every wire.
2. The shape table lists the whole pipeline with correct shapes and parameter counts, and the total reads 42.
3. Clicking a palette block adds it after the selected block; dragging one onto the canvas places it near where it was dropped.
4. Deleting a block works; deleting the input or output is impossible.
5. Dragging a wire from one block's output to another block's input reorders the chain, and dragging it into an illegal configuration changes nothing.
6. `Ctrl/Cmd+Z` and `Ctrl/Cmd+Shift+Z` undo and redo, and the toolbar buttons enable and disable correctly.
7. Hovering a block shows its description, shapes, and parameter count.
8. Changing the Input shape so a stored kernel size becomes invalid clamps the value and shows an announcement.
9. Each generator produces a visibly different point pattern; "New sample" changes the sample; "Clear points" empties the canvas.
10. Clicking the canvas adds a point of the selected class, and clicking outside the drawn area still lands inside the domain.
11. Pressing Play descends the loss curve and visibly changes the boundary; Pause stops it; Step advances one batch; Reset model clears the curve.
12. Making the network invalid disables training and shows an error naming the block and the fix.
13. Reloading restores the network and dataset; after Save model, Load model restores the trained weights and the boundary redraws.
14. Resizing the window and navigating between routes produces no console errors, and in particular no TensorFlow.js "not disposed" or WebGL context warnings.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: document the editor, the example, and persistence"
```

---

## Plan Self-Review

**Spec coverage.** Section 2 (UX principles): shapes on nodes, wires, inspector, and a table (Tasks 5, 7, 8); no block stores an input dimension (engine, unchanged); descriptions everywhere (Tasks 5, 7, 10); errors as instructions (Task 8); no dead ends (Tasks 8, 10, 15 — invalid networks explain themselves rather than showing dead buttons); announced auto-corrections (Tasks 3, 4, 13); undo/redo (Tasks 4, 13); immediate feedback (derived shapes throughout). Section 7 (store and history): Task 4. Section 8 (canvas projection): Task 6, with the pure projection already in the engine. Section 9 (shape visibility and automatic input selection): Tasks 3, 5, 7, 8. Section 10 (architecture vs compilation): Task 15's separate effects. Section 11 (training): Tasks 9, 10, 15. Section 13 (boundary rendering): Task 11. Section 14 (loss chart and stats): Task 9. Section 15 (persistence): Task 14, with weights in IndexedDB. Section 16 (error handling): Tasks 8, 10, 11, 15. Section 17 (routes and embedding contract): Tasks 13, 15. Section 18 (testing): Task 1's harness plus the per-task tests. Section 19 (manual checklist): Task 16.

**Placeholder scan.** No "TBD", "TODO", "similar to Task N", or steps that describe work without showing it. Every code step carries complete code. Every command has an expected result.

**Type consistency.** `NetworkStore`'s surface is fixed in Task 4 and consumed unchanged by Tasks 5-8 and 13. `FlowNode['data']` is defined in the engine and extended with `onremove` only inside `BlockCanvas`. `TrainStats` is defined in the engine and consumed by Tasks 9 and 15. `parameterBounds` and `clampBlockPatch` are defined in Task 3 and used in Task 7. `dropIndexFor` and `insertionIndexFor` are defined in Task 2 and used in Tasks 4 and 6. `Model`, `ModelData`, and `TrainerHandle` are defined in Task 15's `runtime.ts` and used only there.

**Two deliberate scope decisions worth recording.**

1. `BlockCanvas` and `DecisionBoundary` have no automated tests. Both depend on APIs jsdom does not implement (Svelte Flow's layout and pointer handling, and the canvas 2D context). Their pure logic is tested where it lives — `toFlow` and `connectionToIntent` in the engine, `dropIndexFor` in Task 2, `clientToDomain` and `classesToRgba` in the engine — and the components themselves are covered by `npm run check`, `npm run build`, and the Task 16 checklist. `DecisionBoundary`'s click path is tested with a stubbed canvas context.
2. `src/lib/persist/weights.ts` becomes a fifth approved TensorFlow.js import site, because it is the serialization boundary for weights. `AGENTS.md` is updated in Task 16 to say so.

