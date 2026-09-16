# VisNet Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the VisNet engine — a SvelteKit scaffold, pure network-domain modules, a TensorFlow.js model builder, a trainer, a 2D point dataset, and a decision-boundary renderer — all covered by unit tests.

**Architecture:** `src/lib/network/` is pure (no Svelte, no TF.js, no DOM) and is the single source of truth for what a network is, what shape flows through each block, and whether it is valid. `src/lib/tf/` and `src/lib/training/` are the only modules that import TensorFlow.js. `src/lib/data/` and `src/lib/render/` are split into pure parts (tested in Node) and TF.js/DOM parts (verified manually). This plan deliberately stops before any UI; the editor and example page are Plan B.

**Tech Stack:** SvelteKit 2, Svelte 5 (runes), TypeScript strict, Vite, `@sveltejs/adapter-static`, Vitest, `@tensorflow/tfjs`.

## Global Constraints

- Package manager is **npm**. Do not use pnpm, yarn, or bun.
- TypeScript strict mode is on; `npm run check` must pass at the end of every task.
- `src/lib/network/**` must not import Svelte, `@tensorflow/tfjs`, or any DOM API. It must run in plain Node.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`, `src/lib/training/**`, and `src/lib/data/tensors.ts`.
- No backend, no server code. All routes are prerendered; the build output is static files.
- Every validation issue must carry a non-empty `title`, `message`, and `fix`.
- No block stores its input dimension; inputs are always derived from the previous block's output.
- Styling is plain CSS with design tokens in `src/lib/styles/tokens.css`. No CSS framework.
- Do not add code comments unless a non-obvious constraint requires one.
- Commit at the end of every task with the exact message shown in that task.
- Design reference: `docs/superpowers/specs/2026-09-16-visnet-design.md`.

## File Structure

| Path | Responsibility |
| --- | --- |
| `package.json`, `svelte.config.js`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `.prettierrc` | Toolchain configuration |
| `src/app.html`, `src/app.css`, `src/lib/styles/tokens.css` | HTML shell and design tokens |
| `src/routes/+layout.ts`, `src/routes/+layout.svelte`, `src/routes/+page.svelte` | Prerender flag, global shell, placeholder landing page |
| `src/lib/network/types.ts` | `Block`, `BlockKind`, `Network`, `TrainingConfig` |
| `src/lib/network/factory.ts` | Block and network constructors, defaults, id generation |
| `src/lib/network/descriptions.ts` | Plain-language text for every block kind and parameter |
| `src/lib/network/chain.ts` | Immutable chain operations |
| `src/lib/network/inferShapes.ts` | Per-block input/output shapes, edge shapes, parameter counts |
| `src/lib/network/validate.ts` | Error and warning rules with plain-language messages |
| `src/lib/network/serialize.ts` | Versioned JSON round-trip and migration seam |
| `src/lib/editor/flow.ts` | Pure projection of a network to canvas nodes/edges and connection intents |
| `src/lib/editor/history.ts` | Pure undo/redo stack |
| `src/lib/tf/buildModel.ts` | `Network` to `tf.Sequential`, compile, `NetworkInvalidError` |
| `src/lib/training/Trainer.ts` | Play/pause/step training loop with epoch statistics |
| `src/lib/data/rng.ts` | Seeded PRNG |
| `src/lib/data/points.ts` | 2D point dataset and generators (pure) |
| `src/lib/data/tensors.ts` | Dataset to `tf.Tensor2D` conversion |
| `src/lib/render/boundary.ts` | Grid sampling, coordinate mapping, RGBA rasterisation |
| `src/lib/render/palette.ts` | Class colours as hex and RGB triples |

---

### Task 1: Project scaffold, toolchain, and design tokens

**Files:**
- Create: `package.json`, `svelte.config.js`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `.prettierrc`
- Create: `src/app.html`, `src/app.css`, `src/lib/styles/tokens.css`
- Create: `src/routes/+layout.ts`, `src/routes/+layout.svelte`, `src/routes/+page.svelte`
- Test: `src/lib/styles/tokens.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `npm run dev` / `build` / `check` / `test` toolchain, and the CSS custom properties that every later UI task uses. Token names are fixed here: `--color-bg`, `--color-surface`, `--color-border`, `--color-text`, `--color-text-muted`, `--color-accent`, `--color-error`, `--color-warning`, `--color-success`, `--class-0`, `--class-1`, `--space-1` through `--space-6`, `--radius-sm`, `--radius-md`, `--font-sans`, `--font-mono`, `--text-xs`, `--text-sm`, `--text-base`, `--text-lg`, `--text-xl`.

- [ ] **Step 1: Install dependencies**

```bash
npm init -y
npm install @tensorflow/tfjs @xyflow/svelte
npm install -D @sveltejs/kit @sveltejs/adapter-static @sveltejs/vite-plugin-svelte svelte svelte-check typescript vite vitest @types/node eslint @eslint/js eslint-config-prettier eslint-plugin-svelte typescript-eslint prettier prettier-plugin-svelte globals
```

Expected: `added N packages` with no `ERR!`. If npm reports a peer-dependency conflict, re-run the failing command with `--legacy-peer-deps` and record why in the commit message.

- [ ] **Step 2: Write `package.json` scripts**

Replace the `scripts` block in `package.json` with:

```json
{
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
    "lint": "eslint .",
    "format": "prettier --write .",
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

Also ensure `package.json` contains `"type": "module"` and `"private": true`.

- [ ] **Step 3: Write the toolchain config files**

`svelte.config.js`:

```js
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter(),
    prerender: { entries: ['*'] }
  }
};
```

`vite.config.ts`:

```ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit()],
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts']
  }
});
```

`tsconfig.json`:

```json
{
  "extends": "./.svelte-kit/tsconfig.json",
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "sourceMap": true,
    "strict": true,
    "moduleResolution": "bundler"
  }
}
```

`eslint.config.js`:

```js
import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import prettier from 'eslint-config-prettier';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['build/', '.svelte-kit/', 'node_modules/', 'coverage/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...svelte.configs['flat/recommended'],
  prettier,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } }
  }
);
```

`.prettierrc`:

```json
{
  "singleQuote": true,
  "trailingComma": "none",
  "printWidth": 100,
  "plugins": ["prettier-plugin-svelte"],
  "overrides": [{ "files": "*.svelte", "options": { "parser": "svelte" } }]
}
```

- [ ] **Step 4: Write the app shell and tokens**

`src/app.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VisNet</title>
    %sveltekit.head%
  </head>
  <body data-sveltekit-preload-data="hover">
    <div style="display: contents">%sveltekit.body%</div>
  </body>
</html>
```

`src/lib/styles/tokens.css`:

```css
:root {
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-border: #cbd5e1;
  --color-text: #0f172a;
  --color-text-muted: #64748b;
  --color-accent: #6366f1;
  --color-error: #dc2626;
  --color-warning: #d97706;
  --color-success: #10b981;

  --class-0: #38bdf8;
  --class-1: #fb7185;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;

  --radius-sm: 4px;
  --radius-md: 8px;

  --font-sans: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace;

  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.25rem;
  --text-xl: 1.5rem;
}
```

`src/app.css`:

```css
@import './lib/styles/tokens.css';

*,
*::before,
*::after {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
}

body {
  background: var(--color-bg);
  color: var(--color-text);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.5;
}

button {
  font: inherit;
}

code,
pre {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
}
```

`src/routes/+layout.ts`:

```ts
export const prerender = true;
```

`src/routes/+layout.svelte`:

```svelte
<script lang="ts">
  import '../app.css';

  let { children } = $props();
</script>

{@render children()}
```

`src/routes/+page.svelte`:

```svelte
<main>
  <h1>VisNet</h1>
  <p>Build neural networks from visual blocks and watch them learn.</p>
</main>
```

- [ ] **Step 5: Write the failing test**

`src/lib/styles/tokens.test.ts`:

```ts
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const REQUIRED_TOKENS = [
  '--color-bg',
  '--color-surface',
  '--color-border',
  '--color-text',
  '--color-text-muted',
  '--color-accent',
  '--color-error',
  '--color-warning',
  '--color-success',
  '--class-0',
  '--class-1',
  '--space-1',
  '--space-6',
  '--radius-sm',
  '--radius-md',
  '--font-sans',
  '--font-mono',
  '--text-xs',
  '--text-xl'
];

describe('design tokens', () => {
  const css = readFileSync(fileURLToPath(new URL('./tokens.css', import.meta.url)), 'utf8');

  it.each(REQUIRED_TOKENS)('defines %s', (token) => {
    expect(css).toContain(`${token}:`);
  });
});
```

- [ ] **Step 6: Run the test**

Run: `npm test`
Expected: PASS, 19 tests, 1 file.

- [ ] **Step 7: Verify the toolchain**

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm run build`
Expected: build succeeds and `build/index.html` exists.

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold SvelteKit project with design tokens and vitest"
```

---

### Task 2: Domain types

**Files:**
- Create: `src/lib/network/types.ts`
- Test: `src/lib/network/types.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `BlockKind`, `BLOCK_KINDS`, `BlockBase`, `InputBlock`, `LinearBlock`, `Conv2dBlock`, `FlattenBlock`, `ActivationBlock`, `OutputBlock`, `Block`, `TrainingConfig`, `Network`. Every later task imports from this module. `BLOCK_KINDS` is the single runtime source of the eight block kinds; Tasks 3, 4, and 7 import it rather than re-declaring the list. The `Network` shape is exactly:

```ts
interface Network {
  version: 1;
  blocks: Block[];
  training: TrainingConfig;
}
```

- [ ] **Step 1: Write the failing test**

`src/lib/network/types.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { BLOCK_KINDS, type Block, type Network } from './types';

describe('BLOCK_KINDS', () => {
  it('lists the eight block kinds in pipeline order', () => {
    expect([...BLOCK_KINDS]).toEqual([
      'input',
      'linear',
      'conv2d',
      'flatten',
      'relu',
      'sigmoid',
      'softmax',
      'output'
    ]);
  });

  it('contains no duplicates', () => {
    expect(new Set(BLOCK_KINDS).size).toBe(BLOCK_KINDS.length);
  });
});

describe('domain types', () => {
  it('has a valid block representation for every kind', () => {
    const blocks: Block[] = [
      { id: 'a', kind: 'input', shape: [2] },
      { id: 'b', kind: 'linear', units: 8 },
      { id: 'c', kind: 'conv2d', filters: 8, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'd', kind: 'flatten' },
      { id: 'e', kind: 'relu' },
      { id: 'f', kind: 'sigmoid' },
      { id: 'g', kind: 'softmax' },
      { id: 'h', kind: 'output', units: 2 }
    ];

    const network: Network = {
      version: 1,
      blocks,
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
    };

    expect(new Set(network.blocks.map((block) => block.kind))).toEqual(new Set(BLOCK_KINDS));
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/types.test.ts`
Expected: FAIL — `Failed to resolve import "./types"`.

- [ ] **Step 3: Write the implementation**

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

export const BLOCK_KINDS: readonly BlockKind[] = [
  'input',
  'linear',
  'conv2d',
  'flatten',
  'relu',
  'sigmoid',
  'softmax',
  'output'
];

export interface BlockBase {
  id: string;
}

export interface InputBlock extends BlockBase {
  kind: 'input';
  shape: number[];
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
  units: number;
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

export interface Network {
  version: 1;
  blocks: Block[];
  training: TrainingConfig;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/types.test.ts`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/types.ts src/lib/network/types.test.ts
git commit -m "feat: add network domain types"
```

---

### Task 3: Block and network factory

**Files:**
- Create: `src/lib/network/factory.ts`
- Test: `src/lib/network/factory.test.ts`

**Interfaces:**
- Consumes: `Block`, `BlockKind`, `Network` from `./types`.
- Produces:
  - `newBlockId(): string`
  - `createBlock(kind: BlockKind): Block`
  - `createEmptyNetwork(): Network`
  - `cloneNetwork(net: Network): Network`
  - Defaults: `input` → `shape: [2]`; `linear` → `units: 8`; `conv2d` → `filters: 8, kernelSize: 3, stride: 1, padding: 'same'`; `output` → `units: 2`; `flatten`/`relu`/`sigmoid`/`softmax` → no parameters.
  - `createEmptyNetwork()` returns exactly six blocks in this order: `input([2])`, `linear(8)`, `relu`, `linear(2)`, `softmax`, `output(2)`, with training `{ loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }`.

- [ ] **Step 1: Write the failing test**

`src/lib/network/factory.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { cloneNetwork, createBlock, createEmptyNetwork, newBlockId } from './factory';
import { BLOCK_KINDS } from './types';

describe('createBlock', () => {
  it.each([...BLOCK_KINDS])('creates a %s block with a unique id', (kind) => {
    const a = createBlock(kind);
    const b = createBlock(kind);
    expect(a.kind).toBe(kind);
    expect(a.id).toBeTruthy();
    expect(a.id).not.toBe(b.id);
  });

  it('applies the documented defaults', () => {
    expect(createBlock('input')).toMatchObject({ kind: 'input', shape: [2] });
    expect(createBlock('linear')).toMatchObject({ kind: 'linear', units: 8 });
    expect(createBlock('conv2d')).toMatchObject({
      kind: 'conv2d',
      filters: 8,
      kernelSize: 3,
      stride: 1,
      padding: 'same'
    });
    expect(createBlock('output')).toMatchObject({ kind: 'output', units: 2 });
  });
});

describe('newBlockId', () => {
  it('returns a different id on every call', () => {
    const ids = new Set(Array.from({ length: 50 }, () => newBlockId()));
    expect(ids.size).toBe(50);
  });
});

describe('createEmptyNetwork', () => {
  it('builds the default MLP', () => {
    const net = createEmptyNetwork();
    expect(net.version).toBe(1);
    expect(net.blocks.map((b) => b.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
    expect(net.blocks[3]).toMatchObject({ kind: 'linear', units: 2 });
    expect(net.training).toEqual({
      loss: 'crossEntropy',
      optimizer: 'adam',
      learningRate: 0.01,
      batchSize: 32
    });
  });

  it('gives every block a distinct id', () => {
    const net = createEmptyNetwork();
    expect(new Set(net.blocks.map((b) => b.id)).size).toBe(net.blocks.length);
  });

  it('returns a fresh object each time', () => {
    const a = createEmptyNetwork();
    const b = createEmptyNetwork();
    expect(a).not.toBe(b);
    expect(a.blocks[0].id).not.toBe(b.blocks[0].id);
  });
});

describe('cloneNetwork', () => {
  it('deep-copies blocks so mutations do not leak', () => {
    const net = createEmptyNetwork();
    const copy = cloneNetwork(net);
    expect(copy).not.toBe(net);
    expect(copy.blocks).not.toBe(net.blocks);
    expect(copy.blocks[0]).not.toBe(net.blocks[0]);
    expect(copy.blocks[0]).toEqual(net.blocks[0]);
    expect(copy.training).not.toBe(net.training);
    expect(copy.training).toEqual(net.training);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/factory.test.ts`
Expected: FAIL — `Failed to resolve import "./factory"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/factory.ts`:

```ts
import type { Block, BlockKind, LinearBlock, Network, TrainingConfig } from './types';

const DEFAULT_TRAINING: TrainingConfig = {
  loss: 'crossEntropy',
  optimizer: 'adam',
  learningRate: 0.01,
  batchSize: 32
};

export function newBlockId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `b_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

export function createBlock(kind: BlockKind): Block {
  switch (kind) {
    case 'input':
      return { id: newBlockId(), kind: 'input', shape: [2] };
    case 'linear':
      return { id: newBlockId(), kind: 'linear', units: 8 };
    case 'conv2d':
      return {
        id: newBlockId(),
        kind: 'conv2d',
        filters: 8,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      };
    case 'flatten':
      return { id: newBlockId(), kind: 'flatten' };
    case 'relu':
    case 'sigmoid':
    case 'softmax':
      return { id: newBlockId(), kind };
    case 'output':
      return { id: newBlockId(), kind: 'output', units: 2 };
  }
}

export function createEmptyNetwork(): Network {
  const input = createBlock('input');
  const hidden = createBlock('linear');
  const activation = createBlock('relu');
  const outputLayer: LinearBlock = { ...(createBlock('linear') as LinearBlock), units: 2 };
  const softmax = createBlock('softmax');
  const output = createBlock('output');

  return {
    version: 1,
    blocks: [input, hidden, activation, outputLayer, softmax, output],
    training: { ...DEFAULT_TRAINING }
  };
}

export function cloneNetwork(net: Network): Network {
  return {
    version: net.version,
    blocks: net.blocks.map((block) => ({ ...block })),
    training: { ...net.training }
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/factory.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/factory.ts src/lib/network/factory.test.ts
git commit -m "feat: add block and network factory with defaults"
```

---

### Task 4: Block and parameter descriptions

**Files:**
- Create: `src/lib/network/descriptions.ts`
- Test: `src/lib/network/descriptions.test.ts`

**Interfaces:**
- Consumes: `BlockKind` from `./types`.
- Produces:
  - `BLOCK_DESCRIPTIONS: Record<BlockKind, string>`
  - `PARAM_DESCRIPTIONS: Record<ParamKey, string>` where
    `ParamKey = 'inputShape' | 'units' | 'filters' | 'kernelSize' | 'stride' | 'padding' | 'loss' | 'optimizer' | 'learningRate' | 'batchSize'`
  - `classifyInputShape(shape: number[]): 'flat' | 'image' | 'other'` — used by validation and the inspector to talk about the input shape in plain words.

- [ ] **Step 1: Write the failing test**

`src/lib/network/descriptions.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
  BLOCK_DESCRIPTIONS,
  PARAM_DESCRIPTIONS,
  classifyInputShape
} from './descriptions';
import { BLOCK_KINDS } from './types';

describe('BLOCK_DESCRIPTIONS', () => {
  it.each([...BLOCK_KINDS])('describes %s', (kind) => {
    expect(BLOCK_DESCRIPTIONS[kind]).toBeTruthy();
    expect(BLOCK_DESCRIPTIONS[kind].length).toBeGreaterThan(15);
  });

  it('has no extra keys', () => {
    expect(Object.keys(BLOCK_DESCRIPTIONS).sort()).toEqual([...BLOCK_KINDS].sort());
  });
});

describe('PARAM_DESCRIPTIONS', () => {
  it.each(Object.keys(PARAM_DESCRIPTIONS))('describes %s', (key) => {
    expect(PARAM_DESCRIPTIONS[key as keyof typeof PARAM_DESCRIPTIONS]).toBeTruthy();
  });

  it('covers every tunable parameter', () => {
    expect(Object.keys(PARAM_DESCRIPTIONS).sort()).toEqual(
      [
        'batchSize',
        'filters',
        'inputShape',
        'kernelSize',
        'learningRate',
        'loss',
        'optimizer',
        'padding',
        'stride',
        'units'
      ].sort()
    );
  });
});

describe('classifyInputShape', () => {
  it('classifies a flat list', () => {
    expect(classifyInputShape([2])).toBe('flat');
  });

  it('classifies an image', () => {
    expect(classifyInputShape([28, 28, 1])).toBe('image');
  });

  it('classifies anything else as other', () => {
    expect(classifyInputShape([4, 4])).toBe('other');
    expect(classifyInputShape([])).toBe('other');
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/descriptions.test.ts`
Expected: FAIL — `Failed to resolve import "./descriptions"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/descriptions.ts`:

```ts
import type { BlockKind } from './types';

export const BLOCK_DESCRIPTIONS: Record<BlockKind, string> = {
  input: 'Describes the shape of one example your network receives.',
  linear: 'Learns a weighted sum of its inputs. Also called a fully connected or dense layer.',
  conv2d: 'Slides small filters over an image to detect local patterns such as edges.',
  flatten: 'Turns image-shaped data into a flat list so Linear layers can read it.',
  relu: 'Keeps positive values and turns negative ones into zero. Helps the network learn curved patterns.',
  sigmoid: 'Squashes each value into the range 0 to 1.',
  softmax: 'Turns raw scores into probabilities that add up to 1.',
  output: 'Declares what the network predicts and how many classes there are.'
};

export const PARAM_DESCRIPTIONS = {
  inputShape: 'The shape of one example, for example 2 numbers or a 28 by 28 image.',
  units: 'How many numbers this layer produces.',
  filters: 'How many different patterns this layer looks for.',
  kernelSize: 'How large the window sliding over the image is.',
  stride: 'How far the window moves each step. Larger means a smaller result.',
  padding: 'Whether the image keeps its size ("same") or shrinks ("valid").',
  loss: 'The number the network tries to make smaller while training.',
  optimizer: 'The rule used to update the weights after each batch.',
  learningRate: 'How big each learning step is. Smaller is slower but steadier.',
  batchSize: 'How many examples are used for one weight update.'
} as const;

export type ParamKey = keyof typeof PARAM_DESCRIPTIONS;

export function classifyInputShape(shape: number[]): 'flat' | 'image' | 'other' {
  if (shape.length === 1) return 'flat';
  if (shape.length === 3) return 'image';
  return 'other';
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/descriptions.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/descriptions.ts src/lib/network/descriptions.test.ts
git commit -m "feat: add plain-language block and parameter descriptions"
```

---

### Task 5: Immutable chain operations

**Files:**
- Create: `src/lib/network/chain.ts`
- Test: `src/lib/network/chain.test.ts`

**Interfaces:**
- Consumes: `Block`, `Network` from `./types`.
- Produces:
  - `insertAt(net: Network, index: number, block: Block): Network` — index is clamped to `[1, blocks.length - 1]`, so a block can never land before the input or after the output.
  - `moveBlock(net: Network, fromIndex: number, toIndex: number): Network` — the movable range is `[1, blocks.length - 2]`; out-of-range `fromIndex` and any `toIndex` equal to `fromIndex` return the same `Network` reference unchanged.
  - `removeBlock(net: Network, id: string): Network` — refuses to remove the input (index 0), the output (last index), or an unknown id, returning the same reference.
  - `replaceBlock(net: Network, id: string, patch: Partial<Block>): Network` — merges the patch; an unknown id returns the same reference.
  - All four are pure: the input `Network` is never mutated.

- [ ] **Step 1: Write the failing test**

`src/lib/network/chain.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { insertAt, moveBlock, removeBlock, replaceBlock } from './chain';
import { createBlock } from './factory';
import type { Network } from './types';

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

const ids = (n: Network) => n.blocks.map((b) => b.id);

describe('insertAt', () => {
  it('inserts at an interior index', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 2, block))).toEqual(['in', 'a', block.id, 'b', 'c', 'out']);
  });

  it('clamps an index past the end to just before the output', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 99, block))).toEqual(['in', 'a', 'b', 'c', block.id, 'out']);
  });

  it('clamps an index before the input to just after it', () => {
    const block = createBlock('sigmoid');
    expect(ids(insertAt(net(), 0, block))).toEqual(['in', block.id, 'a', 'b', 'c', 'out']);
    expect(ids(insertAt(net(), -5, block))).toEqual(['in', block.id, 'a', 'b', 'c', 'out']);
  });

  it('does not mutate the original network', () => {
    const original = net();
    insertAt(original, 2, createBlock('relu'));
    expect(ids(original)).toEqual(['in', 'a', 'b', 'c', 'out']);
  });
});

describe('moveBlock', () => {
  it('moves a block earlier', () => {
    expect(ids(moveBlock(net(), 3, 1))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });

  it('moves a block later', () => {
    expect(ids(moveBlock(net(), 1, 3))).toEqual(['in', 'b', 'c', 'a', 'out']);
  });

  it('refuses to move the input or the output', () => {
    const original = net();
    expect(moveBlock(original, 0, 2)).toBe(original);
    expect(moveBlock(original, 4, 2)).toBe(original);
  });

  it('returns the same reference when the move is a no-op', () => {
    const original = net();
    expect(moveBlock(original, 2, 2)).toBe(original);
  });

  it('clamps the destination into the movable range', () => {
    expect(ids(moveBlock(net(), 1, 99))).toEqual(['in', 'b', 'c', 'a', 'out']);
    expect(ids(moveBlock(net(), 3, -99))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });
});

describe('removeBlock', () => {
  it('removes an interior block', () => {
    expect(ids(removeBlock(net(), 'b'))).toEqual(['in', 'a', 'c', 'out']);
  });

  it('refuses to remove the input, the output, or an unknown id', () => {
    const original = net();
    expect(removeBlock(original, 'in')).toBe(original);
    expect(removeBlock(original, 'out')).toBe(original);
    expect(removeBlock(original, 'missing')).toBe(original);
  });
});

describe('replaceBlock', () => {
  it('merges the patch into the matching block', () => {
    const updated = replaceBlock(net(), 'a', { units: 16 });
    expect(updated.blocks[1]).toMatchObject({ id: 'a', kind: 'linear', units: 16 });
    expect(updated.blocks[2]).toEqual({ id: 'b', kind: 'relu' });
  });

  it('returns the same reference for an unknown id', () => {
    const original = net();
    expect(replaceBlock(original, 'missing', { units: 16 })).toBe(original);
  });

  it('does not mutate the original block', () => {
    const original = net();
    replaceBlock(original, 'a', { units: 16 });
    expect(original.blocks[1]).toMatchObject({ units: 8 });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/chain.test.ts`
Expected: FAIL — `Failed to resolve import "./chain"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/chain.ts`:

```ts
import type { Block, Network } from './types';

const FIRST_INTERIOR = 1;

export function insertAt(net: Network, index: number, block: Block): Network {
  const last = net.blocks.length - 1;
  const target = Math.min(Math.max(index, FIRST_INTERIOR), last);
  const blocks = [...net.blocks];
  blocks.splice(target, 0, block);
  return { ...net, blocks };
}

export function moveBlock(net: Network, fromIndex: number, toIndex: number): Network {
  const last = net.blocks.length - 2;
  if (fromIndex < FIRST_INTERIOR || fromIndex > last) return net;
  const target = Math.min(Math.max(toIndex, FIRST_INTERIOR), last);
  if (target === fromIndex) return net;

  const blocks = [...net.blocks];
  const [block] = blocks.splice(fromIndex, 1);
  blocks.splice(target, 0, block);
  return { ...net, blocks };
}

export function removeBlock(net: Network, id: string): Network {
  const index = net.blocks.findIndex((block) => block.id === id);
  if (index < FIRST_INTERIOR || index > net.blocks.length - 2) return net;
  return { ...net, blocks: net.blocks.filter((block) => block.id !== id) };
}

export function replaceBlock(net: Network, id: string, patch: Partial<Block>): Network {
  let changed = false;
  const blocks = net.blocks.map((block) => {
    if (block.id !== id) return block;
    changed = true;
    return { ...block, ...patch } as Block;
  });
  return changed ? { ...net, blocks } : net;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/chain.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/chain.ts src/lib/network/chain.test.ts
git commit -m "feat: add immutable chain operations"
```

---

### Task 6: Shape inference

**Files:**
- Create: `src/lib/network/inferShapes.ts`
- Test: `src/lib/network/inferShapes.test.ts`

**Interfaces:**
- Consumes: `Block`, `Network` from `./types`.
- Produces:

```ts
export interface ShapeInfo {
  blockId: string;
  inShape: number[] | null;
  outShape: number[] | null;
  paramCount: number | null;
}

export interface EdgeShape {
  fromId: string;
  toId: string;
  shape: number[] | null;
}

export interface ShapeResult {
  perBlock: ShapeInfo[];
  edges: EdgeShape[];
  totalParamCount: number;
}

export function convOutputSize(size: number, kernelSize: number, stride: number, padding: 'same' | 'valid'): number;
export function inferShapes(net: Network): ShapeResult;
```

- Semantics: `inShape` is the previous block's output shape, or `null` for the input block and for any block after a shape error. `outShape` is `null` when this block cannot produce a shape. Once a block fails, every later block has `inShape: null` and `outShape: null`. `paramCount` is `0` for parameterless blocks, the exact trainable count for `linear` and `conv2d`, and `null` when the shape needed to compute it is unknown. `totalParamCount` sums the non-null counts.
- `convOutputSize` with `'same'` returns `Math.ceil(size / stride)`; with `'valid'` returns `Math.floor((size - kernelSize) / stride) + 1`.
- `paramCount` formulas: `linear` → `inShape[0] * units + units`; `conv2d` → `kernelSize * kernelSize * inShape[2] * filters + filters`.
- A `linear` block whose `inShape` is not rank 1, and a `conv2d` block whose `inShape` is not rank 3, produce `outShape: null` and start the null cascade.

- [ ] **Step 1: Write the failing test**

`src/lib/network/inferShapes.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { convOutputSize, inferShapes } from './inferShapes';
import { createEmptyNetwork } from './factory';
import type { Network } from './types';

function net(blocks: Network['blocks']): Network {
  return {
    version: 1,
    blocks,
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
  };
}

describe('convOutputSize', () => {
  it('keeps the size with same padding', () => {
    expect(convOutputSize(28, 3, 1, 'same')).toBe(28);
    expect(convOutputSize(28, 3, 2, 'same')).toBe(14);
  });

  it('shrinks with valid padding', () => {
    expect(convOutputSize(28, 3, 1, 'valid')).toBe(26);
    expect(convOutputSize(28, 5, 2, 'valid')).toBe(12);
  });
});

describe('inferShapes on the default MLP', () => {
  const result = inferShapes(createEmptyNetwork());

  it('produces one entry per block', () => {
    expect(result.perBlock).toHaveLength(6);
  });

  it('tracks shapes through the pipeline', () => {
    expect(result.perBlock.map((p) => p.inShape)).toEqual([
      null,
      [2],
      [8],
      [8],
      [2],
      [2]
    ]);
    expect(result.perBlock.map((p) => p.outShape)).toEqual([
      [2],
      [8],
      [8],
      [2],
      [2],
      [2]
    ]);
  });

  it('counts parameters', () => {
    expect(result.perBlock.map((p) => p.paramCount)).toEqual([0, 24, 0, 18, 0, 0]);
    expect(result.totalParamCount).toBe(42);
  });

  it('labels every edge with the shape travelling along it', () => {
    expect(result.edges).toHaveLength(5);
    expect(result.edges.map((e) => e.shape)).toEqual([[2], [8], [8], [2], [2]]);
  });
});

describe('inferShapes on a convolutional chain', () => {
  const result = inferShapes(
    net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 10 },
      { id: 'out', kind: 'output', units: 10 }
    ])
  );

  it('computes the convolution output shape', () => {
    expect(result.perBlock[1].outShape).toEqual([28, 28, 4]);
  });

  it('flattens to a single dimension', () => {
    expect(result.perBlock[2].outShape).toEqual([28 * 28 * 4]);
  });

  it('counts convolution parameters from the input channels', () => {
    expect(result.perBlock[1].paramCount).toBe(3 * 3 * 1 * 4 + 4);
  });
});

describe('inferShapes error propagation', () => {
  const result = inferShapes(
    net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'dense', kind: 'linear', units: 8 },
      { id: 'out', kind: 'output', units: 2 }
    ])
  );

  it('records the offending input shape but no output shape', () => {
    expect(result.perBlock[1].inShape).toEqual([28, 28, 1]);
    expect(result.perBlock[1].outShape).toBeNull();
    expect(result.perBlock[1].paramCount).toBeNull();
  });

  it('nulls out every later shape', () => {
    expect(result.perBlock[2].inShape).toBeNull();
    expect(result.perBlock[2].outShape).toBeNull();
  });

  it('excludes unknown counts from the total', () => {
    expect(result.totalParamCount).toBe(0);
  });
});

describe('inferShapes with an impossible convolution', () => {
  it('returns a null output shape', () => {
    const result = inferShapes(
      net([
        { id: 'in', kind: 'input', shape: [2, 2, 1] },
        { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 3, padding: 'valid' },
        { id: 'out', kind: 'output', units: 2 }
      ])
    );
    expect(result.perBlock[1].inShape).toEqual([2, 2, 1]);
    expect(result.perBlock[1].outShape).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/inferShapes.test.ts`
Expected: FAIL — `Failed to resolve import "./inferShapes"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/inferShapes.ts`:

```ts
import type { Block, Network } from './types';

export interface ShapeInfo {
  blockId: string;
  inShape: number[] | null;
  outShape: number[] | null;
  paramCount: number | null;
}

export interface EdgeShape {
  fromId: string;
  toId: string;
  shape: number[] | null;
}

export interface ShapeResult {
  perBlock: ShapeInfo[];
  edges: EdgeShape[];
  totalParamCount: number;
}

export function convOutputSize(
  size: number,
  kernelSize: number,
  stride: number,
  padding: 'same' | 'valid'
): number {
  if (padding === 'same') return Math.ceil(size / stride);
  return Math.floor((size - kernelSize) / stride) + 1;
}

function product(shape: number[]): number {
  return shape.reduce((total, dimension) => total * dimension, 1);
}

function outputShapeFor(block: Block, inShape: number[] | null): number[] | null {
  switch (block.kind) {
    case 'input':
      return [...block.shape];
    case 'linear':
      if (!inShape || inShape.length !== 1) return null;
      return [block.units];
    case 'conv2d': {
      if (!inShape || inShape.length !== 3) return null;
      const [height, width] = inShape;
      const outHeight = convOutputSize(height, block.kernelSize, block.stride, block.padding);
      const outWidth = convOutputSize(width, block.kernelSize, block.stride, block.padding);
      if (outHeight <= 0 || outWidth <= 0) return null;
      return [outHeight, outWidth, block.filters];
    }
    case 'flatten':
      if (!inShape || inShape.length < 1) return null;
      return [product(inShape)];
    case 'relu':
    case 'sigmoid':
    case 'softmax':
    case 'output':
      return inShape ? [...inShape] : null;
    default:
      return null;
  }
}

function paramCountFor(block: Block, inShape: number[] | null): number | null {
  switch (block.kind) {
    case 'linear':
      if (!inShape || inShape.length !== 1) return null;
      return inShape[0] * block.units + block.units;
    case 'conv2d':
      if (!inShape || inShape.length !== 3) return null;
      return block.kernelSize * block.kernelSize * inShape[2] * block.filters + block.filters;
    default:
      return 0;
  }
}

export function inferShapes(net: Network): ShapeResult {
  const perBlock: ShapeInfo[] = [];
  const edges: EdgeShape[] = [];
  let current: number[] | null = null;
  let broken = false;

  net.blocks.forEach((block, index) => {
    const inShape = current;
    const outShape = broken ? null : outputShapeFor(block, inShape);

    if (!broken && outShape === null) broken = true;

    perBlock.push({
      blockId: block.id,
      inShape,
      outShape,
      paramCount: paramCountFor(block, inShape)
    });

    if (index > 0) {
      edges.push({ fromId: net.blocks[index - 1].id, toId: block.id, shape: inShape });
    }

    current = outShape;
  });

  const totalParamCount = perBlock.reduce(
    (total, info) => total + (info.paramCount ?? 0),
    0
  );

  return { perBlock, edges, totalParamCount };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/inferShapes.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/inferShapes.ts src/lib/network/inferShapes.test.ts
git commit -m "feat: add shape inference with parameter counts"
```

---

### Task 7: Validation errors

**Files:**
- Create: `src/lib/network/validate.ts`
- Test: `src/lib/network/validate.test.ts`

**Interfaces:**
- Consumes: `inferShapes` from `./inferShapes`, `Network`/`Block` from `./types`.
- Produces:

```ts
export type Severity = 'error' | 'warning';

export interface Issue {
  severity: Severity;
  title: string;
  message: string;
  fix: string;
  blockId?: string;
}

export interface ValidateOptions {
  expectedClasses?: number;
}

export function validate(net: Network, options?: ValidateOptions): Issue[];
```

- This task implements the **error** rules only. Task 8 adds the warning rules to the same function.
- Error rules and their exact `title` values:
  - `'Missing Input block'` — no `input` block.
  - `'Missing Output block'` — no `output` block.
  - `'More than one Input block'` — two or more `input` blocks.
  - `'More than one Output block'` — two or more `output` blocks.
  - `'Nothing to learn'` — fewer than one block between input and output.
  - `'Linear layer needs a flat list'` — a `linear` block whose `inShape` is not rank 1; the message quotes the shape, e.g. `[28, 28, 1]`.
  - `'Convolution layer needs image data'` — a `conv2d` block whose `inShape` is not rank 3.
  - `'Kernel is larger than the image'` — a `conv2d` block whose `inShape` is rank 3 but whose computed output size is not positive; the message quotes the kernel size, stride, and image dimensions.
  - `'Nothing to flatten'` — a `flatten` block whose `inShape` is rank 1.
  - `'Unrecognised block'` — a block whose `kind` is not one of the eight known kinds (defensive, for deserialised data).
- Shape-bearing errors set `blockId` to the offending block. Network-wide errors leave `blockId` undefined.
- The input and output shape checks use `inferShapes`, so `inShape` is available even when the block itself failed.

- [ ] **Step 1: Write the failing test**

`src/lib/network/validate.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import type { Block, Network } from './types';
import { validate, type Issue } from './validate';

function net(blocks: Network['blocks'], training?: Partial<Network['training']>): Network {
  return {
    version: 1,
    blocks,
    training: {
      loss: 'crossEntropy',
      optimizer: 'adam',
      learningRate: 0.01,
      batchSize: 32,
      ...training
    }
  };
}

const errors = (network: Network): Issue[] =>
  validate(network).filter((issue) => issue.severity === 'error');

const titles = (network: Network): string[] => errors(network).map((issue) => issue.title);

const INPUT: Block = { id: 'in', kind: 'input', shape: [2] };
const OUTPUT: Block = { id: 'out', kind: 'output', units: 2 };

describe('validate errors', () => {
  it('accepts the default network', () => {
    expect(errors(createEmptyNetwork())).toEqual([]);
  });

  it('reports a missing input block', () => {
    expect(titles(net([{ id: 'a', kind: 'relu' }, OUTPUT]))).toContain('Missing Input block');
  });

  it('reports a missing output block', () => {
    expect(titles(net([INPUT, { id: 'a', kind: 'relu' }]))).toContain('Missing Output block');
  });

  it('reports more than one input block', () => {
    const second: Block = { id: 'in2', kind: 'input', shape: [2] };
    expect(titles(net([INPUT, second, { id: 'a', kind: 'relu' }, OUTPUT]))).toContain(
      'More than one Input block'
    );
  });

  it('reports more than one output block', () => {
    const second: Block = { id: 'out2', kind: 'output', units: 2 };
    expect(titles(net([INPUT, { id: 'a', kind: 'relu' }, OUTPUT, second]))).toContain(
      'More than one Output block'
    );
  });

  it('reports a network with nothing to learn', () => {
    expect(titles(net([INPUT, OUTPUT]))).toContain('Nothing to learn');
  });

  it('reports a linear layer receiving image-shaped data and names the shape', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'dense', kind: 'linear', units: 8 },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Linear layer needs a flat list');
    expect(issue).toBeDefined();
    expect(issue?.blockId).toBe('dense');
    expect(issue?.message).toContain('[28, 28, 1]');
  });

  it('reports a convolution layer receiving flat data', () => {
    const network = net([
      INPUT,
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Convolution layer needs image data');
    expect(issue?.blockId).toBe('conv');
  });

  it('reports a kernel larger than the image', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [2, 2, 1] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 3, padding: 'valid' },
      OUTPUT
    ]);
    const issue = errors(network).find((i) => i.title === 'Kernel is larger than the image');
    expect(issue?.blockId).toBe('conv');
    expect(issue?.message).toContain('3');
  });

  it('reports a flatten layer receiving an already flat list', () => {
    const network = net([INPUT, { id: 'flat', kind: 'flatten' }, OUTPUT]);
    const issue = errors(network).find((i) => i.title === 'Nothing to flatten');
    expect(issue?.blockId).toBe('flat');
  });

  it('reports an unrecognised block kind', () => {
    const bogus = { id: 'x', kind: 'dropout' } as unknown as Block;
    const issue = errors(net([INPUT, bogus, OUTPUT])).find(
      (i) => i.title === 'Unrecognised block'
    );
    expect(issue?.blockId).toBe('x');
  });

  it('gives every error a title, message, and fix', () => {
    const networks = [
      net([{ id: 'a', kind: 'relu' }, OUTPUT]),
      net([INPUT, OUTPUT]),
      net([
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        OUTPUT
      ])
    ];
    for (const network of networks) {
      for (const issue of errors(network)) {
        expect(issue.title.length).toBeGreaterThan(0);
        expect(issue.message.length).toBeGreaterThan(0);
        expect(issue.fix.length).toBeGreaterThan(0);
      }
    }
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: FAIL — `Failed to resolve import "./validate"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/validate.ts`:

```ts
import { inferShapes } from './inferShapes';
import { BLOCK_KINDS, type Network } from './types';

export type Severity = 'error' | 'warning';

export interface Issue {
  severity: Severity;
  title: string;
  message: string;
  fix: string;
  blockId?: string;
}

export interface ValidateOptions {
  expectedClasses?: number;
}

function shapeText(shape: number[] | null): string {
  return shape ? `[${shape.join(', ')}]` : 'an unknown shape';
}

export function validate(net: Network, options: ValidateOptions = {}): Issue[] {
  void options;
  const issues: Issue[] = [];
  const { perBlock } = inferShapes(net);

  const inputs = net.blocks.filter((block) => block.kind === 'input');
  const outputs = net.blocks.filter((block) => block.kind === 'output');

  if (inputs.length === 0) {
    issues.push({
      severity: 'error',
      title: 'Missing Input block',
      message:
        'A network needs exactly one Input block to describe the shape of the data it receives.',
      fix: 'Add an Input block to the start of the network.'
    });
  }

  if (outputs.length === 0) {
    issues.push({
      severity: 'error',
      title: 'Missing Output block',
      message: 'A network needs exactly one Output block to say what it predicts.',
      fix: 'Add an Output block to the end of the network.'
    });
  }

  if (inputs.length > 1) {
    issues.push({
      severity: 'error',
      title: 'More than one Input block',
      message: `There are ${inputs.length} Input blocks, but a network can only have one.`,
      fix: 'Delete the extra Input blocks.'
    });
  }

  if (outputs.length > 1) {
    issues.push({
      severity: 'error',
      title: 'More than one Output block',
      message: `There are ${outputs.length} Output blocks, but a network can only have one.`,
      fix: 'Delete the extra Output blocks.'
    });
  }

  if (net.blocks.length - 2 < 1) {
    issues.push({
      severity: 'error',
      title: 'Nothing to learn',
      message:
        'The Input connects straight to the Output, so there are no layers for the network to learn with.',
      fix: 'Add at least one layer, such as a Linear layer, between Input and Output.'
    });
  }

  net.blocks.forEach((block, index) => {
    const info = perBlock[index];

    if (!BLOCK_KINDS.includes(block.kind)) {
      issues.push({
        severity: 'error',
        title: 'Unrecognised block',
        message: `This network contains a block type this version of VisNet does not understand (${block.kind}).`,
        fix: 'Delete the block, or reset the network to start fresh.',
        blockId: block.id
      });
      return;
    }

    if (block.kind === 'linear' && info.inShape && info.inShape.length !== 1) {
      issues.push({
        severity: 'error',
        title: 'Linear layer needs a flat list',
        message: `This Linear layer receives ${shapeText(info.inShape)}, which is image-shaped. Linear layers need a flat list of numbers.`,
        fix: 'Add a Flatten layer before this Linear layer.',
        blockId: block.id
      });
    }

    if (block.kind === 'conv2d' && info.inShape && info.inShape.length !== 3) {
      issues.push({
        severity: 'error',
        title: 'Convolution layer needs image data',
        message: `This Convolution layer receives ${shapeText(info.inShape)}. It expects image data shaped [height, width, channels].`,
        fix: 'Give the Input block a 3D shape such as [28, 28, 1], or remove the Convolution layer.',
        blockId: block.id
      });
    }

    if (
      block.kind === 'conv2d' &&
      info.inShape &&
      info.inShape.length === 3 &&
      info.outShape === null
    ) {
      const [height, width] = info.inShape;
      issues.push({
        severity: 'error',
        title: 'Kernel is larger than the image',
        message: `A ${block.kernelSize}×${block.kernelSize} kernel with stride ${block.stride} leaves no room to slide over a ${height}×${width} image.`,
        fix: "Use a smaller kernel or stride, or set padding to 'same'.",
        blockId: block.id
      });
    }

    if (block.kind === 'flatten' && info.inShape && info.inShape.length === 1) {
      issues.push({
        severity: 'error',
        title: 'Nothing to flatten',
        message: `This Flatten layer receives ${shapeText(info.inShape)}, which is already a flat list.`,
        fix: 'Remove this Flatten layer, or move it after a Convolution layer.',
        blockId: block.id
      });
    }
  });

  return issues;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/validate.ts src/lib/network/validate.test.ts
git commit -m "feat: add validation error rules with plain-language fixes"
```

---

### Task 8: Validation warnings

**Files:**
- Modify: `src/lib/network/validate.ts` (replace the `void options;` line and append the warning rules before `return issues;`)
- Test: `src/lib/network/validate.test.ts` (append a new `describe` block)

**Interfaces:**
- Consumes: everything from Task 7, plus `classifyInputShape` from `./descriptions`.
- Produces: the same `validate(net, options)` signature, now also emitting warnings.
- Warning rules and their exact `title` values:
  - `'Add a Softmax for probabilities'` — `training.loss === 'crossEntropy'` and there is no `softmax` block.
  - `'Softmax is unusual with mean squared error'` — `training.loss === 'mse'` and there is a `softmax` block.
  - `'Softmax is not the last layer'` — a `softmax` block exists at an index that is not the last index whose kind is not `'output'`; `blockId` is the softmax block.
  - `'Output size does not match the data'` — `options.expectedClasses` is defined and the `output` block's `units` differ; `blockId` is the output block.
  - `'Image input without a Convolution layer'` — the input shape is rank 3 and there is no `conv2d` block; `blockId` is the input block.
  - `'Convolution layer without image input'` — the input shape is rank 1 and there is a `conv2d` block; `blockId` is the input block.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/network/validate.test.ts`:

```ts
const warnings = (network: Network, options?: { expectedClasses?: number }): Issue[] =>
  validate(network, options).filter((issue) => issue.severity === 'warning');

const warningTitles = (network: Network, options?: { expectedClasses?: number }): string[] =>
  warnings(network, options).map((issue) => issue.title);

describe('validate warnings', () => {
  it('says nothing about the default network', () => {
    expect(warnings(createEmptyNetwork())).toEqual([]);
  });

  it('suggests a softmax for cross-entropy without one', () => {
    const network = net([
      INPUT,
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ]);
    expect(warningTitles(network)).toContain('Add a Softmax for probabilities');
  });

  it('flags softmax with mean squared error', () => {
    const network = net(
      [INPUT, { id: 'dense', kind: 'linear', units: 2 }, { id: 'sm', kind: 'softmax' }, OUTPUT],
      { loss: 'mse' }
    );
    expect(warningTitles(network)).toContain('Softmax is unusual with mean squared error');
  });

  it('flags a softmax that is not the last layer', () => {
    const network = net([
      INPUT,
      { id: 'sm', kind: 'softmax' },
      { id: 'dense', kind: 'linear', units: 2 },
      OUTPUT
    ]);
    const issue = warnings(network).find((i) => i.title === 'Softmax is not the last layer');
    expect(issue?.blockId).toBe('sm');
  });

  it('does not flag a softmax immediately before the output', () => {
    const network = net([
      INPUT,
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    expect(warningTitles(network)).not.toContain('Softmax is not the last layer');
  });

  it('flags a mismatch between output units and the dataset', () => {
    const issue = warnings(createEmptyNetwork(), { expectedClasses: 3 }).find(
      (i) => i.title === 'Output size does not match the data'
    );
    expect(issue).toBeDefined();
    expect(issue?.message).toContain('2');
    expect(issue?.message).toContain('3');
  });

  it('does not flag matching output units', () => {
    expect(warningTitles(createEmptyNetwork(), { expectedClasses: 2 })).toEqual([]);
  });

  it('flags image input without a convolution layer', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [28, 28, 1] },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network).find(
      (i) => i.title === 'Image input without a Convolution layer'
    );
    expect(issue?.blockId).toBe('in');
  });

  it('flags a convolution layer with flat input', () => {
    const network = net([
      { id: 'in', kind: 'input', shape: [784] },
      { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
      { id: 'flat', kind: 'flatten' },
      { id: 'dense', kind: 'linear', units: 2 },
      { id: 'sm', kind: 'softmax' },
      OUTPUT
    ]);
    const issue = warnings(network).find(
      (i) => i.title === 'Convolution layer without image input'
    );
    expect(issue?.blockId).toBe('in');
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: FAIL — the warning assertions find no issues (the new `describe` block fails while the Task 7 block still passes).

- [ ] **Step 3: Write the implementation**

In `src/lib/network/validate.ts`, change the import block to:

```ts
import { classifyInputShape } from './descriptions';
import { inferShapes } from './inferShapes';
import { BLOCK_KINDS, type Network } from './types';
```

Remove the `void options;` line.

Immediately before `return issues;`, insert:

```ts
  const softmaxIndex = net.blocks.findIndex((block) => block.kind === 'softmax');
  const hasSoftmax = softmaxIndex !== -1;
  const hasConvolution = net.blocks.some((block) => block.kind === 'conv2d');

  if (net.training.loss === 'crossEntropy' && !hasSoftmax) {
    issues.push({
      severity: 'warning',
      title: 'Add a Softmax for probabilities',
      message:
        "Cross-entropy works best when the network's outputs are probabilities, but the network currently ends with raw scores. Training will still run, but it may be less stable.",
      fix: 'Add a Softmax block after the last Linear layer.'
    });
  }

  if (net.training.loss === 'mse' && hasSoftmax) {
    issues.push({
      severity: 'warning',
      title: 'Softmax is unusual with mean squared error',
      message: 'Mean squared error is normally used with raw scores, not probabilities.',
      fix: 'Switch the loss to cross-entropy, or remove the Softmax block.'
    });
  }

  if (hasSoftmax) {
    let lastRealIndex = -1;
    net.blocks.forEach((block, index) => {
      if (block.kind !== 'output') lastRealIndex = index;
    });
    if (softmaxIndex !== lastRealIndex) {
      issues.push({
        severity: 'warning',
        title: 'Softmax is not the last layer',
        message:
          'This Softmax block is followed by more layers, so the probabilities it produces get transformed again.',
        fix: 'Move the Softmax block to just before the Output block.',
        blockId: net.blocks[softmaxIndex].id
      });
    }
  }

  const outputBlock = net.blocks.find((block) => block.kind === 'output');
  if (
    options.expectedClasses !== undefined &&
    outputBlock &&
    outputBlock.kind === 'output' &&
    outputBlock.units !== options.expectedClasses
  ) {
    issues.push({
      severity: 'warning',
      title: 'Output size does not match the data',
      message: `The Output block says ${outputBlock.units} classes, but the dataset has ${options.expectedClasses}.`,
      fix: `Set the Output block to ${options.expectedClasses} units.`,
      blockId: outputBlock.id
    });
  }

  const inputBlock = net.blocks.find((block) => block.kind === 'input');
  if (inputBlock && inputBlock.kind === 'input') {
    const inputKind = classifyInputShape(inputBlock.shape);
    if (inputKind === 'image' && !hasConvolution) {
      issues.push({
        severity: 'warning',
        title: 'Image input without a Convolution layer',
        message: `The Input block is image-shaped ${shapeText(inputBlock.shape)}, but the network has no Convolution layer to look at it.`,
        fix: 'Add a Convolution layer, or change the Input shape to a flat list.',
        blockId: inputBlock.id
      });
    }
    if (inputKind === 'flat' && hasConvolution) {
      issues.push({
        severity: 'warning',
        title: 'Convolution layer without image input',
        message: `The network has a Convolution layer, but the Input block is a flat list ${shapeText(inputBlock.shape)}.`,
        fix: 'Set the Input shape to 3D such as [28, 28, 1], or remove the Convolution layer.',
        blockId: inputBlock.id
      });
    }
  }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/validate.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/validate.ts src/lib/network/validate.test.ts
git commit -m "feat: add validation warning rules"
```

---

### Task 9: Network serialization

**Files:**
- Create: `src/lib/network/serialize.ts`
- Test: `src/lib/network/serialize.test.ts`

**Interfaces:**
- Consumes: `createEmptyNetwork` from `./factory`, `Block`, `Network` from `./types`.
- Produces:
  - `toJSON(net: Network): string` — returns `JSON.stringify({ version: net.version, network: net })`.
  - `fromJSON(raw: string): Network | null` — parses the envelope, runs `migrate`, structural-validates, and returns `null` for corrupt data, an unsupported version, or a malformed network.
  - `migrate(raw: unknown): Network | null` — exported so a future version can be added in one place. For version 1 it returns the `network` field.

- [ ] **Step 1: Write the failing test**

`src/lib/network/serialize.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from './factory';
import { fromJSON, migrate, toJSON } from './serialize';
import type { Network } from './types';

describe('toJSON', () => {
  it('wraps the network in a versioned envelope', () => {
    const envelope = JSON.parse(toJSON(createEmptyNetwork()));
    expect(envelope.version).toBe(1);
    expect(envelope.network.blocks).toHaveLength(6);
  });
});

describe('fromJSON', () => {
  it('round-trips a network', () => {
    const original = createEmptyNetwork();
    const restored = fromJSON(toJSON(original));
    expect(restored).toEqual(original);
  });

  it('preserves training configuration', () => {
    const original = createEmptyNetwork();
    original.training.optimizer = 'sgd';
    original.training.learningRate = 0.25;
    const restored = fromJSON(toJSON(original));
    expect(restored?.training).toEqual(original.training);
  });

  it('returns null for invalid JSON', () => {
    expect(fromJSON('not json')).toBeNull();
  });

  it('returns null for a missing envelope', () => {
    expect(fromJSON(JSON.stringify({ blocks: [] }))).toBeNull();
  });

  it('returns null for an unsupported future version', () => {
    expect(fromJSON(JSON.stringify({ version: 99, network: createEmptyNetwork() }))).toBeNull();
  });

  it('returns null when the network is structurally wrong', () => {
    expect(fromJSON(JSON.stringify({ version: 1, network: { blocks: 'nope' } }))).toBeNull();
    expect(fromJSON(JSON.stringify({ version: 1, network: { blocks: [{ id: 1 }] } }))).toBeNull();
    expect(
      fromJSON(JSON.stringify({ version: 1, network: { blocks: [], training: {} } }))
    ).toBeNull();
  });
});

describe('migrate', () => {
  it('accepts version 1', () => {
    const original: Network = createEmptyNetwork();
    expect(migrate({ version: 1, network: original })).toEqual(original);
  });

  it('rejects unknown versions', () => {
    expect(migrate({ version: 2, network: createEmptyNetwork() })).toBeNull();
  });

  it('rejects non-objects', () => {
    expect(migrate(null)).toBeNull();
    expect(migrate('nope')).toBeNull();
    expect(migrate(42)).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/network/serialize.test.ts`
Expected: FAIL — `Failed to resolve import "./serialize"`.

- [ ] **Step 3: Write the implementation**

`src/lib/network/serialize.ts`:

```ts
import type { Block, Network, TrainingConfig } from './types';

const SUPPORTED_VERSION = 1;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isBlock(value: unknown): value is Block {
  if (!isRecord(value)) return false;
  return typeof value.id === 'string' && typeof value.kind === 'string';
}

function isTrainingConfig(value: unknown): value is TrainingConfig {
  if (!isRecord(value)) return false;
  return (
    (value.loss === 'mse' || value.loss === 'crossEntropy') &&
    (value.optimizer === 'sgd' || value.optimizer === 'adam') &&
    typeof value.learningRate === 'number' &&
    typeof value.batchSize === 'number'
  );
}

function isNetwork(value: unknown): value is Network {
  if (!isRecord(value)) return false;
  if (value.version !== SUPPORTED_VERSION) return false;
  if (!Array.isArray(value.blocks) || !value.blocks.every(isBlock)) return false;
  return isTrainingConfig(value.training);
}

export function migrate(raw: unknown): Network | null {
  if (!isRecord(raw)) return null;
  if (raw.version !== SUPPORTED_VERSION) return null;
  return isNetwork(raw.network) ? raw.network : null;
}

export function toJSON(net: Network): string {
  return JSON.stringify({ version: net.version, network: net });
}

export function fromJSON(raw: string): Network | null {
  try {
    return migrate(JSON.parse(raw));
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/network/serialize.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/network/serialize.ts src/lib/network/serialize.test.ts
git commit -m "feat: add versioned network serialization"
```

---

### Task 10: Canvas projection

**Files:**
- Create: `src/lib/editor/flow.ts`
- Test: `src/lib/editor/flow.test.ts`

**Interfaces:**
- Consumes: `Network`, `BlockKind` from `../network/types`; `ShapeResult` from `../network/inferShapes`; `moveBlock` from `../network/chain`.
- Produces:

```ts
export const NODE_WIDTH = 200;
export const NODE_GAP = 80;

export interface FlowNode {
  id: string;
  position: { x: number; y: number };
  data: {
    kind: BlockKind;
    inShape: number[] | null;
    outShape: number[] | null;
    paramCount: number | null;
    index: number;
    removable: boolean;
  };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  label: string | null;
}

export interface ChainMove {
  type: 'move';
  from: number;
  to: number;
}

export function shapeLabel(shape: number[] | null): string | null;
export function toFlow(net: Network, shapes: ShapeResult): { nodes: FlowNode[]; edges: FlowEdge[] };
export function connectionToIntent(
  connection: { source: string; target: string },
  net: Network
): ChainMove | null;
```

- This module is pure and must **not** import `@xyflow/svelte`. The canvas component in Plan B adapts `FlowNode`/`FlowEdge` to Svelte Flow's own node and edge shapes.
- `shapeLabel` returns `'[2]'` for `[2]`, `'[28 × 28 × 4]'` for `[28, 28, 4]`, and `null` for `null`.
- `toFlow` positions node `i` at `x = i * (NODE_WIDTH + NODE_GAP)`, `y = 0`. `removable` is `false` for the input and output blocks. Edge ids are `"{fromId}->{toId}"`.
- `connectionToIntent` returns `null` when: either id is unknown, the source is the input or the output, the target is the input, the target is already the block immediately after the source, or the move would be a no-op. Otherwise it returns the move that would put the source block directly before the target block.
- Applying the returned intent with `moveBlock(net, intent.from, intent.to)` must produce the expected order.

- [ ] **Step 1: Write the failing test**

`src/lib/editor/flow.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { moveBlock } from '../network/chain';
import { inferShapes } from '../network/inferShapes';
import type { Network } from '../network/types';
import { NODE_GAP, NODE_WIDTH, connectionToIntent, shapeLabel, toFlow } from './flow';

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

const ids = (network: Network) => network.blocks.map((b) => b.id);

describe('shapeLabel', () => {
  it('renders a one-dimensional shape', () => {
    expect(shapeLabel([2])).toBe('[2]');
  });

  it('renders a multi-dimensional shape', () => {
    expect(shapeLabel([28, 28, 4])).toBe('[28 × 28 × 4]');
  });

  it('renders nothing for an unknown shape', () => {
    expect(shapeLabel(null)).toBeNull();
  });
});

describe('toFlow', () => {
  const network = net();
  const flow = toFlow(network, inferShapes(network));

  it('creates one node per block, positioned left to right', () => {
    expect(flow.nodes.map((node) => node.id)).toEqual(ids(network));
    expect(flow.nodes.map((node) => node.position.x)).toEqual([
      0,
      NODE_WIDTH + NODE_GAP,
      2 * (NODE_WIDTH + NODE_GAP),
      3 * (NODE_WIDTH + NODE_GAP),
      4 * (NODE_WIDTH + NODE_GAP)
    ]);
    expect(flow.nodes.every((node) => node.position.y === 0)).toBe(true);
  });

  it('marks only the interior blocks as removable', () => {
    expect(flow.nodes.map((node) => node.data.removable)).toEqual([false, true, true, true, false]);
  });

  it('attaches shapes and parameter counts to nodes', () => {
    expect(flow.nodes[1].data).toMatchObject({
      kind: 'linear',
      inShape: [2],
      outShape: [8],
      paramCount: 24,
      index: 1
    });
  });

  it('creates one labelled edge per adjacent pair', () => {
    expect(flow.edges.map((edge) => edge.id)).toEqual([
      'in->a',
      'a->b',
      'b->c',
      'c->out'
    ]);
    expect(flow.edges.map((edge) => edge.label)).toEqual(['[2]', '[8]', '[8]', '[2]']);
  });
});

describe('connectionToIntent', () => {
  it('moves a block later when dropped on a later block', () => {
    const intent = connectionToIntent({ source: 'a', target: 'c' }, net());
    expect(intent).toEqual({ type: 'move', from: 1, to: 2 });
    expect(ids(moveBlock(net(), 1, 2))).toEqual(['in', 'b', 'a', 'c', 'out']);
  });

  it('moves a block earlier when dropped on an earlier block', () => {
    const intent = connectionToIntent({ source: 'c', target: 'a' }, net());
    expect(intent).toEqual({ type: 'move', from: 3, to: 1 });
    expect(ids(moveBlock(net(), 3, 1))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });

  it('lets a block be dropped on the output to become the last layer', () => {
    const intent = connectionToIntent({ source: 'a', target: 'out' }, net());
    expect(intent).toEqual({ type: 'move', from: 1, to: 3 });
    expect(ids(moveBlock(net(), 1, 3))).toEqual(['in', 'b', 'c', 'a', 'out']);
  });

  it('ignores a self connection', () => {
    expect(connectionToIntent({ source: 'a', target: 'a' }, net())).toBeNull();
  });

  it('ignores connections from the input or the output', () => {
    expect(connectionToIntent({ source: 'in', target: 'c' }, net())).toBeNull();
    expect(connectionToIntent({ source: 'out', target: 'a' }, net())).toBeNull();
  });

  it('ignores connections into the input', () => {
    expect(connectionToIntent({ source: 'c', target: 'in' }, net())).toBeNull();
  });

  it('ignores connections between already adjacent blocks', () => {
    expect(connectionToIntent({ source: 'a', target: 'b' }, net())).toBeNull();
  });

  it('ignores unknown ids', () => {
    expect(connectionToIntent({ source: 'a', target: 'missing' }, net())).toBeNull();
    expect(connectionToIntent({ source: 'missing', target: 'a' }, net())).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/editor/flow.test.ts`
Expected: FAIL — `Failed to resolve import "./flow"`.

- [ ] **Step 3: Write the implementation**

`src/lib/editor/flow.ts`:

```ts
import type { ShapeResult } from '../network/inferShapes';
import type { BlockKind, Network } from '../network/types';

export const NODE_WIDTH = 200;
export const NODE_GAP = 80;

export interface FlowNode {
  id: string;
  position: { x: number; y: number };
  data: {
    kind: BlockKind;
    inShape: number[] | null;
    outShape: number[] | null;
    paramCount: number | null;
    index: number;
    removable: boolean;
  };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  label: string | null;
}

export interface ChainMove {
  type: 'move';
  from: number;
  to: number;
}

export function shapeLabel(shape: number[] | null): string | null {
  return shape ? `[${shape.join(' × ')}]` : null;
}

export function toFlow(
  net: Network,
  shapes: ShapeResult
): { nodes: FlowNode[]; edges: FlowEdge[] } {
  const nodes: FlowNode[] = net.blocks.map((block, index) => {
    const info = shapes.perBlock[index];
    return {
      id: block.id,
      position: { x: index * (NODE_WIDTH + NODE_GAP), y: 0 },
      data: {
        kind: block.kind,
        inShape: info.inShape,
        outShape: info.outShape,
        paramCount: info.paramCount,
        index,
        removable: block.kind !== 'input' && block.kind !== 'output'
      }
    };
  });

  const edges: FlowEdge[] = shapes.edges.map((edge) => ({
    id: `${edge.fromId}->${edge.toId}`,
    source: edge.fromId,
    target: edge.toId,
    label: shapeLabel(edge.shape)
  }));

  return { nodes, edges };
}

export function connectionToIntent(
  connection: { source: string; target: string },
  net: Network
): ChainMove | null {
  const from = net.blocks.findIndex((block) => block.id === connection.source);
  const to = net.blocks.findIndex((block) => block.id === connection.target);

  if (from === -1 || to === -1) return null;
  if (from === 0 || from === net.blocks.length - 1) return null;
  if (to === 0) return null;
  if (to === from + 1) return null;

  const destination = from < to ? to - 1 : to;
  if (destination === from) return null;

  return { type: 'move', from, to: destination };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/editor/flow.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/editor/flow.ts src/lib/editor/flow.test.ts
git commit -m "feat: add pure canvas projection and connection intents"
```

---

### Task 11: Undo/redo history

**Files:**
- Create: `src/lib/editor/history.ts`
- Test: `src/lib/editor/history.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:

```ts
export class History<T> {
  constructor(limit?: number); // default 50
  push(previous: T): void;
  undo(current: T): T | null;
  redo(current: T): T | null;
  clear(): void;
  get canUndo(): boolean;
  get canRedo(): boolean;
}
```

- Semantics: `push` records the state that existed *before* a change and clears the redo stack. `undo(current)` returns the previous state and pushes `current` onto the redo stack, or `null` when there is nothing to undo. `redo(current)` is the mirror image. When the stack exceeds `limit`, the oldest entry is dropped. This class is deliberately plain TypeScript with no runes so it can be unit-tested in Node.

- [ ] **Step 1: Write the failing test**

`src/lib/editor/history.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { History } from './history';

describe('History', () => {
  it('starts empty', () => {
    const history = new History<string>();
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
    expect(history.undo('now')).toBeNull();
    expect(history.redo('now')).toBeNull();
  });

  it('undoes back through pushed states', () => {
    const history = new History<string>();
    history.push('one');
    history.push('two');

    expect(history.canUndo).toBe(true);
    expect(history.undo('three')).toBe('two');
    expect(history.undo('two')).toBe('one');
    expect(history.undo('one')).toBeNull();
    expect(history.canUndo).toBe(false);
  });

  it('redoes what was undone', () => {
    const history = new History<string>();
    history.push('one');
    const undone = history.undo('two');

    expect(undone).toBe('one');
    expect(history.canRedo).toBe(true);
    expect(history.redo('one')).toBe('two');
    expect(history.canRedo).toBe(false);
  });

  it('drops the redo stack when a new change is pushed', () => {
    const history = new History<string>();
    history.push('one');
    history.undo('two');
    history.push('one-bis');
    expect(history.canRedo).toBe(false);
    expect(history.redo('one-bis')).toBeNull();
  });

  it('keeps at most `limit` entries', () => {
    const history = new History<number>(2);
    history.push(1);
    history.push(2);
    history.push(3);
    expect(history.undo(4)).toBe(3);
    expect(history.undo(3)).toBe(2);
    expect(history.undo(2)).toBeNull();
  });

  it('clears both stacks', () => {
    const history = new History<string>();
    history.push('one');
    history.undo('two');
    history.clear();
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/editor/history.test.ts`
Expected: FAIL — `Failed to resolve import "./history"`.

- [ ] **Step 3: Write the implementation**

`src/lib/editor/history.ts`:

```ts
export class History<T> {
  private past: T[] = [];
  private future: T[] = [];

  constructor(private readonly limit: number = 50) {}

  push(previous: T): void {
    this.past.push(previous);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
  }

  undo(current: T): T | null {
    const previous = this.past.pop();
    if (previous === undefined) return null;
    this.future.push(current);
    return previous;
  }

  redo(current: T): T | null {
    const next = this.future.pop();
    if (next === undefined) return null;
    this.past.push(current);
    return next;
  }

  clear(): void {
    this.past = [];
    this.future = [];
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/editor/history.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/editor/history.ts src/lib/editor/history.test.ts
git commit -m "feat: add undo/redo history"
```

---

### Task 12: TensorFlow.js model builder

**Files:**
- Create: `src/lib/tf/buildModel.ts`
- Test: `src/lib/tf/buildModel.test.ts`

**Interfaces:**
- Consumes: `validate`, `Issue` from `../network/validate`; `Network`, `Block`, `TrainingConfig` from `../network/types`.
- Produces:

```ts
export class NetworkInvalidError extends Error {
  readonly issues: Issue[];
}

export function buildModel(net: Network): tf.Sequential;
export function compileModel(model: tf.LayersModel, training: TrainingConfig): void;
```

- `buildModel` throws `NetworkInvalidError` (carrying only the `severity: 'error'` issues) when the network is invalid, so callers can display the same plain-language messages the Issues panel shows.
- The `input` block supplies `inputShape` to the **first** layer added and no other; every later layer's input size is inferred by TensorFlow.js. The `output` block adds no layer.
- Layer mapping: `linear` → `tf.layers.dense({ units })`; `conv2d` → `tf.layers.conv2d({ filters, kernelSize, strides: stride, padding, activation: 'linear' })`; `flatten` → `tf.layers.flatten()`; `relu`/`sigmoid`/`softmax` → `tf.layers.activation({ activation: kind })`.
- `compileModel` maps `sgd` → `tf.train.sgd(learningRate)`, `adam` → `tf.train.adam(learningRate)`, `mse` → `'meanSquaredError'`, `crossEntropy` → `'categoricalCrossentropy'`. Calling it on an already-compiled model preserves weights.

- [ ] **Step 1: Write the failing test**

`src/lib/tf/buildModel.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { Network } from '../network/types';
import { NetworkInvalidError, buildModel, compileModel } from './buildModel';

let models: tf.Sequential[] = [];

function build(net: Network): tf.Sequential {
  const model = buildModel(net);
  models.push(model);
  return model;
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('buildModel', () => {
  it('builds the default MLP with the expected layers and shapes', () => {
    const model = build(createEmptyNetwork());
    expect(model.layers.map((layer) => layer.getClassName())).toEqual([
      'Dense',
      'Activation',
      'Dense',
      'Activation'
    ]);
    expect(model.inputs[0].shape).toEqual([null, 2]);
    expect(model.outputs[0].shape).toEqual([null, 2]);
  });

  it('gives only the first layer an explicit input shape', () => {
    const model = build(createEmptyNetwork());
    expect(model.layers[0].batchInputShape).toEqual([null, 2]);
    expect(model.layers[1].outputShape).toEqual([null, 8]);
    expect(model.layers[1].batchInputShape).toBeUndefined();
  });

  it('builds a convolutional chain with flattening', () => {
    const network: Network = {
      version: 1,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'conv', kind: 'conv2d', filters: 4, kernelSize: 3, stride: 1, padding: 'same' },
        { id: 'flat', kind: 'flatten' },
        { id: 'dense', kind: 'linear', units: 10 },
        { id: 'sm', kind: 'softmax' },
        { id: 'out', kind: 'output', units: 10 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
    };
    const model = build(network);
    expect(model.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  it('refuses to build an invalid network and carries the issues', () => {
    const network: Network = {
      version: 1,
      blocks: [
        { id: 'in', kind: 'input', shape: [28, 28, 1] },
        { id: 'dense', kind: 'linear', units: 8 },
        { id: 'out', kind: 'output', units: 2 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 }
    };

    expect(() => buildModel(network)).toThrow(NetworkInvalidError);

    try {
      buildModel(network);
    } catch (error) {
      const invalid = error as NetworkInvalidError;
      expect(invalid.name).toBe('NetworkInvalidError');
      expect(invalid.issues.length).toBeGreaterThan(0);
      expect(invalid.issues.every((issue) => issue.severity === 'error')).toBe(true);
      expect(invalid.issues[0].fix.length).toBeGreaterThan(0);
    }
  });

  it('runs a forward pass with the shape the network describes', () => {
    const model = build(createEmptyNetwork());
    const prediction = model.predict(tf.tensor2d([[0.1, -0.2]])) as tf.Tensor;
    expect(prediction.shape).toEqual([1, 2]);
    const values = Array.from(prediction.dataSync());
    expect(values[0] + values[1]).toBeCloseTo(1, 5);
    prediction.dispose();
  });
});

describe('compileModel', () => {
  it('preserves weights when the training configuration changes', () => {
    const net = createEmptyNetwork();
    const model = build(net);
    const before = model.getWeights().map((weight) => Array.from(weight.dataSync()));

    compileModel(model, { ...net.training, optimizer: 'sgd', learningRate: 0.5, loss: 'mse' });

    const after = model.getWeights().map((weight) => Array.from(weight.dataSync()));
    expect(after).toEqual(before);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/tf/buildModel.test.ts`
Expected: FAIL — `Failed to resolve import "./buildModel"`.

- [ ] **Step 3: Write the implementation**

`src/lib/tf/buildModel.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import type { Block, Network, TrainingConfig } from '../network/types';
import { validate, type Issue } from '../network/validate';

const LOSSES: Record<TrainingConfig['loss'], string> = {
  mse: 'meanSquaredError',
  crossEntropy: 'categoricalCrossentropy'
};

export class NetworkInvalidError extends Error {
  readonly issues: Issue[];

  constructor(issues: Issue[]) {
    super('The network has errors and cannot be built.');
    this.name = 'NetworkInvalidError';
    this.issues = issues;
  }
}

function layerFor(block: Block, inputShape: number[] | undefined): tf.layers.Layer {
  switch (block.kind) {
    case 'linear':
      return tf.layers.dense({ units: block.units, inputShape });
    case 'conv2d':
      return tf.layers.conv2d({
        filters: block.filters,
        kernelSize: block.kernelSize,
        strides: block.stride,
        padding: block.padding,
        activation: 'linear',
        inputShape
      });
    case 'flatten':
      return tf.layers.flatten({ inputShape });
    case 'relu':
    case 'sigmoid':
    case 'softmax':
      return tf.layers.activation({ activation: block.kind, inputShape });
    default:
      throw new Error(`Cannot build a layer for block kind ${block.kind}`);
  }
}

export function compileModel(model: tf.LayersModel, training: TrainingConfig): void {
  const optimizer =
    training.optimizer === 'adam'
      ? tf.train.adam(training.learningRate)
      : tf.train.sgd(training.learningRate);

  model.compile({ optimizer, loss: LOSSES[training.loss] });
}

export function buildModel(net: Network): tf.Sequential {
  const errors = validate(net).filter((issue) => issue.severity === 'error');
  if (errors.length > 0) throw new NetworkInvalidError(errors);

  const inputBlock = net.blocks.find((block) => block.kind === 'input');
  const inputShape = inputBlock && inputBlock.kind === 'input' ? inputBlock.shape : undefined;

  const layers = net.blocks.filter((block) => block.kind !== 'input' && block.kind !== 'output');

  const model = tf.sequential();
  layers.forEach((block, index) => {
    model.add(layerFor(block, index === 0 ? inputShape : undefined));
  });

  compileModel(model, net.training);
  return model;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/tf/buildModel.test.ts`
Expected: PASS. If the first run fails with `Backend name 'cpu' not found`, the test runner is missing a Node backend registration; in that case import `@tensorflow/tfjs-backend-cpu` at the top of the test file and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/lib/tf/buildModel.ts src/lib/tf/buildModel.test.ts
git commit -m "feat: add TensorFlow.js model builder and compile helper"
```

---

### Task 13: Trainer

**Files:**
- Create: `src/lib/training/Trainer.ts`
- Test: `src/lib/training/Trainer.test.ts`

**Interfaces:**
- Consumes: `buildModel` from `../tf/buildModel` (tests only); `createEmptyNetwork` from `../network/factory` (tests only).
- Produces:

```ts
export interface TrainStats {
  epoch: number;
  batch: number;
  batchLoss: number;
  epochMeanLoss: number | null;
  epochAccuracy: number | null;
}

export type YieldFn = () => Promise<void>;

export class Trainer {
  constructor(
    model: tf.LayersModel,
    data: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    yieldFn?: YieldFn
  );
  play(): Promise<void>;
  pause(): void;
  step(): Promise<void>;
  dispose(): void;
  get isPlaying(): boolean;
  get batchesPerEpoch(): number;
}
```

- Semantics: `batchesPerEpoch` is `Math.max(1, Math.ceil(exampleCount / batchSize))`. `step()` trains exactly one batch, incrementing the batch counter; when the counter reaches `batchesPerEpoch` the epoch counter advances, the batch counter resets, and `epochMeanLoss` (mean of that epoch's batch losses) and `epochAccuracy` (fraction correct on the whole dataset, computed with `argMax` over one-hot labels) are reported. In-progress steps report `epochMeanLoss: null` and `epochAccuracy: null`. `play()` runs `step()` then `yieldFn()` repeatedly until `pause()` is called or `dispose()` happens, and resolves when the loop stops — which makes it awaitable in tests. The default `yieldFn` is `() => tf.nextFrame()`.
- `model.trainOnBatch(xs, ys)` returns `Promise<number | number[]>` in TensorFlow.js 4.x, not a `Scalar`. `step()` therefore awaits it and disposes the gathered batch tensors in a `finally` block, because `tf.tidy` cannot span an `await`.
- Epoch numbering counts completed epochs: the first mid-epoch step reports `epoch: 0`, and the step that completes epoch 0 reports `epoch: 1` with `batch: 0`. With `batchesPerEpoch` 2, six steps report epochs `[0, 1, 1, 2, 2, 3]`.
- Batch sampling: maintain a shuffled pool of example indices, refilling it when exhausted, so each epoch visits every example once in random order.

- [ ] **Step 1: Write the failing test**

`src/lib/training/Trainer.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { buildModel } from '../tf/buildModel';
import { Trainer, type TrainStats } from './Trainer';

let models: tf.Sequential[] = [];
let tensors: tf.Tensor[] = [];

function makeData(count = 8): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < count; i++) {
    const positive = i % 2 === 0;
    xs.push([positive ? 0.5 : -0.5, positive ? 0.5 : -0.5]);
    ys.push(positive ? [0, 1] : [1, 0]);
  }
  const x = tf.tensor2d(xs, [count, 2]);
  const y = tf.tensor2d(ys, [count, 2]);
  tensors.push(x, y);
  return { xs: x, ys: y };
}

function makeTrainer(onStats: (stats: TrainStats) => void, batchSize = 4, yieldFn?: () => Promise<void>) {
  const model = buildModel(createEmptyNetwork());
  models.push(model);
  return new Trainer(model, makeData(), batchSize, onStats, yieldFn);
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  tensors.forEach((tensor) => tensor.dispose());
  models = [];
  tensors = [];
});

describe('Trainer bookkeeping', () => {
  it('computes batches per epoch', () => {
    expect(makeTrainer(() => {}).batchesPerEpoch).toBe(2);
  });

  it('reports a finite loss for every step', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s));

    await trainer.step();
    await trainer.step();

    expect(stats).toHaveLength(2);
    expect(stats.every((s) => Number.isFinite(s.batchLoss))).toBe(true);
  });

  it('reports null epoch statistics mid-epoch and real ones at the end', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s));

    await trainer.step();
    expect(stats[0]).toMatchObject({ epoch: 0, batch: 1, epochMeanLoss: null, epochAccuracy: null });

    await trainer.step();
    expect(stats[1].epoch).toBe(1);
    expect(stats[1].batch).toBe(0);
    expect(stats[1].epochMeanLoss).toBeGreaterThan(0);
    expect(stats[1].epochAccuracy).toBeGreaterThanOrEqual(0);
    expect(stats[1].epochAccuracy).toBeLessThanOrEqual(1);
  });

  it('advances the epoch every batchesPerEpoch steps', async () => {
    const epochs: number[] = [];
    const trainer = makeTrainer((s) => epochs.push(s.epoch));
    expect(trainer.batchesPerEpoch).toBe(2);
    for (let i = 0; i < 6; i++) await trainer.step();
    expect(epochs).toEqual([0, 1, 1, 2, 2, 3]);
  });

  it('never requests a batch larger than the dataset', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s), 100);
    expect(trainer.batchesPerEpoch).toBe(1);
    await trainer.step();
    expect(Number.isFinite(stats[0].batchLoss)).toBe(true);
  });
});

describe('Trainer play loop', () => {
  it('runs steps until paused and resolves', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      yields += 1;
      if (yields >= 3) trainer.pause();
    });

    expect(trainer.isPlaying).toBe(false);
    const loop = trainer.play();
    expect(trainer.isPlaying).toBe(true);
    await loop;

    expect(trainer.isPlaying).toBe(false);
    expect(stats).toHaveLength(3);
  });

  it('is a no-op when already playing', async () => {
    const stats: TrainStats[] = [];
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      trainer.pause();
    });

    const first = trainer.play();
    const second = trainer.play();
    await Promise.all([first, second]);

    expect(stats).toHaveLength(1);
  });

  it('stops immediately after dispose', async () => {
    const stats: TrainStats[] = [];
    let yields = 0;
    const trainer = makeTrainer((s) => stats.push(s), 4, async () => {
      yields += 1;
      if (yields >= 2) trainer.dispose();
    });

    await trainer.play();
    const seen = stats.length;
    await trainer.step();
    expect(stats).toHaveLength(seen);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/training/Trainer.test.ts`
Expected: FAIL — `Failed to resolve import "./Trainer"`.

- [ ] **Step 3: Write the implementation**

`src/lib/training/Trainer.ts`:

```ts
import * as tf from '@tensorflow/tfjs';

export interface TrainStats {
  epoch: number;
  batch: number;
  batchLoss: number;
  epochMeanLoss: number | null;
  epochAccuracy: number | null;
}

export type YieldFn = () => Promise<void>;

const nextFrame: YieldFn = () => tf.nextFrame();

export class Trainer {
  private readonly batchesPerEpochCount: number;
  private epoch = 0;
  private batch = 0;
  private batchLosses: number[] = [];
  private pool: number[] = [];
  private poolPosition = 0;
  private playing = false;
  private disposed = false;

  constructor(
    private readonly model: tf.LayersModel,
    private readonly data: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    private readonly batchSize: number,
    private readonly onStats: (stats: TrainStats) => void,
    private readonly yieldFn: YieldFn = nextFrame
  ) {
    const examples = data.xs.shape[0];
    this.batchesPerEpochCount = Math.max(1, Math.ceil(examples / batchSize));
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get batchesPerEpoch(): number {
    return this.batchesPerEpochCount;
  }

  play(): Promise<void> {
    if (this.playing || this.disposed) return Promise.resolve();
    this.playing = true;
    return this.loop();
  }

  pause(): void {
    this.playing = false;
  }

  async step(): Promise<void> {
    if (this.disposed) return;

    const indices = tf.tensor1d(this.sampleIndices(), 'int32');
    const batchXs = tf.gather(this.data.xs, indices);
    const batchYs = tf.gather(this.data.ys, indices);
    indices.dispose();

    let batchLoss: number;
    try {
      const result = await this.model.trainOnBatch(batchXs, batchYs);
      batchLoss = Array.isArray(result) ? result[0] : result;
    } finally {
      batchXs.dispose();
      batchYs.dispose();
    }

    this.batchLosses.push(batchLoss);
    this.batch += 1;

    if (this.batch >= this.batchesPerEpochCount) {
      const mean = this.batchLosses.reduce((total, value) => total + value, 0) / this.batchLosses.length;
      const accuracy = this.accuracy();
      this.epoch += 1;
      this.batch = 0;
      this.batchLosses = [];
      this.onStats({
        epoch: this.epoch,
        batch: 0,
        batchLoss,
        epochMeanLoss: mean,
        epochAccuracy: accuracy
      });
      return;
    }

    this.onStats({
      epoch: this.epoch,
      batch: this.batch,
      batchLoss,
      epochMeanLoss: null,
      epochAccuracy: null
    });
  }

  dispose(): void {
    this.disposed = true;
    this.playing = false;
  }

  private async loop(): Promise<void> {
    while (this.playing && !this.disposed) {
      await this.step();
      if (!this.playing || this.disposed) break;
      await this.yieldFn();
    }
    this.playing = false;
  }

  private sampleIndices(): number[] {
    const examples = this.data.xs.shape[0];
    const size = Math.min(this.batchSize, examples);
    const indices: number[] = [];

    while (indices.length < size) {
      if (this.poolPosition >= this.pool.length) {
        this.pool = Array.from({ length: examples }, (_, index) => index);
        for (let i = this.pool.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [this.pool[i], this.pool[j]] = [this.pool[j], this.pool[i]];
        }
        this.poolPosition = 0;
      }
      indices.push(this.pool[this.poolPosition]);
      this.poolPosition += 1;
    }

    return indices;
  }

  private accuracy(): number {
    return tf.tidy(() => {
      const logits = this.model.predict(this.data.xs) as tf.Tensor;
      const predictions = tf.argMax(logits, 1);
      const labels = tf.argMax(this.data.ys, 1);
      const correct = tf.cast(tf.equal(predictions, labels), 'float32').mean();
      return correct.dataSync()[0];
    });
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/training/Trainer.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/training/Trainer.ts src/lib/training/Trainer.test.ts
git commit -m "feat: add trainer with play, pause, step, and epoch statistics"
```

---

### Task 14: Seeded random numbers and the 2D point dataset

**Files:**
- Create: `src/lib/data/rng.ts`
- Create: `src/lib/data/points.ts`
- Create: `src/lib/data/tensors.ts`
- Test: `src/lib/data/rng.test.ts`, `src/lib/data/points.test.ts`, `src/lib/data/tensors.test.ts`

**Interfaces:**
- Consumes: nothing (except `@tensorflow/tfjs` in `tensors.ts`).
- Produces:

```ts
// rng.ts
export function mulberry32(seed: number): () => number;

// points.ts  (pure, no TensorFlow.js)
export interface Point { x: number; y: number; label: 0 | 1 }
export interface PointDataset { points: Point[]; numClasses: 2 }
export type GeneratorName = 'twoGaussians' | 'spirals' | 'xor' | 'circles';
export const GENERATOR_NAMES: GeneratorName[];
export const GENERATOR_DESCRIPTIONS: Record<GeneratorName, string>;
export function generate(name: GeneratorName, count: number, seed: number): PointDataset;
export function addPoint(dataset: PointDataset, x: number, y: number, label: 0 | 1): PointDataset;
export function clearPoints(dataset: PointDataset): PointDataset;

// tensors.ts
export function toTensors(dataset: PointDataset): { xs: tf.Tensor2D; ys: tf.Tensor2D };
```

- `mulberry32` returns a function producing numbers in `[0, 1)`. The same seed always produces the same sequence.
- The domain is `x, y ∈ [-1, 1]`, matching the rendering surface.
- `generate` always returns exactly `count` points, each labelled `0` or `1`, with `numClasses: 2`.
- `addPoint` and `clearPoints` are pure and never mutate their argument.
- `toTensors` produces `xs` shaped `[count, 2]` and one-hot `ys` shaped `[count, 2]`, both `float32`. An empty dataset produces shapes `[0, 2]` and `[0, 2]`.

- [ ] **Step 1: Write the failing tests**

`src/lib/data/rng.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { mulberry32 } from './rng';

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const sequenceA = Array.from({ length: 10 }, () => a());
    const sequenceB = Array.from({ length: 10 }, () => b());
    expect(sequenceA).toEqual(sequenceB);
  });

  it('differs between seeds', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    expect(a()).not.toBe(b());
  });

  it('stays within [0, 1)', () => {
    const rng = mulberry32(7);
    for (let i = 0; i < 500; i++) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});
```

`src/lib/data/points.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
  GENERATOR_DESCRIPTIONS,
  GENERATOR_NAMES,
  addPoint,
  clearPoints,
  generate
} from './points';

describe('GENERATOR_DESCRIPTIONS', () => {
  it.each(GENERATOR_NAMES)('describes %s', (name) => {
    expect(GENERATOR_DESCRIPTIONS[name]).toBeTruthy();
  });
});

describe('generate', () => {
  it.each(GENERATOR_NAMES)('produces the requested number of points for %s', (name) => {
    const dataset = generate(name, 120, 1);
    expect(dataset.points).toHaveLength(120);
    expect(dataset.numClasses).toBe(2);
  });

  it.each(GENERATOR_NAMES)('keeps %s inside the domain', (name) => {
    for (const point of generate(name, 200, 3).points) {
      expect(point.x).toBeGreaterThanOrEqual(-1);
      expect(point.x).toBeLessThanOrEqual(1);
      expect(point.y).toBeGreaterThanOrEqual(-1);
      expect(point.y).toBeLessThanOrEqual(1);
      expect([0, 1]).toContain(point.label);
    }
  });

  it('uses both classes', () => {
    const labels = new Set(generate('twoGaussians', 100, 5).points.map((p) => p.label));
    expect(labels).toEqual(new Set([0, 1]));
  });

  it('is reproducible for a given seed', () => {
    expect(generate('spirals', 50, 9)).toEqual(generate('spirals', 50, 9));
  });

  it('changes when the seed changes', () => {
    expect(generate('spirals', 50, 9)).not.toEqual(generate('spirals', 50, 10));
  });

  it('labels xor quadrants in a checkerboard', () => {
    for (const point of generate('xor', 400, 11).points) {
      const expected = (point.x > 0) !== (point.y > 0) ? 1 : 0;
      expect(point.label).toBe(expected);
    }
  });

  it('labels circles by distance from the centre', () => {
    for (const point of generate('circles', 400, 13).points) {
      const radius = Math.hypot(point.x, point.y);
      if (point.label === 0) expect(radius).toBeLessThanOrEqual(0.3);
      else expect(radius).toBeGreaterThanOrEqual(0.6);
    }
  });
});

describe('addPoint', () => {
  it('appends without mutating the original', () => {
    const original = generate('xor', 4, 1);
    const updated = addPoint(original, 0.25, -0.25, 1);
    expect(original.points).toHaveLength(4);
    expect(updated.points).toHaveLength(5);
    expect(updated.points[4]).toEqual({ x: 0.25, y: -0.25, label: 1 });
    expect(updated.numClasses).toBe(2);
  });
});

describe('clearPoints', () => {
  it('empties the dataset without mutating the original', () => {
    const original = generate('xor', 4, 1);
    const cleared = clearPoints(original);
    expect(original.points).toHaveLength(4);
    expect(cleared.points).toEqual([]);
    expect(cleared.numClasses).toBe(2);
  });
});
```

`src/lib/data/tensors.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { generate } from './points';
import { toTensors } from './tensors';

let created: tf.Tensor[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  created.forEach((tensor) => tensor.dispose());
  created = [];
});

describe('toTensors', () => {
  it('produces float32 tensors with the documented shapes', () => {
    const { xs, ys } = toTensors(generate('xor', 12, 1));
    created.push(xs, ys);
    expect(xs.shape).toEqual([12, 2]);
    expect(ys.shape).toEqual([12, 2]);
    expect(xs.dtype).toBe('float32');
    expect(ys.dtype).toBe('float32');
  });

  it('one-hot encodes the labels', () => {
    const { ys } = toTensors(generate('xor', 6, 1));
    created.push(ys);
    const rows = ys.arraySync() as number[][];
    for (const row of rows) {
      expect(row.reduce((total, value) => total + value, 0)).toBe(1);
      expect([0, 1]).toContain(row[0]);
    }
  });

  it('handles an empty dataset', () => {
    const { xs, ys } = toTensors({ points: [], numClasses: 2 });
    created.push(xs, ys);
    expect(xs.shape).toEqual([0, 2]);
    expect(ys.shape).toEqual([0, 2]);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/lib/data`
Expected: FAIL — `Failed to resolve import "./rng"`, `"./points"`, `"./tensors"`.

- [ ] **Step 3: Write the implementations**

`src/lib/data/rng.ts`:

```ts
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

`src/lib/data/points.ts`:

```ts
import { mulberry32 } from './rng';

export interface Point {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface PointDataset {
  points: Point[];
  numClasses: 2;
}

export type GeneratorName = 'twoGaussians' | 'spirals' | 'xor' | 'circles';

export const GENERATOR_NAMES: GeneratorName[] = ['twoGaussians', 'spirals', 'xor', 'circles'];

export const GENERATOR_DESCRIPTIONS: Record<GeneratorName, string> = {
  twoGaussians: 'Two round clusters of points, one in each corner.',
  spirals: 'Two interlocking spirals. Needs a hidden layer to separate.',
  xor: 'Points in four quadrants, labelled in a checkerboard pattern.',
  circles: 'A small disc of one class surrounded by a ring of the other.'
};

function clamp(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function twoGaussians(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, (_, index) => {
    const label: 0 | 1 = index % 2 === 0 ? 0 : 1;
    const centre = label === 0 ? -0.5 : 0.5;
    return {
      x: clamp(centre + gaussian(rng) * 0.15),
      y: clamp(centre + gaussian(rng) * 0.15),
      label
    };
  });
}

function xor(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, () => {
    const x = rng() * 2 - 1;
    const y = rng() * 2 - 1;
    return { x, y, label: ((x > 0) !== (y > 0) ? 1 : 0) as 0 | 1 };
  });
}

function circles(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, () => {
    const label: 0 | 1 = rng() < 0.5 ? 0 : 1;
    const radius = label === 0 ? rng() * 0.3 : 0.6 + rng() * 0.3;
    const angle = rng() * Math.PI * 2;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, label };
  });
}

function spirals(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, (_, index) => {
    const label: 0 | 1 = index % 2 === 0 ? 0 : 1;
    const progress = Math.floor(index / 2) / Math.max(1, count / 2);
    const radius = progress * 0.9;
    const angle = progress * 3.2 * Math.PI + label * Math.PI + (rng() - 0.5) * 0.15;
    return { x: clamp(Math.cos(angle) * radius), y: clamp(Math.sin(angle) * radius), label };
  });
}

export function generate(name: GeneratorName, count: number, seed: number): PointDataset {
  const rng = mulberry32(seed);
  const generators: Record<GeneratorName, (count: number, rng: () => number) => Point[]> = {
    twoGaussians,
    spirals,
    xor,
    circles
  };
  return { points: generators[name](count, rng), numClasses: 2 };
}

export function addPoint(dataset: PointDataset, x: number, y: number, label: 0 | 1): PointDataset {
  return { points: [...dataset.points, { x, y, label }], numClasses: 2 };
}

export function clearPoints(dataset: PointDataset): PointDataset {
  return { points: [], numClasses: dataset.numClasses };
}
```

`src/lib/data/tensors.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import type { PointDataset } from './points';

export function toTensors(dataset: PointDataset): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const count = dataset.points.length;
  const xs = tf.tensor2d(
    dataset.points.map((point) => [point.x, point.y]),
    [count, 2],
    'float32'
  );
  const ys = tf.tensor2d(
    dataset.points.map((point) => (point.label === 0 ? [1, 0] : [0, 1])),
    [count, dataset.numClasses],
    'float32'
  );
  return { xs, ys };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/lib/data`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/data
git commit -m "feat: add seeded rng, 2D point dataset, and tensor conversion"
```

---

### Task 15: Decision-boundary rendering primitives

**Files:**
- Create: `src/lib/render/palette.ts`
- Create: `src/lib/render/boundary.ts`
- Test: `src/lib/render/boundary.test.ts`

**Interfaces:**
- Consumes: `@tensorflow/tfjs` (only in `sampleGrid`).
- Produces:

```ts
// palette.ts
export interface ClassColour { hex: string; rgb: [number, number, number] }
export const CLASS_COLOURS: [ClassColour, ClassColour];
export const BACKGROUND_RGB: [number, number, number];

// boundary.ts
export const GRID_SIZE = 64;
export interface Rect { left: number; top: number; width: number; height: number }
export function clientToDomain(px: number, py: number, rect: Rect): { x: number; y: number };
export function cellCentre(gx: number, gy: number, size?: number): { x: number; y: number };
export function classesToRgba(
  classes: Int32Array,
  colours: Array<[number, number, number]>,
  size?: number
): Uint8ClampedArray;
export function sampleGrid(model: tf.LayersModel, size?: number): Int32Array;
```

- `CLASS_COLOURS` must match the `--class-0` and `--class-1` values in `src/lib/styles/tokens.css` (`#38bdf8` and `#fb7185`).
- `clientToDomain` maps a pixel position inside `rect` to the `[-1, 1]²` domain, with `+y` pointing up: `x = ((px - rect.left) / rect.width) * 2 - 1` and `y = 1 - ((py - rect.top) / rect.height) * 2`.
- `cellCentre` returns the domain coordinates of the centre of grid cell `(gx, gy)`.
- `classesToRgba` returns a `Uint8ClampedArray` of length `size * size * 4`, row-major from the top-left, fully opaque, using the colour at the class index (falling back to black for an out-of-range class).
- `sampleGrid` forwards all `size * size` cell centres through the model in one batch and returns the `argMax` class index per cell. It is synchronous and wraps everything in `tf.tidy`.

- [ ] **Step 1: Write the failing test**

`src/lib/render/boundary.test.ts`:

```ts
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { BACKGROUND_RGB, CLASS_COLOURS } from './palette';
import { GRID_SIZE, cellCentre, classesToRgba, clientToDomain, sampleGrid } from './boundary';

const RECT = { left: 10, top: 20, width: 200, height: 100 };

let models: tf.LayersModel[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterEach(() => {
  models.forEach((model) => model.dispose());
  models = [];
});

describe('palette', () => {
  it('matches the design tokens', () => {
    expect(CLASS_COLOURS[0].hex).toBe('#38bdf8');
    expect(CLASS_COLOURS[1].hex).toBe('#fb7185');
  });

  it('provides rgb triples', () => {
    expect(CLASS_COLOURS[0].rgb).toEqual([56, 189, 248]);
    expect(CLASS_COLOURS[1].rgb).toEqual([251, 113, 133]);
    expect(BACKGROUND_RGB).toHaveLength(3);
  });
});

describe('clientToDomain', () => {
  it('maps the rect corners to the domain corners', () => {
    expect(clientToDomain(10, 120, RECT)).toEqual({ x: -1, y: -1 });
    expect(clientToDomain(210, 20, RECT)).toEqual({ x: 1, y: 1 });
  });

  it('maps the centre to the origin', () => {
    const centre = clientToDomain(110, 70, RECT);
    expect(centre.x).toBeCloseTo(0, 10);
    expect(centre.y).toBeCloseTo(0, 10);
  });

  it('flips the y axis so up is positive', () => {
    expect(clientToDomain(110, 20, RECT).y).toBeCloseTo(1, 10);
    expect(clientToDomain(110, 120, RECT).y).toBeCloseTo(-1, 10);
  });
});

describe('cellCentre', () => {
  it('centres the first cell in the top-left', () => {
    const centre = cellCentre(0, 0, 2);
    expect(centre.x).toBeCloseTo(-0.5, 10);
    expect(centre.y).toBeCloseTo(0.5, 10);
  });

  it('centres the last cell in the bottom-right', () => {
    const centre = cellCentre(1, 1, 2);
    expect(centre.x).toBeCloseTo(0.5, 10);
    expect(centre.y).toBeCloseTo(-0.5, 10);
  });

  it('stays inside the domain for the default grid', () => {
    for (const gx of [0, 1, GRID_SIZE - 1]) {
      for (const gy of [0, 1, GRID_SIZE - 1]) {
        const centre = cellCentre(gx, gy);
        expect(Math.abs(centre.x)).toBeLessThan(1);
        expect(Math.abs(centre.y)).toBeLessThan(1);
      }
    }
  });
});

describe('classesToRgba', () => {
  const colours: Array<[number, number, number]> = [
    [10, 20, 30],
    [40, 50, 60]
  ];

  it('produces four bytes per cell', () => {
    expect(classesToRgba(new Int32Array(4), colours, 2)).toHaveLength(2 * 2 * 4);
  });

  it('writes opaque colours in row-major order', () => {
    const rgba = classesToRgba(Int32Array.from([0, 1, 1, 0]), colours, 2);
    expect(Array.from(rgba.slice(0, 4))).toEqual([10, 20, 30, 255]);
    expect(Array.from(rgba.slice(4, 8))).toEqual([40, 50, 60, 255]);
    expect(Array.from(rgba.slice(8, 12))).toEqual([40, 50, 60, 255]);
    expect(Array.from(rgba.slice(12, 16))).toEqual([10, 20, 30, 255]);
  });

  it('falls back to black for an unknown class', () => {
    const rgba = classesToRgba(Int32Array.from([9]), colours, 1);
    expect(Array.from(rgba)).toEqual([0, 0, 0, 255]);
  });
});

describe('sampleGrid', () => {
  it('classifies every cell', () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [2], useBias: false }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    const classes = sampleGrid(model, 8);
    expect(classes).toHaveLength(64);
    expect(Array.from(classes).every((value) => value === 0 || value === 1)).toBe(true);
  });

  it('uses the default grid size when none is given', () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
    model.add(tf.layers.softmax());
    model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
    models.push(model);

    expect(sampleGrid(model)).toHaveLength(GRID_SIZE * GRID_SIZE);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/render/boundary.test.ts`
Expected: FAIL — `Failed to resolve import "./boundary"`.

- [ ] **Step 3: Write the implementations**

`src/lib/render/palette.ts`:

```ts
export interface ClassColour {
  hex: string;
  rgb: [number, number, number];
}

export const CLASS_COLOURS: [ClassColour, ClassColour] = [
  { hex: '#38bdf8', rgb: [56, 189, 248] },
  { hex: '#fb7185', rgb: [251, 113, 133] }
];

export const BACKGROUND_RGB: [number, number, number] = [248, 250, 252];
```

`src/lib/render/boundary.ts`:

```ts
import * as tf from '@tensorflow/tfjs';

export const GRID_SIZE = 64;

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function clientToDomain(px: number, py: number, rect: Rect): { x: number; y: number } {
  return {
    x: ((px - rect.left) / rect.width) * 2 - 1,
    y: 1 - ((py - rect.top) / rect.height) * 2
  };
}

export function cellCentre(gx: number, gy: number, size: number = GRID_SIZE): { x: number; y: number } {
  return {
    x: ((gx + 0.5) / size) * 2 - 1,
    y: 1 - ((gy + 0.5) / size) * 2
  };
}

export function classesToRgba(
  classes: Int32Array,
  colours: Array<[number, number, number]>,
  size: number = GRID_SIZE
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let index = 0; index < size * size; index++) {
    const colour = colours[classes[index]] ?? [0, 0, 0];
    data[index * 4] = colour[0];
    data[index * 4 + 1] = colour[1];
    data[index * 4 + 2] = colour[2];
    data[index * 4 + 3] = 255;
  }
  return data;
}

export function sampleGrid(model: tf.LayersModel, size: number = GRID_SIZE): Int32Array {
  const cells = new Float32Array(size * size * 2);
  for (let gy = 0; gy < size; gy++) {
    for (let gx = 0; gx < size; gx++) {
      const centre = cellCentre(gx, gy, size);
      const offset = (gy * size + gx) * 2;
      cells[offset] = centre.x;
      cells[offset + 1] = centre.y;
    }
  }

  return tf.tidy(() => {
    const input = tf.tensor2d(cells, [size * size, 2]);
    const logits = model.predict(input) as tf.Tensor;
    return Int32Array.from(tf.argMax(logits, 1).dataSync());
  });
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/render/boundary.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/render
git commit -m "feat: add decision-boundary rendering primitives"
```

---

### Task 16: Update project documentation and verify the whole engine

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything built in Tasks 1-15.
- Produces: documentation that matches the shipped toolchain, and a green full verification run.

- [ ] **Step 1: Update `AGENTS.md`**

Replace the `## Status` and `## Stack` sections with:

```markdown
## Status

Engine complete and unit-tested: pure network domain modules (`src/lib/network/`),
the TensorFlow.js model builder (`src/lib/tf/`), the trainer
(`src/lib/training/`), the 2D dataset (`src/lib/data/`), and the
decision-boundary renderer (`src/lib/render/`). The editor UI and the MLP example
page are not built yet.

## Stack

- SvelteKit 2 + Svelte 5 (runes) + Vite + TypeScript (strict), npm
- `@tensorflow/tfjs` for model definition and training
- `@xyflow/svelte` for the network canvas
- `@sveltejs/adapter-static`; all routes prerendered, static output only, no backend
- Plain CSS with design tokens in `src/lib/styles/tokens.css`
- Vitest for unit tests

## Commands

- `npm run dev` — development server
- `npm run build` — static build
- `npm run check` — svelte-check and TypeScript
- `npm run lint` — ESLint
- `npm test` — Vitest (run once)
- `npm run format` — Prettier

## Architecture rules

- `src/lib/network/**` is pure: no Svelte, no TensorFlow.js, no DOM. It must run in
  plain Node and is the single source of truth for network validity and shapes.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`,
  `src/lib/training/**`, and `src/lib/data/tensors.ts`.
- No block stores its input dimension; inputs derive from the previous block's
  output.
- Every validation issue carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.

Design reference: `docs/superpowers/specs/2026-09-16-visnet-design.md`.
```

- [ ] **Step 2: Update `README.md`**

In the "Available modules" list, add `flatten` and `softmax`:

```markdown
Available modules:
- linear
- convolutional
- flatten
- ReLU
- Sigmoid
- Softmax
```

Then append:

```markdown
## Architecture

The engine is framework-independent. `src/lib/network/` defines networks, infers
the tensor shape flowing through every block, and reports validation problems as
plain-language errors with a suggested fix. TensorFlow.js is confined to
`src/lib/tf/`, `src/lib/training/`, and `src/lib/data/tensors.ts`.

The editor UI and example pages are built on top of this engine and are not part
of the engine itself.

## Development

npm run dev      # development server
npm run build    # static build
npm run check    # type checking
npm run lint     # linting
npm test         # unit tests
```

- [ ] **Step 3: Run the full verification suite**

Run: `npm run lint`
Expected: no errors.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: all test files pass, 0 failures. The count should be at least 60 tests across the files created in Tasks 1-15.

Run: `npm run build`
Expected: build succeeds and `build/index.html` exists.

If any command fails, fix the cause before committing. Do not adjust a test to match a wrong implementation; fix the implementation.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: document the shipped engine, stack, and commands"
```

---

## Plan Self-Review

**Spec coverage.** Spec sections 3 (stack/tooling), 4 (domain model), 5 (pure modules), 6 (validation messages), 10 (model building), 11 (training), 12 (data), 13 (boundary rendering), 18 (testing), and the engine-relevant parts of 20 (implementation order) are covered by Tasks 1-16. Sections 2, 7, 8, 9, 14, 15, 16, 17, 19 (UX principles, store, canvas projection, shape visibility, loss chart, persistence, error handling, routes, manual checklist) are Plan B. Spec section 18 lists `editor/networkStore.svelte.ts` and `history.svelte.ts` as tested; this plan deliberately makes history a plain `history.ts` so it is testable without runes, and leaves the runes store to Plan B with manual verification. The spec should be updated to say `history.ts` when Plan B is written.

**Placeholder scan.** No "TBD", "TODO", or "similar to Task N" references. Every code step contains complete code. Every command has an expected result.

**Type consistency.** `Network`/`Block` are defined once in Task 2 and imported everywhere. `ShapeResult`/`ShapeInfo`/`EdgeShape` are defined in Task 6 and consumed by Task 10. `Issue` is defined in Task 7 and consumed by Task 12. `TrainStats` is defined in Task 13 and is what Plan B's UI consumes. `CLASS_COLOURS` is defined in Task 15 and matches the token names created in Task 1. `newBlockId` is exported in Task 3 and used by `createEmptyNetwork` in the same file.

**Known deliberate deviation.** Spec section 11 declares `epochMeanLoss` and `epochAccuracy` as `number`; Task 13 makes them `number | null` so mid-epoch steps are unambiguous. Plan B's UI must treat `null` as "not yet available". The spec should be updated to match.

