# VisNet CNN Data Pipeline Implementation Plan (C1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the digit-image data pipeline for the CNN example — a prep script that downloads MNIST into a gitignored directory, a binary format with a tested parser, image-to-tensor conversion — and fix the four recorded issues that the CNN example would otherwise expose.

**Architecture:** The prep script writes a compact binary per split; a pure parser in `src/lib/data/` reads it back, and a thin fetch layer turns a missing or corrupt asset into a typed error the page can explain. The format's writer and reader are tied together by a round-trip test so they cannot drift. The fixes touch the editor store (`clampNetwork` on every mutation, an initial network for `reset`), weight persistence (per-example ids), and storage keys (explicit rather than hardcoded).

**Tech Stack:** Node ESM (no dependencies) for the prep script, TypeScript strict, Vitest (engine project in Node, ui project in jsdom), `@tensorflow/tfjs`.

## Global Constraints

- Package manager is **npm**. Do not use pnpm, yarn, or bun.
- TypeScript strict mode is on; `npm run check` must report 0 errors and 0 warnings.
- `src/lib/network/**` stays pure: no Svelte, no `@tensorflow/tfjs`, no DOM.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`, `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`, and `src/lib/persist/weights.ts`. This plan adds no new import site.
- No backend, no server code. All routes prerendered; the build output is static files.
- **No dataset is committed.** `static/mnist/` is gitignored, and `dev` and `build` never touch the network.
- Do not add code comments unless a non-obvious constraint requires one.
- No `any`.
- Plain CSS with the tokens in `src/lib/styles/tokens.css`; no new tokens.
- Commit at the end of every task with the exact message shown in that task.
- Do not commit anything under `.superpowers/`.
- Design reference: `docs/superpowers/specs/2026-09-16-visnet-cnn-design.md`.

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/lib/data/images.ts` | Pure: the `ImageDataset` type, the binary parser, single-image access, format constants |
| `src/lib/data/mnist.ts` | Fetching both splits and turning a missing or corrupt asset into `ImageDataUnavailableError` |
| `src/lib/data/tensors.ts` | Modify: gains `imagesToTensors` beside the existing point conversion |
| `scripts/prepare-mnist.mjs` | Download MNIST, parse IDX, write the two split files; exports its pure parts |
| `scripts/prepare-mnist.test.ts` | IDX parsing, the format round trip against the real parser |
| `src/lib/editor/networkStore.svelte.ts` | Modify: `clampNetwork` on every mutation, an initial network for `reset` |
| `src/lib/persist/weights.ts` | Modify: weights addressed by an explicit example id |
| `src/lib/persist/storage.ts` | Modify: storage keys passed in rather than hardcoded |
| `src/lib/examples/mlp/example.ts` | Modify: exports the MLP's storage keys |
| `src/lib/examples/mlp/runtime.ts` | Modify: passes the `mlp` weight id |
| `src/routes/examples/mlp/+page.svelte` | Modify: passes the MLP storage keys |
| `.gitignore` | Modify: ignores `static/mnist/` |
| `vite.config.ts` | Modify: the engine project also collects `scripts/**/*.test.ts` |
| `package.json` | Modify: adds the `data:mnist` script |
| `AGENTS.md` | Modify: the new command and the gitignored data directory |

---

### Task 1: The digit asset format parser

**Files:**
- Create: `src/lib/data/images.ts`
- Test: `src/lib/data/images.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:

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

- The format is little-endian: 4 bytes of ASCII magic `VSNT`, then version, rows, cols and numClasses as single bytes, then a `uint32` count at offset 8, a reserved `uint32` at offset 12, then `count * rows * cols` grayscale pixel bytes, then `count` label bytes.
- `parseSplit` returns `null` — never throws — for: a buffer shorter than 16 bytes; a wrong magic; a version other than 1; a non-zero reserved field; zero rows, cols, numClasses or count; a length that is not exactly `16 + count * rows * cols + count`; or any label `>= numClasses`.
- The returned dataset owns copies of the pixel and label bytes, so it does not alias the input buffer.
- `imageAt` returns the `rows * cols` slice for one image and throws a `RangeError` for a non-integer or out-of-range index.

- [ ] **Step 1: Write the failing test**

`src/lib/data/images.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { HEADER_BYTES, MAGIC, parseSplit, imageAt, type ImageDataset } from './images';

const ROWS = 2;
const COLS = 3;
const CLASSES = 3;

function build(options: {
  count?: number;
  rows?: number;
  cols?: number;
  numClasses?: number;
  magic?: string;
  version?: number;
  reserved?: number;
  extraBytes?: number;
  labels?: number[];
} = {}): ArrayBuffer {
  const count = options.count ?? 2;
  const rows = options.rows ?? ROWS;
  const cols = options.cols ?? COLS;
  const numClasses = options.numClasses ?? CLASSES;
  const labels = options.labels ?? Array.from({ length: count }, (_, i) => i % numClasses);
  const pixelBytes = count * rows * cols;

  const buffer = new ArrayBuffer(HEADER_BYTES + pixelBytes + count + (options.extraBytes ?? 0));
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  const magic = options.magic ?? MAGIC;
  for (let i = 0; i < 4; i++) view.setUint8(i, magic.charCodeAt(i) ?? 0);
  view.setUint8(4, options.version ?? 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, options.reserved ?? 0, true);

  for (let i = 0; i < pixelBytes; i++) bytes[HEADER_BYTES + i] = i % 256;
  for (let i = 0; i < count; i++) bytes[HEADER_BYTES + pixelBytes + i] = labels[i];
  return buffer;
}

function dataset(overrides: Partial<ImageDataset> = {}): ImageDataset {
  const parsed = parseSplit(build());
  if (!parsed) throw new Error('fixture failed to parse');
  return { ...parsed, ...overrides };
}

describe('parseSplit', () => {
  it('reads the header and the pixel and label bytes', () => {
    const parsed = parseSplit(build());
    expect(parsed).not.toBeNull();
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS, numClasses: CLASSES });
    expect(parsed?.pixels).toHaveLength(2 * ROWS * COLS);
    expect(Array.from(parsed?.labels ?? [])).toEqual([0, 1]);
  });

  it('copies the bytes rather than aliasing the buffer', () => {
    const buffer = build();
    const parsed = parseSplit(buffer);
    new Uint8Array(buffer)[HEADER_BYTES] = 99;
    expect(parsed?.pixels[0]).toBe(0);
  });

  it('rejects a buffer shorter than the header', () => {
    expect(parseSplit(new ArrayBuffer(HEADER_BYTES - 1))).toBeNull();
  });

  it('rejects a wrong magic', () => {
    expect(parseSplit(build({ magic: 'NOPE' }))).toBeNull();
  });

  it('rejects an unsupported version', () => {
    expect(parseSplit(build({ version: 2 }))).toBeNull();
  });

  it('rejects a non-zero reserved field', () => {
    expect(parseSplit(build({ reserved: 7 }))).toBeNull();
  });

  it('rejects zero dimensions, classes or count', () => {
    expect(parseSplit(build({ rows: 0 }))).toBeNull();
    expect(parseSplit(build({ cols: 0 }))).toBeNull();
    expect(parseSplit(build({ numClasses: 0 }))).toBeNull();
    expect(parseSplit(build({ count: 0, labels: [] }))).toBeNull();
  });

  it('rejects trailing bytes', () => {
    expect(parseSplit(build({ extraBytes: 4 }))).toBeNull();
  });

  it('rejects a label beyond the class count', () => {
    expect(parseSplit(build({ numClasses: 2, labels: [0, 5] }))).toBeNull();
  });
});

describe('imageAt', () => {
  it('returns that image slice', () => {
    const parsed = dataset();
    expect(Array.from(imageAt(parsed, 1))).toEqual([6, 7, 8, 9, 10, 11]);
  });

  it('rejects an out-of-range or non-integer index', () => {
    const parsed = dataset();
    expect(() => imageAt(parsed, 2)).toThrow(RangeError);
    expect(() => imageAt(parsed, -1)).toThrow(RangeError);
    expect(() => imageAt(parsed, 0.5)).toThrow(RangeError);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/data/images.test.ts`
Expected: FAIL — `Failed to resolve import "./images"`.

- [ ] **Step 3: Write the implementation**

`src/lib/data/images.ts`:

```ts
export interface ImageDataset {
  count: number;
  rows: number;
  cols: number;
  numClasses: number;
  pixels: Uint8Array;
  labels: Uint8Array;
}

export const MAGIC = 'VSNT';
export const FORMAT_VERSION = 1;
export const HEADER_BYTES = 16;

function magicAt(view: DataView): string {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
}

export function parseSplit(buffer: ArrayBuffer): ImageDataset | null {
  if (buffer.byteLength < HEADER_BYTES) return null;

  const view = new DataView(buffer);
  if (magicAt(view) !== MAGIC) return null;
  if (view.getUint8(4) !== FORMAT_VERSION) return null;

  const rows = view.getUint8(5);
  const cols = view.getUint8(6);
  const numClasses = view.getUint8(7);
  const count = view.getUint32(8, true);
  const reserved = view.getUint32(12, true);

  if (rows === 0 || cols === 0 || numClasses === 0 || count === 0) return null;
  if (reserved !== 0) return null;

  const pixelBytes = count * rows * cols;
  if (!Number.isSafeInteger(pixelBytes)) return null;
  if (buffer.byteLength !== HEADER_BYTES + pixelBytes + count) return null;

  const pixels = new Uint8Array(buffer, HEADER_BYTES, pixelBytes).slice();
  const labels = new Uint8Array(buffer, HEADER_BYTES + pixelBytes, count).slice();
  for (const label of labels) {
    if (label >= numClasses) return null;
  }

  return { count, rows, cols, numClasses, pixels, labels };
}

export function imageAt(dataset: ImageDataset, index: number): Uint8Array {
  if (!Number.isInteger(index) || index < 0 || index >= dataset.count) {
    throw new RangeError(`Image ${index} is out of range for a dataset of ${dataset.count}.`);
  }
  const size = dataset.rows * dataset.cols;
  return dataset.pixels.subarray(index * size, (index + 1) * size);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/data/images.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/data/images.ts src/lib/data/images.test.ts
git commit -m "feat: add the digit asset format parser"
```

---

### Task 2: The MNIST prep script

**Files:**
- Create: `scripts/prepare-mnist.mjs`
- Test: `scripts/prepare-mnist.test.ts`
- Modify: `package.json` (add the `data:mnist` script)
- Modify: `.gitignore` (ignore `static/mnist/`)
- Modify: `vite.config.ts` (the engine project also collects `scripts/**/*.test.ts`)

**Interfaces:**
- Consumes: nothing at runtime. The script is plain ESM and cannot import the TypeScript module, so it restates the magic bytes, the version, and the header size. **The round-trip test is what keeps that restatement honest** — if `src/lib/data/images.ts` ever changes its format, the test fails.
- Produces:
  - `parseIdxImages(buffer: ArrayBuffer): { count: number; rows: number; cols: number; pixels: Uint8Array }` — IDX is **big-endian**; magic `0x00000803`.
  - `parseIdxLabels(buffer: ArrayBuffer): Uint8Array` — magic `0x00000801`.
  - `encodeSplit(dataset: { count; rows; cols; numClasses; pixels; labels }): ArrayBuffer` — writes the little-endian format Task 1 parses.
  - `npm run data:mnist` — downloads, prepares, and writes `static/mnist/train.bin` and `static/mnist/test.bin`.

**Behaviour:**
- Sources: `https://storage.googleapis.com/cvdf-datasets/mnist/` + `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
- Flags: `--train=N` (default 1000), `--test=M` (default 200), `--force`.
- Idempotent: when both outputs exist and `--force` is absent, it prints that the data is already prepared and exits **without any network access**.
- Creates the output directory if needed, prints the source, counts, dimensions, bytes written, the output paths, and the NIST attribution.
- Exits non-zero with a clear message when a download fails, the IDX magic is wrong, or a requested count exceeds the file's contents.
- Runs `main()` only when executed directly, so the exports are testable without touching the network.

- [ ] **Step 1: Write the failing test**

`scripts/prepare-mnist.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { parseSplit } from '../src/lib/data/images';
import { encodeSplit, parseIdxImages, parseIdxLabels } from './prepare-mnist.mjs';

const ROWS = 2;
const COLS = 3;

function idxImages(count: number, rows = ROWS, cols = COLS): ArrayBuffer {
  const buffer = new ArrayBuffer(16 + count * rows * cols);
  const view = new DataView(buffer);
  view.setUint32(0, 0x00000803, false);
  view.setUint32(4, count, false);
  view.setUint32(8, rows, false);
  view.setUint32(12, cols, false);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < count * rows * cols; i++) bytes[16 + i] = i % 256;
  return buffer;
}

function idxLabels(labels: number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(8 + labels.length);
  const view = new DataView(buffer);
  view.setUint32(0, 0x00000801, false);
  view.setUint32(4, labels.length, false);
  const bytes = new Uint8Array(buffer);
  labels.forEach((label, i) => (bytes[8 + i] = label));
  return buffer;
}

describe('parseIdxImages', () => {
  it('reads the big-endian header and pixels', () => {
    const parsed = parseIdxImages(idxImages(2));
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS });
    expect(parsed.pixels).toHaveLength(2 * ROWS * COLS);
    expect(parsed.pixels[0]).toBe(0);
  });

  it('rejects a wrong magic', () => {
    const buffer = new ArrayBuffer(16);
    new DataView(buffer).setUint32(0, 0x00000801, false);
    expect(() => parseIdxImages(buffer)).toThrow(/image/i);
  });

  it('rejects a truncated body', () => {
    const buffer = idxImages(2);
    expect(() => parseIdxImages(buffer.slice(0, buffer.byteLength - 1))).toThrow(/truncated/i);
  });
});

describe('parseIdxLabels', () => {
  it('reads the big-endian header and labels', () => {
    expect(Array.from(parseIdxLabels(idxLabels([3, 7])))).toEqual([3, 7]);
  });

  it('rejects a wrong magic', () => {
    expect(() => parseIdxLabels(idxImages(1))).toThrow(/label/i);
  });

  it('rejects a truncated body', () => {
    const buffer = idxLabels([1, 2]);
    expect(() => parseIdxLabels(buffer.slice(0, buffer.byteLength - 1))).toThrow(/truncated/i);
  });
});

describe('encodeSplit', () => {
  it('round-trips through the parser the browser uses', () => {
    const encoded = encodeSplit({
      count: 2,
      rows: ROWS,
      cols: COLS,
      numClasses: 3,
      pixels: Uint8Array.from({ length: 2 * ROWS * COLS }, (_, i) => i % 256),
      labels: Uint8Array.from([0, 2])
    });

    const parsed = parseSplit(encoded);
    expect(parsed).not.toBeNull();
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS, numClasses: 3 });
    expect(Array.from(parsed?.labels ?? [])).toEqual([0, 2]);
    expect(parsed?.pixels[0]).toBe(0);
    expect(parsed?.pixels[5]).toBe(5);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run scripts/prepare-mnist.test.ts`
Expected: FAIL — no test files found, because the engine project's `include` only covers `src/**`. Widen it in Step 3 before re-running.

- [ ] **Step 3: Widen the engine project's include**

In `vite.config.ts`, change the `engine` project's `include` to:

```ts
          include: ['src/**/*.test.ts', 'scripts/**/*.test.ts'],
```

Then run: `npx vitest run scripts/prepare-mnist.test.ts`
Expected: FAIL — `Failed to resolve import "./prepare-mnist.mjs"`.

- [ ] **Step 4: Write the script**

`scripts/prepare-mnist.mjs`:

```js
#!/usr/bin/env node
import { access, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

const SOURCE = 'https://storage.googleapis.com/cvdf-datasets/mnist';
const OUT_DIR = fileURLToPath(new URL('../static/mnist/', import.meta.url));
const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz'
};
const MAGIC_IMAGES = 0x00000803;
const MAGIC_LABELS = 0x00000801;

export function parseIdxImages(buffer) {
  if (buffer.byteLength < 16) throw new Error('Truncated IDX image header.');
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== MAGIC_IMAGES) throw new Error('Not an IDX image file.');
  const count = view.getUint32(4, false);
  const rows = view.getUint32(8, false);
  const cols = view.getUint32(12, false);
  const expected = 16 + count * rows * cols;
  if (buffer.byteLength !== expected) {
    throw new Error(`Truncated image data: ${buffer.byteLength} of ${expected} bytes.`);
  }
  return { count, rows, cols, pixels: new Uint8Array(buffer, 16, count * rows * cols) };
}

export function parseIdxLabels(buffer) {
  if (buffer.byteLength < 8) throw new Error('Truncated IDX label header.');
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== MAGIC_LABELS) throw new Error('Not an IDX label file.');
  const count = view.getUint32(4, false);
  if (buffer.byteLength !== 8 + count) {
    throw new Error(`Truncated label data: ${buffer.byteLength} of ${8 + count} bytes.`);
  }
  return new Uint8Array(buffer, 8, count);
}

export function encodeSplit({ count, rows, cols, numClasses, pixels, labels }) {
  const pixelBytes = count * rows * cols;
  const buffer = new ArrayBuffer(16 + pixelBytes + count);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  bytes.set([0x56, 0x53, 0x4e, 0x54], 0);
  view.setUint8(4, 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true);
  bytes.set(pixels.subarray(0, pixelBytes), 16);
  bytes.set(labels.subarray(0, count), 16 + pixelBytes);
  return buffer;
}

async function download(name) {
  const response = await fetch(`${SOURCE}/${name}`);
  if (!response.ok) {
    throw new Error(`Could not download ${name}: HTTP ${response.status}.`);
  }
  const raw = gunzipSync(new Uint8Array(await response.arrayBuffer()));
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
}

function take(limit, label, images, labels) {
  const available = Math.min(images.count, labels.length);
  if (limit > available) {
    throw new Error(`Asked for ${limit} ${label} images but the file holds ${available}.`);
  }
  const size = images.rows * images.cols;
  return {
    count: limit,
    rows: images.rows,
    cols: images.cols,
    numClasses: 10,
    pixels: images.pixels.slice(0, limit * size),
    labels: labels.slice(0, limit)
  };
}

function flag(args, name, fallback) {
  const match = args.find((arg) => arg.startsWith(`${name}=`));
  if (!match) return fallback;
  const value = Number(match.slice(name.length + 1));
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} needs a positive whole number.`);
  }
  return value;
}

async function present(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const force = args.includes('--force');

  const trainPath = join(OUT_DIR, 'train.bin');
  const testPath = join(OUT_DIR, 'test.bin');

  if (!force && (await present(trainPath)) && (await present(testPath))) {
    console.log(`Already prepared at ${OUT_DIR}. Pass --force to download again.`);
    return;
  }

  const trainCount = flag(args, '--train', 1000);
  const testCount = flag(args, '--test', 200);

  console.log(`Downloading MNIST from ${SOURCE}`);
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    download(FILES.trainImages).then(parseIdxImages),
    download(FILES.trainLabels).then(parseIdxLabels),
    download(FILES.testImages).then(parseIdxImages),
    download(FILES.testLabels).then(parseIdxLabels)
  ]);

  const train = take(trainCount, 'training', trainImages, trainLabels);
  const test = take(testCount, 'test', testImages, testLabels);

  const trainBytes = encodeSplit(train);
  const testBytes = encodeSplit(test);

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(trainPath, new Uint8Array(trainBytes));
  await writeFile(testPath, new Uint8Array(testBytes));

  console.log(`Wrote ${train.count} training and ${test.count} test digits, ${train.rows}x${train.cols}.`);
  console.log(`  ${trainPath} (${trainBytes.byteLength} bytes)`);
  console.log(`  ${testPath} (${testBytes.byteLength} bytes)`);
  console.log('MNIST is a derivative of the NIST Special Database 19.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
```

- [ ] **Step 5: Add the npm script and the gitignore entry**

In `package.json`'s `scripts`, add:

```json
    "data:mnist": "node scripts/prepare-mnist.mjs",
```

In `.gitignore`, add:

```
# downloaded MNIST subset (never committed)
static/mnist/
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `npx vitest run scripts/prepare-mnist.test.ts`
Expected: PASS, 7 tests.

- [ ] **Step 7: Verify the script and the tooling**

Run: `npm run lint`
Expected: no errors. If ESLint rejects the `.mjs` file, fix the cause rather than excluding it, and report what you changed.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`. `checkJs` is on, so if the generated SvelteKit tsconfig pulls `scripts/` into the program and reports errors in the `.mjs` file, add `"exclude": ["scripts"]` to the `compilerOptions`-bearing `tsconfig.json` and report it.

Run: `npm run data:mnist -- --train=50 --test=20`
Expected: it downloads, prints the counts and paths, and writes both files. Confirm with `ls -l static/mnist`.

Run: `npm run data:mnist`
Expected: `Already prepared at …`. Confirm no network access by running it with the network unavailable if you can; otherwise confirm from the code path that it returns before any `fetch`.

Run: `npm run data:mnist -- --train=50 --test=20 --force`
Expected: downloads again, overwriting the previous subset.

Then restore the defaults, so the working tree is left holding the subset the
example actually expects:

Run: `npm run data:mnist -- --force`
Expected: writes 1,000 training and 200 test digits.

Run: `npm run build`
Expected: build succeeds. Note that `static/mnist/` is copied into `build/`, which is expected.

- [ ] **Step 8: Commit**

```bash
git add scripts/prepare-mnist.mjs scripts/prepare-mnist.test.ts package.json .gitignore vite.config.ts
git commit -m "feat: add the MNIST prep script"
```

---

### Task 3: Loading the prepared data

**Files:**
- Create: `src/lib/data/mnist.ts`
- Test: `src/lib/data/mnist.test.ts`

**Interfaces:**
- Consumes: `parseSplit`, `ImageDataset` from `./images`.
- Produces:

```ts
export interface MnistData {
  train: ImageDataset;
  test: ImageDataset;
}

export class ImageDataUnavailableError extends Error {}

export async function loadMnistData(base?: string): Promise<MnistData>;
```

- `loadMnistData` fetches `${base}/train.bin` and `${base}/test.bin` in parallel (`base` defaults to `/mnist`) and throws `ImageDataUnavailableError` when a request fails, a response is not `ok`, or a buffer fails to parse.
- The error's default message names the prep command, because a missing asset is the only realistic cause: `The digit images are not prepared. Run `npm run data:mnist`, then reload.`

- [ ] **Step 1: Write the failing test**

`src/lib/data/mnist.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { HEADER_BYTES, MAGIC } from './images';
import { ImageDataUnavailableError, loadMnistData } from './mnist';

function split(count = 2): ArrayBuffer {
  const rows = 2;
  const cols = 2;
  const numClasses = 3;
  const buffer = new ArrayBuffer(HEADER_BYTES + count * rows * cols + count);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < 4; i++) view.setUint8(i, MAGIC.charCodeAt(i));
  view.setUint8(4, 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true);
  for (let i = 0; i < count; i++) bytes[HEADER_BYTES + count * rows * cols + i] = i % numClasses;
  return buffer;
}

function ok(buffer: ArrayBuffer): Response {
  return { ok: true, arrayBuffer: async () => buffer } as unknown as Response;
}

function notFound(): Response {
  return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadMnistData', () => {
  it('loads both splits', async () => {
    const fetchMock = vi.fn(async () => ok(split(2)));
    vi.stubGlobal('fetch', fetchMock);

    const data = await loadMnistData();

    expect(data.train.count).toBe(2);
    expect(data.test.count).toBe(2);
    expect(fetchMock).toHaveBeenCalledWith('/mnist/train.bin');
    expect(fetchMock).toHaveBeenCalledWith('/mnist/test.bin');
  });

  it('honours a custom base path', async () => {
    const fetchMock = vi.fn(async () => ok(split(2)));
    vi.stubGlobal('fetch', fetchMock);

    await loadMnistData('/assets/digits');

    expect(fetchMock).toHaveBeenCalledWith('/assets/digits/train.bin');
  });

  it('reports a missing asset with the prep command', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => notFound()));

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
    await expect(loadMnistData()).rejects.toThrow(/npm run data:mnist/);
  });

  it('reports a corrupt asset', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok(new ArrayBuffer(4))));

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
  });

  it('reports a failed request', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/data/mnist.test.ts`
Expected: FAIL — `Failed to resolve import "./mnist"`.

- [ ] **Step 3: Write the implementation**

`src/lib/data/mnist.ts`:

```ts
import { parseSplit, type ImageDataset } from './images';

export interface MnistData {
  train: ImageDataset;
  test: ImageDataset;
}

export class ImageDataUnavailableError extends Error {
  constructor(
    message = 'The digit images are not prepared. Run `npm run data:mnist`, then reload.'
  ) {
    super(message);
    this.name = 'ImageDataUnavailableError';
  }
}

async function loadSplit(url: string): Promise<ImageDataset> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new ImageDataUnavailableError();
  }
  if (!response.ok) throw new ImageDataUnavailableError();

  const parsed = parseSplit(await response.arrayBuffer());
  if (!parsed) throw new ImageDataUnavailableError();
  return parsed;
}

export async function loadMnistData(base = '/mnist'): Promise<MnistData> {
  const [train, test] = await Promise.all([
    loadSplit(`${base}/train.bin`),
    loadSplit(`${base}/test.bin`)
  ]);
  return { train, test };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/data/mnist.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/data/mnist.ts src/lib/data/mnist.test.ts
git commit -m "feat: load the prepared digit data with a clear missing-asset error"
```

---

### Task 4: Image tensors

**Files:**
- Modify: `src/lib/data/tensors.ts`
- Test: `src/lib/data/tensors.test.ts` (extend)

**Interfaces:**
- Consumes: `ImageDataset`, `imageAt` from `./images`.
- Produces:

```ts
export function imagesToTensors(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor2D };
```

- `xs` is `[n, rows, cols, 1]` float32 with each pixel divided by 255; `ys` is one-hot `[n, numClasses]` float32.
- `indices` selects a subset and defaults to every image, in order.
- The caller owns the returned tensors.
- This lives in `tensors.ts` rather than a new file because that module's job is already "turn our datasets into tensors", and because `data/tensors.ts` is an approved TensorFlow.js import site — a new file would need a sixth allow-list entry.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/data/tensors.test.ts`:

```ts
import type { ImageDataset } from './images';
import { imagesToTensors } from './tensors';

function images(count = 3, rows = 2, cols = 2, numClasses = 3): ImageDataset {
  const size = rows * cols;
  const pixels = new Uint8Array(count * size);
  for (let i = 0; i < pixels.length; i++) pixels[i] = i === 0 ? 255 : i % 256;
  return {
    count,
    rows,
    cols,
    numClasses,
    pixels,
    labels: Uint8Array.from({ length: count }, (_, i) => i % numClasses)
  };
}

describe('imagesToTensors', () => {
  it('produces a 4d image batch and a one-hot label batch', () => {
    const { xs, ys } = imagesToTensors(images());
    created.push(xs, ys);

    expect(xs.shape).toEqual([3, 2, 2, 1]);
    expect(ys.shape).toEqual([3, 3]);
    expect(xs.dtype).toBe('float32');
    expect(ys.dtype).toBe('float32');
  });

  it('scales pixels into 0..1', () => {
    const { xs } = imagesToTensors(images());
    created.push(xs);

    const values = Array.from(xs.dataSync());
    expect(values[0]).toBe(1);
    expect(values[1]).toBeCloseTo(1 / 255, 6);
  });

  it('one-hot encodes the labels', () => {
    const { ys } = imagesToTensors(images());
    created.push(ys);

    const rows = ys.arraySync() as number[][];
    expect(rows).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]);
  });

  it('selects a subset in the given order', () => {
    const dataset = images();
    const { xs, ys } = imagesToTensors(dataset, [2, 0]);
    created.push(xs, ys);

    expect(xs.shape).toEqual([2, 2, 2, 1]);
    expect(ys.arraySync()).toEqual([
      [0, 0, 1],
      [1, 0, 0]
    ]);
    expect(Array.from(xs.dataSync()).slice(0, 4)).toEqual(
      Array.from(dataset.pixels.slice(8, 12), (value) => value / 255)
    );
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/data/tensors.test.ts`
Expected: FAIL — `imagesToTensors is not a function` / no exported member.

- [ ] **Step 3: Write the implementation**

Add to `src/lib/data/tensors.ts`, keeping the existing point conversion untouched:

```ts
export function imagesToTensors(
  dataset: ImageDataset,
  indices?: number[]
): { xs: tf.Tensor4D; ys: tf.Tensor2D } {
  const selected = indices ?? Array.from({ length: dataset.count }, (_, index) => index);
  const size = dataset.rows * dataset.cols;
  const pixels = new Float32Array(selected.length * size);
  const oneHot = new Float32Array(selected.length * dataset.numClasses);

  selected.forEach((imageIndex, row) => {
    const source = imageAt(dataset, imageIndex);
    for (let i = 0; i < size; i++) pixels[row * size + i] = source[i] / 255;
    oneHot[row * dataset.numClasses + dataset.labels[imageIndex]] = 1;
  });

  return {
    xs: tf.tensor4d(pixels, [selected.length, dataset.rows, dataset.cols, 1]),
    ys: tf.tensor2d(oneHot, [selected.length, dataset.numClasses])
  };
}
```

The file's imports become:

```ts
import * as tf from '@tensorflow/tfjs';
import { imageAt, type ImageDataset } from './images';
import type { PointDataset } from './points';
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/lib/data/tensors.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/data/tensors.ts src/lib/data/tensors.test.ts
git commit -m "feat: convert the image dataset to tensors"
```

---

### Task 5: Clamp the network on every mutation

**Files:**
- Modify: `src/lib/editor/networkStore.svelte.ts`
- Test: `src/lib/editor/networkStore.svelte.test.ts` (extend)

**Interfaces:**
- Consumes: `clampNetwork` from `../network/constraints`.
- Produces: no new public API. `addBlock`, `moveBlock`, and `load` now clamp the resulting network and append every correction to `announcements`, exactly as `updateBlock` already does.

**Why:** `clampNetwork` currently runs only from `updateBlock`, so `addBlock` can insert a `conv2d` whose default 3×3 kernel is larger than a small input, leaving an error the user did not cause and no announcement. That is unreachable while `conv2d` is absent from the palette; the CNN example makes it reachable.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/editor/networkStore.svelte.test.ts`:

```ts
describe('clamping on every mutation', () => {
  it('clamps a convolution added to a small image and announces it', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[0].id, { shape: [2, 2, 1] });
    instance.dismissAnnouncements();

    const created = instance.addBlock('conv2d', 1);

    expect(instance.network.blocks[1]).toMatchObject({ id: created, kernelSize: 2 });
    expect(instance.announcements).toEqual([
      'Kernel size changed from 3 to 2 because the incoming data is 2×2.'
    ]);
  });

  it('clamps a network that is loaded already out of range', () => {
    const instance = store();
    instance.load({
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [2, 2, 1] },
        { id: 'conv', kind: 'conv2d', filters: 8, kernelSize: 5, stride: 1, padding: 'same' },
        { id: 'out', kind: 'output', units: 2 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    });

    expect(instance.network.blocks[1]).toMatchObject({ kernelSize: 2 });
    expect(instance.announcements).toEqual([
      'Kernel size changed from 5 to 2 because the incoming data is 2×2.'
    ]);
  });

  it('leaves no convolution out of range after a reorder', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[0].id, { shape: [2, 2, 1] });
    instance.addBlock('conv2d', 1);
    instance.addBlock('conv2d', 2);
    instance.dismissAnnouncements();

    instance.moveBlock(2, 1);

    const kernels = instance.network.blocks
      .filter((block) => block.kind === 'conv2d')
      .map((block) => (block.kind === 'conv2d' ? block.kernelSize : 0));
    expect(kernels.every((size) => size <= 2)).toBe(true);
    expect(instance.errors.map((issue) => issue.title)).not.toContain(
      'Kernel is larger than the image'
    );
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: FAIL — the added convolution keeps `kernelSize: 3`, and loading an out-of-range network leaves `kernelSize: 5`.

- [ ] **Step 3: Write the implementation**

In `src/lib/editor/networkStore.svelte.ts`, add a private helper beside `#commit`:

```ts
  #commitClamped(next: Network): void {
    const result = clampNetwork(next);
    this.#commit(result.network);
    if (result.announcements.length > 0) {
      this.announcements = [...this.announcements, ...result.announcements];
    }
  }
```

Change `addBlock` to commit through it:

```ts
  addBlock(kind: BlockKind, index?: number, position?: NodePosition): string {
    const block = createBlock(kind);
    const target = index ?? insertionIndexFor(this.network, this.selectedBlockId);
    const inserted = insertAt(this.network, target, block);
    this.#commitClamped(
      position ? { ...inserted, positions: { ...inserted.positions, [block.id]: position } } : inserted
    );
    this.selectedBlockId = block.id;
    return block.id;
  }
```

Change `moveBlock` to commit through it:

```ts
  moveBlock(from: number, to: number): void {
    this.#commitClamped(moveBlock(this.network, from, to));
  }
```

Replace `load` so it clamps and reports:

```ts
  load(net: Network): void {
    const result = clampNetwork(net);
    this.#history.clear();
    this.network = result.network;
    this.selectedBlockId = null;
    this.announcements = [...result.announcements];
    this.#syncHistoryFlags();
  }
```

`updateBlock` is unchanged: it clamps the patch first, so `clampNetwork` cannot report the same block twice.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify nothing else regressed**

Run: `npm test`
Expected: the full suite passes. `clampNetwork` returns the same reference for a network with nothing to clamp, so the no-op paths still record no history — if any existing store test fails, that is the invariant to check first.

- [ ] **Step 6: Commit**

```bash
git add src/lib/editor/networkStore.svelte.ts src/lib/editor/networkStore.svelte.test.ts
git commit -m "fix: clamp the network on every mutation, not only block edits"
```

---

### Task 6: Weights addressed by example id

**Files:**
- Modify: `src/lib/persist/weights.ts`
- Modify: `src/lib/examples/mlp/runtime.ts`
- Test: `src/lib/persist/weights.test.ts` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces:

```ts
export function weightsUrl(id: string): string;
export function weightShapes(model: tf.LayersModel): number[][];
export function shapesMatch(a: number[][], b: number[][]): boolean;
export function saveWeights(model: tf.LayersModel, id: string): Promise<void>;
export function loadWeightsInto(model: tf.LayersModel, id: string): Promise<boolean>;
```

- `weightsUrl(id)` returns `indexeddb://visnet/weights/${id}`. The old `WEIGHTS_URL` constant is removed.
- `shapesMatch` and `weightShapes` are unchanged.

**Why:** with one constant URL the MLP and CNN examples would overwrite each other's saved weights.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/persist/weights.test.ts`:

```ts
import { weightsUrl } from './weights';

describe('weightsUrl', () => {
  it('namespaces by example id', () => {
    expect(weightsUrl('mlp')).toBe('indexeddb://visnet/weights/mlp');
    expect(weightsUrl('cnn')).toBe('indexeddb://visnet/weights/cnn');
    expect(weightsUrl('mlp')).not.toBe(weightsUrl('cnn'));
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/persist/weights.test.ts`
Expected: FAIL — `weightsUrl is not a function`.

- [ ] **Step 3: Write the implementation**

In `src/lib/persist/weights.ts`, replace the `WEIGHTS_URL` constant with:

```ts
export function weightsUrl(id: string): string {
  return `indexeddb://visnet/weights/${id}`;
}
```

Change the two IO functions to take the id:

```ts
export async function saveWeights(model: tf.LayersModel, id: string): Promise<void> {
  await model.save(weightsUrl(id));
}

export async function loadWeightsInto(model: tf.LayersModel, id: string): Promise<boolean> {
  let saved: tf.LayersModel | null = null;
  try {
    saved = await tf.loadLayersModel(weightsUrl(id));
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

In `src/lib/examples/mlp/runtime.ts`, pass the MLP's id:

```ts
    saveWeights: (model) => weights.saveWeights(model, 'mlp'),
    loadWeightsInto: (model) => weights.loadWeightsInto(model, 'mlp')
```

- [ ] **Step 4: Run the tests and the type check**

Run: `npx vitest run src/lib/persist/weights.test.ts`
Expected: PASS.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`. A remaining reference to `WEIGHTS_URL` is the likely failure; the only other consumer is `examples/mlp/runtime.ts`, updated above.

- [ ] **Step 5: Commit**

```bash
git add src/lib/persist/weights.ts src/lib/persist/weights.test.ts src/lib/examples/mlp/runtime.ts
git commit -m "fix: namespace saved weights per example"
```

---

### Task 7: Storage keys passed in rather than hardcoded

**Files:**
- Modify: `src/lib/persist/storage.ts`
- Modify: `src/lib/examples/mlp/example.ts`
- Modify: `src/routes/examples/mlp/+page.svelte`
- Test: `src/lib/persist/storage.test.ts` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces:

```ts
export interface StorageKeys {
  network: string;
  dataset: string;
}

export function createStorage(backing: KeyValueStore, keys: StorageKeys): NetworkStorage;
export function createBrowserStorage(keys: StorageKeys): NetworkStorage | null;
```

- The `NETWORK_KEY` and `DATASET_KEY` exports are removed.
- `src/lib/examples/mlp/example.ts` gains:

```ts
export const MLP_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:network:v1',
  dataset: 'visnet:mlp:dataset:v1'
};
```

**Why:** with one hardcoded key the CNN page would load and overwrite the MLP's saved network. **The MLP keeps its existing key strings deliberately**, so a network already saved in a browser still loads.

- [ ] **Step 1: Write the failing test**

In `src/lib/persist/storage.test.ts`, the existing tests reference `NETWORK_KEY` and `DATASET_KEY`. Replace those imports and the `createStorage(backing)` calls with a local key set:

```ts
const KEYS: StorageKeys = { network: 'test:network', dataset: 'test:dataset' };
```

and use `createStorage(backing, KEYS)`. Then append:

```ts
describe('key namespacing', () => {
  it('does not read another key set data', () => {
    const first = createStorage(backing, { network: 'a:network', dataset: 'a:dataset' });
    const second = createStorage(backing, { network: 'b:network', dataset: 'b:dataset' });

    first.saveNetwork(createEmptyNetwork());

    expect(first.loadNetwork()).not.toBeNull();
    expect(second.loadNetwork()).toBeNull();
    expect(second.hasStoredNetwork()).toBe(false);
  });

  it('clears only its own keys', () => {
    const first = createStorage(backing, { network: 'a:network', dataset: 'a:dataset' });
    const second = createStorage(backing, { network: 'b:network', dataset: 'b:dataset' });

    first.saveNetwork(createEmptyNetwork());
    second.saveNetwork(createEmptyNetwork());
    first.clear();

    expect(first.loadNetwork()).toBeNull();
    expect(second.loadNetwork()).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/lib/persist/storage.test.ts`
Expected: FAIL — `createStorage` expects one argument, and `NETWORK_KEY` no longer resolves once you remove the import.

- [ ] **Step 3: Write the implementation**

In `src/lib/persist/storage.ts`, remove the two key constants and add:

```ts
export interface StorageKeys {
  network: string;
  dataset: string;
}
```

Change the factories to take the keys and use them throughout:

```ts
export function createStorage(backing: KeyValueStore, keys: StorageKeys): NetworkStorage {
  return {
    saveNetwork(net) {
      backing.setItem(keys.network, toJSON(net));
    },
    loadNetwork() {
      const raw = backing.getItem(keys.network);
      return raw === null ? null : fromJSON(raw);
    },
    hasStoredNetwork() {
      return backing.getItem(keys.network) !== null;
    },
    saveDataset(dataset) {
      backing.setItem(keys.dataset, JSON.stringify({ points: dataset.points }));
    },
    loadDataset() {
      const raw = backing.getItem(keys.dataset);
      return raw === null ? null : parseDataset(raw);
    },
    clear() {
      backing.removeItem(keys.network);
      backing.removeItem(keys.dataset);
    }
  };
}

export function createBrowserStorage(keys: StorageKeys): NetworkStorage | null {
  try {
    const probe = '__visnet_probe__';
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return createStorage(window.localStorage, keys);
  } catch {
    return null;
  }
}
```

In `src/lib/examples/mlp/example.ts`, add the import and the constant:

```ts
import type { StorageKeys } from '../../persist/storage';

export const MLP_STORAGE_KEYS: StorageKeys = {
  network: 'visnet:network:v1',
  dataset: 'visnet:mlp:dataset:v1'
};
```

In `src/routes/examples/mlp/+page.svelte`, change the `createBrowserStorage()` call to `createBrowserStorage(MLP_STORAGE_KEYS)` and add `MLP_STORAGE_KEYS` to the existing import from `$lib/examples/mlp/example`.

- [ ] **Step 4: Run the tests and the type check**

Run: `npx vitest run src/lib/persist/storage.test.ts`
Expected: PASS.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 5: Commit**

```bash
git add src/lib/persist/storage.ts src/lib/persist/storage.test.ts src/lib/examples/mlp/example.ts src/routes/examples/mlp/+page.svelte
git commit -m "fix: take storage keys as a parameter so examples cannot share one save"
```

---

### Task 8: The store's initial network

**Files:**
- Modify: `src/lib/editor/networkStore.svelte.ts`
- Test: `src/lib/editor/networkStore.svelte.test.ts` (extend)

**Interfaces:**
- Consumes: `createEmptyNetwork` from `../network/factory`.
- Produces: `new NetworkStore(initial?: Network)` — the constructor's network becomes the network, and `reset()` returns to it. The default argument keeps `new NetworkStore()` behaving exactly as it does today.

**Why:** `reset()` currently calls `createEmptyNetwork()`, so the CNN page's "Reset network" would replace a convolutional network with the MLP's default.

- [ ] **Step 1: Write the failing test**

Append to `src/lib/editor/networkStore.svelte.test.ts`:

```ts
describe('initial network', () => {
  it('defaults to the MLP network', () => {
    const instance = new NetworkStore();
    expect(instance.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
  });

  it('resets to the network it was constructed with', () => {
    const initial: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [4, 4, 1] },
        { id: 'conv', kind: 'conv2d', filters: 2, kernelSize: 2, stride: 1, padding: 'same' },
        { id: 'out', kind: 'output', units: 3 }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const instance = new NetworkStore(initial);

    instance.addBlock('relu', 1);
    expect(instance.network.blocks).toHaveLength(4);

    instance.reset();

    expect(instance.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'conv2d',
      'output'
    ]);
    expect(instance.canUndo).toBe(false);
  });
});
```

Add `import type { Network } from '../network/types';` to the test file's imports if it is not already there.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: FAIL — `new NetworkStore(initial)` takes no arguments, and `reset()` returns the MLP network.

- [ ] **Step 3: Write the implementation**

In `src/lib/editor/networkStore.svelte.ts`, add a private field and a constructor:

```ts
export class NetworkStore {
  network: Network = $state(createEmptyNetwork());
  // ... existing fields unchanged ...

  #initial: Network;

  constructor(initial: Network = createEmptyNetwork()) {
    this.network = initial;
    this.#initial = initial;
  }
```

and change `reset`:

```ts
  reset(): void {
    this.load(this.#initial);
  }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run --project ui src/lib/editor/networkStore.svelte.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify the rest of the suite**

Run: `npm test`
Expected: the full suite passes. The MLP page constructs `new NetworkStore()` with no argument, so its behaviour is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/lib/editor/networkStore.svelte.ts src/lib/editor/networkStore.svelte.test.ts
git commit -m "feat: let the store be constructed with its example's network"
```

---

### Task 9: Document the prep step and verify the pipeline

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: everything built in Tasks 1-8.
- Produces: documentation that matches the repository, and a green full verification.

- [ ] **Step 1: Update `AGENTS.md`**

Add to the Commands list:

```markdown
- `npm run data:mnist` — download the MNIST subset into `static/mnist/` (gitignored)
```

Add a rule after the architecture rules:

```markdown
- `static/mnist/` is gitignored and holds a downloaded subset of MNIST, written by
  `npm run data:mnist`. Nothing in the repository or the build downloads it, and the
  CNN example degrades to an instruction when it is absent.
```

- [ ] **Step 2: Run the full verification**

Run: `npm run lint`
Expected: no errors.

Run: `npm run check`
Expected: `svelte-check found 0 errors and 0 warnings`.

Run: `npm test`
Expected: both projects green, 0 failures, no warnings. The engine project now also runs `scripts/prepare-mnist.test.ts`.

Run: `npm run build`
Expected: build succeeds and `build/index.html`, `build/examples/mlp.html`, and `build/examples/cnn.html` all exist.

Run: `git status --short`
Expected: clean apart from anything under `static/mnist/`, which must not appear because it is ignored.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: record the MNIST prep command and the ignored data directory"
```

---

## Plan Self-Review

**Spec coverage.** Spec §3 (prep script) → Task 2. §4 (asset format) → Tasks 1 and 2, with the round-trip test in Task 2 tying the writer to the reader. §5.1 (`clampNetwork` on every mutation) → Task 5. §5.2 (namespaced weights) → Task 6. §5.3 (the strict kernel bound) → no task, because the decision is to keep the existing behaviour; Task 9's docs step is not required to record it, as the spec already does. §6 (storage keys) → Task 7. §7 (image data modules) → Tasks 1, 3, and 4. §8 (the page) and §9 (the sample grid) → **plan C2**, not this plan. §10 (error handling) → the `ImageDataUnavailableError` path in Task 3; the page-level handling is C2. §11 (testing) → each task's tests; the `NetworkStore` initial-network test is Task 8. §12 (manual checklist) → the prep-script items are exercised in Task 2's Step 7; the page items are C2. §13 (docs) → Task 9 covers the command and the ignored directory; the CNN example's README change is C2. §14 (decisions) → realised across Tasks 1-8.

**Placeholder scan.** No "TBD", "TODO", "similar to Task N", or steps that describe work without showing it. Every code step carries complete code, and every command has an expected result.

**Type consistency.** `ImageDataset` is defined once in Task 1 and consumed by Tasks 3 and 4. `parseSplit` is used by Task 3 and by Task 2's round-trip test. `StorageKeys` is defined in Task 7 and consumed by Task 7's MLP keys. `weightsUrl` is defined in Task 6 and used only there. `clampNetwork` comes from the existing engine module and is used by Task 5. The `Network` shape used in Task 5's and Task 8's test fixtures matches the version 2 model from the previous phase, including `positions`.

