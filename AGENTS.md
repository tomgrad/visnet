# AGENTS.md

Static web app for teaching neural networks; see `README.md` for feature scope and supported layers/losses/optimizers.

## Status

Engine complete and unit-tested, and the editor UI and the 2D points example are
built on top of it. A user can drag blocks onto a canvas, edit them, train, and
watch a live decision boundary, and their network and dataset are restored on
reload. The convolutional (MNIST) example page is a placeholder.

Shipped modules: `src/lib/network/` (pure domain), `src/lib/tf/` (model builder),
`src/lib/training/`, `src/lib/data/`, `src/lib/render/`, `src/lib/persist/`,
`src/lib/editor/`, `src/lib/components/` (editor UI),
`src/lib/examples/` (the MLP example), and design tokens in
`src/lib/styles/tokens.css`.

The `ui` Vitest project runs component tests in jsdom; the `engine` project runs
the rest in Node. See `npm test`.

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
- `npm run data:mnist` — download the MNIST subset into `static/mnist/` (gitignored)
- `npm run preview` — preview the production build locally
- `npm run check` — svelte-check and TypeScript
- `npm run lint` — ESLint
- `npm test` — Vitest (run once)
- `npm run test:watch` — Vitest in watch mode
- `npm run format` — Prettier

## Architecture rules

- `src/lib/network/**` is pure: no Svelte, no TensorFlow.js, no DOM. It must run in
  plain Node and is the single source of truth for network validity and shapes.
- `@tensorflow/tfjs` may only be imported by `src/lib/tf/**`,
  `src/lib/training/**`, `src/lib/data/tensors.ts`,
  `src/lib/render/boundary.ts` (the only render module that runs a forward pass),
  `src/lib/persist/weights.ts`, and the test files colocated with those modules.
  No other production module may import it.
- TensorFlow.js and `@xyflow/svelte` must only be reached from the browser. Import
  them dynamically (or inside an `onMount`/`$effect`) so `npm run build` can
  prerender every route with SSR on.
- `src/lib/editor/flow.ts` and `src/lib/editor/history.ts` stay free of Svelte,
  TensorFlow.js, DOM, and `@xyflow/svelte`.
- No block stores its input dimension; inputs derive from the previous block's
  output.
- Node positions are stored on the network (`positions`, keyed by block id) and are
  purely cosmetic: the `blocks` array defines the chain order, so moving a node
  never changes the network.
- Every validation issue carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.
- `static/mnist/` is gitignored and holds a downloaded subset of MNIST, written by
  `npm run data:mnist`. No build or install step downloads it automatically, and the
  CNN example degrades to an instruction when it is absent.

Design reference: `docs/superpowers/specs/2026-09-16-visnet-design.md`.
