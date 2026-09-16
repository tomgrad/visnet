# AGENTS.md

Static web app for teaching neural networks; see `README.md` for feature scope and supported layers/losses/optimizers.

## Status

Engine complete and unit-tested: pure network domain modules (`src/lib/network/`),
the TensorFlow.js model builder (`src/lib/tf/`), the trainer
(`src/lib/training/`), the 2D dataset (`src/lib/data/`), the
decision-boundary renderer (`src/lib/render/`), and the editor's
framework-free canvas projection and undo/redo logic (`src/lib/editor/`). The
Svelte editor UI and the MLP example page are not built yet.

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
  `src/lib/training/**`, `src/lib/data/tensors.ts`, `src/lib/render/boundary.ts`
  (the only render module that runs a forward pass), and the test files colocated
  with those modules. No other production module may import it.
- No block stores its input dimension; inputs derive from the previous block's
  output.
- Every validation issue carries a non-empty `title`, `message`, and `fix`.
- Do not add code comments unless a non-obvious constraint requires one.

Design reference: `docs/superpowers/specs/2026-09-16-visnet-design.md`.
