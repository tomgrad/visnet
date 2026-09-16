# AGENTS.md

Static web app for teaching neural networks; see `README.md` for feature scope and supported layers/losses/optimizers.

## Status

Greenfield. Only `README.md` exists — no `package.json`, toolchain, CI, or git repo. Scaffold the stack before adding features; do not assume build/test commands exist yet.

## Stack

- Svelte + Vite + TypeScript
- npm (do not use pnpm/yarn/bun)
- TensorFlow.js for model definition/training
- Static output only, no backend
- No test tooling yet
