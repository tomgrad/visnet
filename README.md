# VisNet

Web app for teaching neural networks.

Inspired by https://cs.stanford.edu/people/karpathy/convnetjs but with visual blocks and drag-and-drop mechanics.

Allows users to build simple neural networks using visual blocks, that represent the network's layers and connections.
Layers can be added, removed, and connected to each other to create a network.
Drag and drop mechanics.
Modular. Network editor can be embedded in subpages with examples.

Examples:
- classification with MLP (points in 2D space)
- classification with CNN (MNIST images)

Available modules:
- linear
- convolutional
- flatten
- ReLU
- Sigmoid
- Softmax

Available loss functions:
- mean squared error
- cross entropy

Available optimizers:
- stochastic gradient descent
- Adam

Specifics:
- no backend, hosted as a static site
- uses TensorFlow.js
- saving models to local storage

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

```
npm run dev         # development server
npm run build       # static build
npm run check       # type checking
npm run lint        # linting
npm test            # unit and component tests
```
