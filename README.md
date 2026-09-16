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

## Architecture

The engine is framework-independent. `src/lib/network/` defines networks, infers
the tensor shape flowing through every block, and reports validation problems as
plain-language errors with a suggested fix. TensorFlow.js is confined to
`src/lib/tf/`, `src/lib/training/`, `src/lib/data/tensors.ts`, and
`src/lib/render/boundary.ts`.

The editor UI and example pages are built on top of this engine and are not part
of the engine itself.

## Development

```
npm run dev      # development server
npm run build    # static build
npm run check    # type checking
npm run lint     # linting
npm test         # unit tests
```
