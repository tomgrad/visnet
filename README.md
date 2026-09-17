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
- max pooling
- upsampling
- flatten
- reshape
- ReLU
- Sigmoid
- Tanh
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
- saving model weights in the browser

## What works today

- Build a network from blocks on a canvas: add, delete, reorder, and connect them.
- See the tensor shape at every point in the pipeline, on each block and on each wire,
  with a table of the whole network and its parameter counts.
- Train on generated 2D points or points you click onto the canvas, and watch the
  decision boundary change as the loss falls.
- See the latent space of the block selected on the canvas: the input grid warped
  through that layer, with a button to cycle which pair of dimensions is shown.
- Undo and redo every edit.
- Your network and dataset are restored when you reload, and you can load your saved weights
  on demand.
- Train a small convolutional network on handwritten digits and watch a grid of test
  digits turn from wrong to right.
- See the feature maps of the block selected on the canvas for a test digit, with
  a button to step through the sample digits.

## Digit data

The handwritten-digit example needs its data prepared once, locally:

```sh
npm run data:mnist
```

That downloads a subset of MNIST into `static/mnist/`, which is gitignored — no dataset
is committed to this repository and nothing downloads it at build or install time. Until
you run it, the example page explains what to do rather than failing. MNIST is a
derivative of the NIST Special Database 19.

## Architecture

The engine is framework-independent. `src/lib/network/` defines networks, infers
the tensor shape flowing through every block, and reports validation problems as
plain-language errors with a suggested fix. Persistence is a shallow codec:
`src/lib/persist/networkCodec.ts` turns a network into plain data and back, and
`src/lib/persist/weights.ts` saves and loads trained weights. TensorFlow.js is
confined to `src/lib/tf/`, `src/lib/training/`, `src/lib/data/tensors.ts`,
`src/lib/render/boundary.ts`, `src/lib/render/latent.ts`,
`src/lib/render/activations.ts`, `src/lib/render/features.ts`, and
`src/lib/persist/weights.ts`.

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
