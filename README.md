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
- ReLU
- Sigmoid

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
