import { classifyInputShape } from './descriptions';
import { inferShapes } from './inferShapes';
import { BLOCK_KINDS, type Network } from './types';

export type Severity = 'error' | 'warning';

export interface Problem {
  severity: Severity;
  title: string;
  message: string;
  fix: string;
  blockId?: string;
}

export interface ProblemOptions {
  expectedClasses?: number;
  expectedInputShape?: number[];
}

function error(problem: Omit<Problem, 'severity'>): Problem {
  return { severity: 'error', ...problem };
}

function warning(problem: Omit<Problem, 'severity'>): Problem {
  return { severity: 'warning', ...problem };
}

function shapeText(shape: number[] | null): string {
  return shape ? `[${shape.join(', ')}]` : 'an unknown shape';
}

export function findProblems(net: Network, options: ProblemOptions = {}): Problem[] {
  const problems: Problem[] = [];
  const { perBlock } = inferShapes(net);

  const inputs = net.blocks.filter((block) => block.kind === 'input');
  const outputs = net.blocks.filter((block) => block.kind === 'output');

  if (inputs.length === 0) {
    problems.push(
      error({
        title: 'Missing Input block',
        message:
          'A network needs exactly one Input block to describe the shape of the data it receives.',
        fix: 'Add an Input block to the start of the network.'
      })
    );
  }

  if (outputs.length === 0) {
    problems.push(
      error({
        title: 'Missing Output block',
        message: 'A network needs exactly one Output block to say what it predicts.',
        fix: 'Add an Output block to the end of the network.'
      })
    );
  }

  if (inputs.length > 1) {
    problems.push(
      error({
        title: 'More than one Input block',
        message: `There are ${inputs.length} Input blocks, but a network can only have one.`,
        fix: 'Delete the extra Input blocks.'
      })
    );
  }

  if (outputs.length > 1) {
    problems.push(
      error({
        title: 'More than one Output block',
        message: `There are ${outputs.length} Output blocks, but a network can only have one.`,
        fix: 'Delete the extra Output blocks.'
      })
    );
  }

  const learnableCount = net.blocks.filter(
    (block) => block.kind !== 'input' && block.kind !== 'output'
  ).length;

  if (learnableCount === 0) {
    problems.push(
      error({
        title: 'Nothing to learn',
        message:
          'The Input connects straight to the Output, so there are no layers for the network to learn with.',
        fix: 'Add at least one layer, such as a Linear layer, between Input and Output.'
      })
    );
  }

  net.blocks.forEach((block, index) => {
    const info = perBlock[index];

    if (!BLOCK_KINDS.includes(block.kind)) {
      problems.push(
        error({
          title: 'Unrecognised block',
          message: `This network contains a block type this version of VisNet does not understand (${block.kind}).`,
          fix: 'Delete the block, or reset the network to start fresh.',
          blockId: block.id
        })
      );
      return;
    }

    if (block.kind === 'linear' && info.inShape && info.inShape.length !== 1) {
      problems.push(
        error({
          title: 'Linear layer needs a flat list',
          message: `This Linear layer receives ${shapeText(info.inShape)}, which is not a flat list of numbers. Linear layers need a flat list.`,
          fix: 'Add a Flatten layer before this Linear layer.',
          blockId: block.id
        })
      );
    }

    if (block.kind === 'conv2d' && info.inShape && info.inShape.length !== 3) {
      problems.push(
        error({
          title: 'Convolution layer needs image data',
          message: `This Convolution layer receives ${shapeText(info.inShape)}. It expects image data shaped [height, width, channels].`,
          fix: 'Give the Input block a 3D shape such as [28, 28, 1], or remove the Convolution layer.',
          blockId: block.id
        })
      );
    }

    if (
      block.kind === 'conv2d' &&
      info.inShape &&
      info.inShape.length === 3 &&
      info.outShape === null
    ) {
      const [height, width] = info.inShape;
      problems.push(
        error({
          title: 'Kernel is larger than the image',
          message: `A ${block.kernelSize}×${block.kernelSize} kernel with stride ${block.stride} leaves no room to slide over a ${height}×${width} image.`,
          fix: "Use a smaller kernel or stride, or set padding to 'same'.",
          blockId: block.id
        })
      );
    }

    if (block.kind === 'maxpool2d' && info.inShape && info.inShape.length !== 3) {
      problems.push(
        error({
          title: 'Pooling layer needs image data',
          message: `This Pooling layer receives ${shapeText(info.inShape)}. It expects image data shaped [height, width, channels].`,
          fix: 'Give the Input block a 3D shape such as [28, 28, 1], or remove the Pooling layer.',
          blockId: block.id
        })
      );
    }

    if (
      block.kind === 'maxpool2d' &&
      info.inShape &&
      info.inShape.length === 3 &&
      info.outShape === null
    ) {
      const [height, width] = info.inShape;
      problems.push(
        error({
          title: 'Pool window is larger than the image',
          message: `A ${block.poolSize}×${block.poolSize} pool with stride ${block.stride} leaves no room to slide over a ${height}×${width} image.`,
          fix: "Use a smaller pool or stride, or set padding to 'same'.",
          blockId: block.id
        })
      );
    }

    if (block.kind === 'upsampling2d' && info.inShape && info.inShape.length !== 3) {
      problems.push(
        error({
          title: 'Upsampling layer needs image data',
          message: `This Upsampling layer receives ${shapeText(info.inShape)}. It expects image data shaped [height, width, channels].`,
          fix: 'Give the Input block a 3D shape such as [28, 28, 1], or remove the Upsampling layer.',
          blockId: block.id
        })
      );
    }

    if (block.kind === 'flatten' && info.inShape && info.inShape.length === 1) {
      problems.push(
        error({
          title: 'Nothing to flatten',
          message: `This Flatten layer receives ${shapeText(info.inShape)}, which is already a flat list.`,
          fix: 'Remove this Flatten layer, or move it after a Convolution layer.',
          blockId: block.id
        })
      );
    }
  });

  let lastRealIndex = -1;
  net.blocks.forEach((block, index) => {
    if (block.kind !== 'output') lastRealIndex = index;
  });

  const outputBlock = net.blocks.find((block) => block.kind === 'output');

  if (outputBlock && outputBlock.kind === 'output' && lastRealIndex >= 0) {
    const lastShape = perBlock[lastRealIndex].outShape;
    if (lastShape && lastShape.length !== 1) {
      problems.push(
        error({
          title: 'Output must be a list of scores',
          message: `The last layer produces ${shapeText(lastShape)}, which is not a list of numbers, so it cannot be compared with the ${outputBlock.units} scores the Output block expects.`,
          fix: 'End the network with a Linear layer so the output is a list of numbers.',
          blockId: net.blocks[lastRealIndex].id
        })
      );
    } else if (lastShape && lastShape[0] !== outputBlock.units) {
      problems.push(
        error({
          title: 'Last layer size does not match the Output block',
          message: `The last layer produces ${lastShape[0]} numbers, but the Output block says ${outputBlock.units}.`,
          fix: `Set the last layer to ${outputBlock.units} units, or change the Output block to ${lastShape[0]}.`,
          blockId: net.blocks[lastRealIndex].id
        })
      );
    }
  }

  const softmaxIndex = net.blocks.findIndex((block) => block.kind === 'softmax');
  const hasSoftmax = softmaxIndex !== -1;
  const hasConvolution = net.blocks.some((block) => block.kind === 'conv2d');

  if (net.training.loss === 'crossEntropy' && !hasSoftmax) {
    problems.push(
      warning({
        title: 'Add a Softmax for probabilities',
        message:
          "Cross-entropy works best when the network's outputs are probabilities, but the network currently ends with raw scores. Training will still run, but it may be less stable.",
        fix: 'Add a Softmax block after the last Linear layer.'
      })
    );
  }

  if (net.training.loss === 'mse' && hasSoftmax) {
    problems.push(
      warning({
        title: 'Softmax is unusual with mean squared error',
        message: 'Mean squared error is normally used with raw scores, not probabilities.',
        fix: 'Switch the loss to cross-entropy, or remove the Softmax block.'
      })
    );
  }

  if (hasSoftmax) {
    if (softmaxIndex !== lastRealIndex) {
      problems.push(
        warning({
          title: 'Softmax is not the last layer',
          message:
            'This Softmax block is followed by more layers, so the probabilities it produces get transformed again.',
          fix: 'Move the Softmax block to just before the Output block.',
          blockId: net.blocks[softmaxIndex].id
        })
      );
    }
  }

  if (
    options.expectedClasses !== undefined &&
    outputBlock &&
    outputBlock.kind === 'output' &&
    outputBlock.units !== options.expectedClasses
  ) {
    problems.push(
      warning({
        title: 'Output size does not match the data',
        message: `The Output block says ${outputBlock.units} classes, but the dataset has ${options.expectedClasses}.`,
        fix: `Make the last layer produce ${options.expectedClasses} numbers, and set the Output block to ${options.expectedClasses} units.`,
        blockId: outputBlock.id
      })
    );
  }

  const inputBlock = net.blocks.find((block) => block.kind === 'input');
  if (inputBlock && inputBlock.kind === 'input') {
    const inputKind = classifyInputShape(inputBlock.shape);
    if (inputKind === 'image' && !hasConvolution) {
      problems.push(
        warning({
          title: 'Image input without a Convolution layer',
          message: `The Input block is image-shaped ${shapeText(inputBlock.shape)}, but the network has no Convolution layer to look at it.`,
          fix: 'Add a Convolution layer, or change the Input shape to a flat list.',
          blockId: inputBlock.id
        })
      );
    }
    if (inputKind === 'flat' && hasConvolution) {
      problems.push(
        warning({
          title: 'Convolution layer without image input',
          message: `The network has a Convolution layer, but the Input block is a flat list ${shapeText(inputBlock.shape)}.`,
          fix: 'Set the Input shape to 3D such as [28, 28, 1], or remove the Convolution layer.',
          blockId: inputBlock.id
        })
      );
    }
  }

  if (inputBlock && inputBlock.kind === 'input' && options.expectedInputShape) {
    const expected = options.expectedInputShape;
    const actual = inputBlock.shape;
    const matches =
      actual.length === expected.length && actual.every((size, i) => size === expected[i]);

    if (!matches) {
      problems.push(
        warning({
          title: 'Input shape does not match the data',
          message: `The Input block is ${shapeText(actual)}, but the images are ${shapeText(expected)}.`,
          fix: `Set the Input block to ${shapeText(expected)}.`,
          blockId: inputBlock.id
        })
      );
    }
  }

  return problems;
}
