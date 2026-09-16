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
