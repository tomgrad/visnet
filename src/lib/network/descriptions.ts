import type { BlockKind } from './types';

export const BLOCK_DESCRIPTIONS: Record<BlockKind, string> = {
  input: 'Describes the shape of one example your network receives.',
  linear: 'Learns a weighted sum of its inputs. Also called a fully connected or dense layer.',
  conv2d: 'Slides small filters over an image to detect local patterns such as edges.',
  maxpool2d: 'Shrinks an image by keeping the largest value in each small window.',
  upsampling2d: 'Enlarges an image by repeating each pixel into a block.',
  conv2dtranspose: 'The reverse of a convolution: it learns to enlarge an image.',
  flatten: 'Turns image-shaped data into a flat list so Linear layers can read it.',
  reshape: 'Rearranges the numbers into a different shape without changing them.',
  relu: 'Keeps positive values and turns negative ones into zero. Helps the network learn curved patterns.',
  sigmoid: 'Squashes each value into the range 0 to 1.',
  tanh: 'Squashes each value into the range -1 to 1.',
  softmax: 'Turns raw scores into probabilities that add up to 1.',
  output: 'Declares the shape the network predicts.'
};

export const PARAM_DESCRIPTIONS = {
  inputShape: 'The shape of one example, for example 2 numbers or a 28 by 28 image.',
  outputShape: 'The shape the network should produce, for example 2 numbers or a 28 by 28 image.',
  reshapeShape: 'The new shape for the same numbers.',
  units: 'How many numbers this layer produces.',
  filters: 'How many different patterns this layer looks for.',
  kernelSize: 'How large the window sliding over the image is.',
  poolSize: 'How large the window is. The largest value in it survives.',
  size: 'How many times larger each side of the image becomes.',
  stride: 'How far the window moves each step. Larger means a smaller result.',
  padding: 'Whether the image keeps its size ("same") or shrinks ("valid").',
  loss: 'The number the network tries to make smaller while training.',
  optimizer: 'The rule used to update the weights after each batch.',
  learningRate: 'How big each learning step is. Smaller is slower but steadier.',
  batchSize: 'How many examples are used for one weight update.'
} as const;

export type ParamKey = keyof typeof PARAM_DESCRIPTIONS;

export function classifyInputShape(shape: number[]): 'flat' | 'image' | 'other' {
  if (shape.length === 1) return 'flat';
  if (shape.length === 3) return 'image';
  return 'other';
}
