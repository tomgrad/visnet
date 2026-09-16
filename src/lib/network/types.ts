export type BlockKind =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'flatten'
  | 'relu'
  | 'sigmoid'
  | 'softmax'
  | 'output';

export const BLOCK_KINDS: readonly BlockKind[] = [
  'input',
  'linear',
  'conv2d',
  'flatten',
  'relu',
  'sigmoid',
  'softmax',
  'output'
];

export interface BlockBase {
  id: string;
}

export interface InputBlock extends BlockBase {
  kind: 'input';
  shape: number[];
}

export interface LinearBlock extends BlockBase {
  kind: 'linear';
  units: number;
}

export interface Conv2dBlock extends BlockBase {
  kind: 'conv2d';
  filters: number;
  kernelSize: number;
  stride: number;
  padding: 'same' | 'valid';
}

export interface FlattenBlock extends BlockBase {
  kind: 'flatten';
}

export interface ActivationBlock extends BlockBase {
  kind: 'relu' | 'sigmoid' | 'softmax';
}

export interface OutputBlock extends BlockBase {
  kind: 'output';
  units: number;
}

export type Block =
  | InputBlock
  | LinearBlock
  | Conv2dBlock
  | FlattenBlock
  | ActivationBlock
  | OutputBlock;

export interface TrainingConfig {
  loss: 'mse' | 'crossEntropy';
  optimizer: 'sgd' | 'adam';
  learningRate: number;
  batchSize: number;
}

export interface Network {
  version: 1;
  blocks: Block[];
  training: TrainingConfig;
}
