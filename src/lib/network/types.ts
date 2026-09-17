export type BlockKind =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'maxpool2d'
  | 'upsampling2d'
  | 'flatten'
  | 'reshape'
  | 'relu'
  | 'sigmoid'
  | 'tanh'
  | 'softmax'
  | 'output';

export const BLOCK_KINDS: readonly BlockKind[] = [
  'input',
  'linear',
  'conv2d',
  'maxpool2d',
  'upsampling2d',
  'flatten',
  'reshape',
  'relu',
  'sigmoid',
  'tanh',
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

export interface MaxPool2dBlock extends BlockBase {
  kind: 'maxpool2d';
  poolSize: number;
  stride: number;
  padding: 'same' | 'valid';
}

export interface UpSampling2dBlock extends BlockBase {
  kind: 'upsampling2d';
  size: number;
}

export interface FlattenBlock extends BlockBase {
  kind: 'flatten';
}

export interface ReshapeBlock extends BlockBase {
  kind: 'reshape';
  shape: number[];
}

export interface ActivationBlock extends BlockBase {
  kind: 'relu' | 'sigmoid' | 'tanh' | 'softmax';
}

export interface OutputBlock extends BlockBase {
  kind: 'output';
  shape: number[];
}

export type Block =
  | InputBlock
  | LinearBlock
  | Conv2dBlock
  | MaxPool2dBlock
  | UpSampling2dBlock
  | FlattenBlock
  | ReshapeBlock
  | ActivationBlock
  | OutputBlock;

export interface TrainingConfig {
  loss: 'mse' | 'crossEntropy';
  optimizer: 'sgd' | 'adam';
  learningRate: number;
  batchSize: number;
}

export interface NodePosition {
  x: number;
  y: number;
}

export interface Network {
  blocks: Block[];
  training: TrainingConfig;
  positions: Record<string, NodePosition>;
}
