import type { BlockKind } from './types';

export interface BlockCategory {
  id: string;
  label: string;
  kinds: readonly BlockKind[];
}

export const BLOCK_CATEGORIES: readonly BlockCategory[] = [
  { id: 'dense', label: 'Dense', kinds: ['linear', 'flatten', 'reshape'] },
  {
    id: 'image',
    label: 'Image',
    kinds: ['conv2d', 'conv2dtranspose', 'maxpool2d', 'upsampling2d']
  },
  { id: 'activation', label: 'Activations', kinds: ['relu', 'sigmoid', 'tanh', 'softmax'] }
];

export const PALETTE_KINDS: readonly BlockKind[] = BLOCK_CATEGORIES.flatMap(
  (category) => category.kinds
);
