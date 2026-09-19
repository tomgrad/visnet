import type { BlockKind } from './types';

export interface BlockCategory {
  id: string;
  label: string;
  kinds: BlockKind[];
}

export const BLOCK_CATEGORIES: BlockCategory[] = [
  { id: 'dense', label: 'Dense', kinds: ['linear', 'flatten', 'reshape'] },
  {
    id: 'image',
    label: 'Image',
    kinds: ['conv2d', 'maxpool2d', 'upsampling2d', 'conv2dtranspose']
  },
  { id: 'activation', label: 'Activations', kinds: ['relu', 'sigmoid', 'tanh', 'softmax'] }
];

export const PALETTE_KINDS: BlockKind[] = BLOCK_CATEGORIES.flatMap((category) => category.kinds);
