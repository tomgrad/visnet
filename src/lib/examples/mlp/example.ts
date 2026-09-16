import { generate, type GeneratorName, type PointDataset } from '../../data/points';
import type { BlockKind } from '../../network/types';

export const MLP_PALETTE: BlockKind[] = ['linear', 'relu', 'sigmoid', 'softmax'];
export const DEFAULT_GENERATOR: GeneratorName = 'twoGaussians';
export const DEFAULT_POINT_COUNT = 200;
export const DEFAULT_SEED = 1;
export const CLASS_LABELS: [string, string] = ['blue', 'pink'];

export function defaultDataset(): PointDataset {
  return generate(DEFAULT_GENERATOR, DEFAULT_POINT_COUNT, DEFAULT_SEED);
}
