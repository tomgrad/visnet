import {
  addPoint,
  clearPoints,
  generate,
  type GeneratorName,
  type PointDataset
} from '../../data/points';
import { DEFAULT_GENERATOR, DEFAULT_POINT_COUNT, DEFAULT_SEED, defaultDataset } from './example';

export class DatasetStore {
  dataset = $state<PointDataset>(defaultDataset());
  generator = $state<GeneratorName>(DEFAULT_GENERATOR);
  pointCount = $state<number>(DEFAULT_POINT_COUNT);
  seed = $state<number>(DEFAULT_SEED);
  selectedLabel = $state<0 | 1>(0);

  regenerate(): void {
    this.dataset = generate(this.generator, this.pointCount, this.seed);
  }

  reseed(): void {
    this.seed += 1;
    this.regenerate();
  }

  addPoint(x: number, y: number): void {
    const clamp = (value: number): number => Math.max(-1, Math.min(1, value));
    this.dataset = addPoint(this.dataset, clamp(x), clamp(y), this.selectedLabel);
  }

  clear(): void {
    this.dataset = clearPoints(this.dataset);
  }

  selectLabel(label: 0 | 1): void {
    this.selectedLabel = label;
  }
}
