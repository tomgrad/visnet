import { mulberry32 } from './rng';

export interface Point {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface PointDataset {
  points: Point[];
  numClasses: 2;
}

export type GeneratorName = 'twoGaussians' | 'spirals' | 'xor' | 'circles';

export const GENERATOR_NAMES: GeneratorName[] = ['twoGaussians', 'spirals', 'xor', 'circles'];

export const GENERATOR_DESCRIPTIONS: Record<GeneratorName, string> = {
  twoGaussians: 'Two round clusters of points, one in each corner.',
  spirals: 'Two interlocking spirals. Needs a hidden layer to separate.',
  xor: 'Points in four quadrants, labelled in a checkerboard pattern.',
  circles: 'A small disc of one class surrounded by a ring of the other.'
};

function clamp(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function twoGaussians(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, (_, index) => {
    const label: 0 | 1 = index % 2 === 0 ? 0 : 1;
    const centre = label === 0 ? -0.5 : 0.5;
    return {
      x: clamp(centre + gaussian(rng) * 0.15),
      y: clamp(centre + gaussian(rng) * 0.15),
      label
    };
  });
}

function xor(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, () => {
    const x = rng() * 2 - 1;
    const y = rng() * 2 - 1;
    return { x, y, label: (x > 0 !== y > 0 ? 1 : 0) as 0 | 1 };
  });
}

function circles(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, () => {
    const label: 0 | 1 = rng() < 0.5 ? 0 : 1;
    const radius = label === 0 ? rng() * 0.3 : 0.6 + rng() * 0.3;
    const angle = rng() * Math.PI * 2;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, label };
  });
}

function spirals(count: number, rng: () => number): Point[] {
  return Array.from({ length: count }, (_, index) => {
    const label: 0 | 1 = index % 2 === 0 ? 0 : 1;
    const progress = Math.floor(index / 2) / Math.max(1, count / 2);
    const radius = progress * 0.9;
    const angle = progress * 3.2 * Math.PI + label * Math.PI + (rng() - 0.5) * 0.15;
    return { x: clamp(Math.cos(angle) * radius), y: clamp(Math.sin(angle) * radius), label };
  });
}

export function generate(name: GeneratorName, count: number, seed: number): PointDataset {
  const rng = mulberry32(seed);
  const generators: Record<GeneratorName, (count: number, rng: () => number) => Point[]> = {
    twoGaussians,
    spirals,
    xor,
    circles
  };
  return { points: generators[name](count, rng), numClasses: 2 };
}

export function addPoint(dataset: PointDataset, x: number, y: number, label: 0 | 1): PointDataset {
  return { points: [...dataset.points, { x, y, label }], numClasses: 2 };
}

export function clearPoints(dataset: PointDataset): PointDataset {
  return { points: [], numClasses: dataset.numClasses };
}
