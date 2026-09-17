import * as tf from '@tensorflow/tfjs';

export const LATENT_GRID_SIZE = 32;

export interface LatentBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface LatentSample {
  grid: Float32Array;
  gridClasses: Int32Array;
  points: Float32Array;
  bounds: LatentBounds;
}

export function gridInputs(size: number = LATENT_GRID_SIZE): Float32Array {
  const cells = new Float32Array(size * size * 2);
  for (let gy = 0; gy < size; gy++) {
    for (let gx = 0; gx < size; gx++) {
      const offset = (gy * size + gx) * 2;
      cells[offset] = ((gx + 0.5) / size) * 2 - 1;
      cells[offset + 1] = 1 - ((gy + 0.5) / size) * 2;
    }
  }
  return cells;
}

function boundsOf(arrays: Float32Array[]): LatentBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const array of arrays) {
    for (let index = 0; index < array.length; index += 2) {
      const x = array[index];
      const y = array[index + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

export function projectLatent(
  model: tf.LayersModel,
  cells: Float32Array,
  points: Float32Array,
  source: 'input' | number,
  dimA: number
): LatentSample {
  const cellCount = cells.length / 2;
  const pointCount = points.length / 2;
  const all = new Float32Array(cells.length + points.length);
  all.set(cells, 0);
  all.set(points, cells.length);

  const result = tf.tidy(() => {
    const input = tf.tensor2d(all, [cellCount + pointCount, 2]);
    const activations: tf.Tensor[] = [];
    let current: tf.Tensor = input;
    for (const layer of model.layers) {
      current = layer.apply(current) as tf.Tensor;
      activations.push(current);
    }

    const final = activations[activations.length - 1];
    const gridClasses = Int32Array.from(tf.argMax(final, 1).dataSync().slice(0, cellCount));

    let coords: Float32Array;
    if (source === 'input') {
      coords = all.slice();
    } else {
      const activation = activations[source];
      const a = activation.slice([0, dimA], [cellCount + pointCount, 1]);
      const b = activation.slice([0, dimA + 1], [cellCount + pointCount, 1]);
      coords = Float32Array.from(tf.concat([a, b], 1).dataSync());
    }

    return {
      grid: coords.slice(0, cellCount * 2),
      gridClasses,
      points: coords.slice(cellCount * 2)
    };
  });

  return { ...result, bounds: boundsOf([result.grid, result.points]) };
}
