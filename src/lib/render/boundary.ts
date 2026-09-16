import * as tf from '@tensorflow/tfjs';

export const GRID_SIZE = 64;

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function clientToDomain(px: number, py: number, rect: Rect): { x: number; y: number } {
  return {
    x: ((px - rect.left) / rect.width) * 2 - 1,
    y: 1 - ((py - rect.top) / rect.height) * 2
  };
}

export function cellCentre(gx: number, gy: number, size: number = GRID_SIZE): { x: number; y: number } {
  return {
    x: ((gx + 0.5) / size) * 2 - 1,
    y: 1 - ((gy + 0.5) / size) * 2
  };
}

export function classesToRgba(
  classes: Int32Array,
  colours: Array<[number, number, number]>,
  size: number = GRID_SIZE
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let index = 0; index < size * size; index++) {
    const colour = colours[classes[index]] ?? [0, 0, 0];
    data[index * 4] = colour[0];
    data[index * 4 + 1] = colour[1];
    data[index * 4 + 2] = colour[2];
    data[index * 4 + 3] = 255;
  }
  return data;
}

export function sampleGrid(model: tf.LayersModel, size: number = GRID_SIZE): Int32Array {
  const cells = new Float32Array(size * size * 2);
  for (let gy = 0; gy < size; gy++) {
    for (let gx = 0; gx < size; gx++) {
      const centre = cellCentre(gx, gy, size);
      const offset = (gy * size + gx) * 2;
      cells[offset] = centre.x;
      cells[offset + 1] = centre.y;
    }
  }

  return tf.tidy(() => {
    const input = tf.tensor2d(cells, [size * size, 2]);
    const logits = model.predict(input) as tf.Tensor;
    return Int32Array.from(tf.argMax(logits, 1).dataSync());
  });
}
