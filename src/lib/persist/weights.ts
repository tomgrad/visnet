import * as tf from '@tensorflow/tfjs';

export const WEIGHTS_URL = 'indexeddb://visnet/weights/main';

export function weightShapes(model: tf.LayersModel): number[][] {
  return model.getWeights().map((tensor) => [...tensor.shape]);
}

export function shapesMatch(a: number[][], b: number[][]): boolean {
  if (a.length !== b.length) return false;
  return a.every((shape, index) => {
    const other = b[index];
    if (shape.length !== other.length) return false;
    return shape.every((dimension, position) => dimension === other[position]);
  });
}

export async function saveWeights(model: tf.LayersModel): Promise<void> {
  await model.save(WEIGHTS_URL);
}

export async function loadWeightsInto(model: tf.LayersModel): Promise<boolean> {
  let saved: tf.LayersModel | null = null;
  try {
    saved = await tf.loadLayersModel(WEIGHTS_URL);
    if (!shapesMatch(weightShapes(model), weightShapes(saved))) return false;
    model.setWeights(saved.getWeights());
    return true;
  } catch {
    return false;
  } finally {
    saved?.dispose();
  }
}
