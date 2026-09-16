import type { PointDataset } from '../../data/points';
import type { Network, TrainingConfig } from '../../network/types';
import type { TrainStats } from '../../training/Trainer';
import type * as tf from '@tensorflow/tfjs';

export type Model = tf.LayersModel;
export type ModelData = {
  xs: tf.Tensor2D;
  ys: tf.Tensor2D;
};
export type TrainerHandle = import('../../training/Trainer').Trainer;

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
}

export async function loadRuntime(): Promise<Runtime> {
  const [builder, tensors, trainerModule, weights] = await Promise.all([
    import('../../tf/buildModel'),
    import('../../data/tensors'),
    import('../../training/Trainer'),
    import('../../persist/weights')
  ]);

  return {
    buildModel: (net) => builder.buildModel(net),
    compileModel: (model, training) => builder.compileModel(model, training),
    toTensors: (dataset) => tensors.toTensors(dataset),
    disposeData: (data) => {
      data?.xs.dispose();
      data?.ys.dispose();
    },
    disposeModel: (model) => {
      model?.dispose();
    },
    createTrainer: (model, data, batchSize, onStats, onError) =>
      new trainerModule.Trainer(model, data, batchSize, onStats, undefined, onError),
    saveWeights: (model) => weights.saveWeights(model),
    loadWeightsInto: (model) => weights.loadWeightsInto(model)
  };
}
