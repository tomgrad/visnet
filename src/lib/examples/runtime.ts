import type { ImageDataset } from '../data/images';
import type { PointDataset } from '../data/points';
import type { Network, TrainingConfig } from '../network/types';
import type { VaeNetwork } from '../network/vae';
import type { TrainStats } from '../training/Trainer';
import type { VaeModels } from '../tf/vae';
import type * as tf from '@tensorflow/tfjs';

export type Model = tf.LayersModel;
export type ModelData = {
  xs: tf.Tensor;
  ys: tf.Tensor;
};
export type TrainerHandle = import('../training/Trainer').Trainer;
export type VaeTrainerHandle = import('../training/VaeTrainer').VaeTrainer;
export type { VaeModels };

export interface Runtime {
  buildModel(net: Network): Model;
  compileModel(model: Model, training: TrainingConfig): void;
  toTensors(dataset: PointDataset): ModelData;
  imagesToTensors(dataset: ImageDataset, indices?: number[]): ModelData;
  imagesToReconstruction(dataset: ImageDataset, indices?: number[]): ModelData;
  disposeData(data: ModelData | null): void;
  disposeModel(model: Model | null): void;
  createTrainer(
    model: Model,
    data: ModelData,
    batchSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void,
    computeAccuracy?: boolean
  ): TrainerHandle;
  saveWeights(model: Model): Promise<void>;
  loadWeightsInto(model: Model): Promise<boolean>;
  buildVaeModels(vae: VaeNetwork): VaeModels;
  createVaeTrainer(
    models: VaeModels,
    data: ModelData,
    training: TrainingConfig,
    latentSize: number,
    onStats: (stats: TrainStats) => void,
    onError: (error: unknown) => void
  ): VaeTrainerHandle;
  saveVaeWeights(models: VaeModels): Promise<void>;
  loadVaeWeightsInto(models: VaeModels): Promise<boolean>;
  reconstruct(models: VaeModels, xs: tf.Tensor, latentSize: number): tf.Tensor;
  generate(models: VaeModels, z: tf.Tensor): tf.Tensor;
}

export async function loadRuntime(weightsId: string): Promise<Runtime> {
  const [builder, tensors, trainerModule, weights, vaeModule, vaeTrainerModule] = await Promise.all(
    [
      import('../tf/buildModel'),
      import('../data/tensors'),
      import('../training/Trainer'),
      import('../persist/weights'),
      import('../tf/vae'),
      import('../training/VaeTrainer')
    ]
  );

  return {
    buildModel: (net) => builder.buildModel(net),
    compileModel: (model, training) => builder.compileModel(model, training),
    toTensors: (dataset) => tensors.toTensors(dataset),
    imagesToTensors: (dataset, indices) => tensors.imagesToTensors(dataset, indices),
    imagesToReconstruction: (dataset, indices) => tensors.imagesToReconstruction(dataset, indices),
    disposeData: (data) => {
      data?.xs.dispose();
      data?.ys.dispose();
    },
    disposeModel: (model) => {
      model?.dispose();
    },
    createTrainer: (model, data, batchSize, onStats, onError, computeAccuracy) =>
      new trainerModule.Trainer(
        model,
        data,
        batchSize,
        onStats,
        undefined,
        onError,
        computeAccuracy
      ),
    saveWeights: (model) => weights.saveWeights(model, weightsId),
    loadWeightsInto: (model) => weights.loadWeightsInto(model, weightsId),
    buildVaeModels: (vae) => vaeModule.buildVaeModels(vae),
    createVaeTrainer: (models, data, training, latentSize, onStats, onError) =>
      new vaeTrainerModule.VaeTrainer(
        models,
        data,
        training,
        latentSize,
        onStats,
        undefined,
        onError
      ),
    saveVaeWeights: (models) => weights.saveVaeWeights(models, weightsId),
    loadVaeWeightsInto: (models) => weights.loadVaeWeightsInto(models, weightsId),
    reconstruct: (models, xs, latentSize) => vaeModule.reconstruct(models, xs, latentSize),
    generate: (models, z) => vaeModule.generate(models, z)
  };
}
