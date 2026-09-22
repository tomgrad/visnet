import * as tf from '@tensorflow/tfjs';
import type { TrainingConfig } from '../network/types';
import { klDivergence, sampleLatent, splitLatent, type VaeModels } from '../tf/vae';
import type { TrainStats, YieldFn } from './Trainer';

const nextFrame: YieldFn = () => tf.nextFrame();

export class VaeTrainer {
  private readonly batchesPerEpochCount: number;
  private readonly batchSize: number;
  private readonly optimizer: tf.Optimizer;
  private readonly variables: tf.Variable[];
  private epoch = 0;
  private batch = 0;
  private batchLosses: number[] = [];
  private pool: number[] = [];
  private poolPosition = 0;
  private playing = false;
  private disposed = false;

  constructor(
    private readonly models: VaeModels,
    private readonly data: { xs: tf.Tensor },
    training: TrainingConfig,
    private readonly latentSize: number,
    private readonly onStats: (stats: TrainStats) => void,
    private readonly yieldFn: YieldFn = nextFrame,
    private readonly onError: (error: unknown) => void = () => {}
  ) {
    const examples = data.xs.shape[0];
    this.batchSize = training.batchSize;
    this.batchesPerEpochCount = Math.max(1, Math.ceil(examples / this.batchSize));
    this.optimizer = tf.train.adam(training.learningRate);
    this.variables = [...models.encoder.trainableWeights, ...models.decoder.trainableWeights].map(
      (weight) => (weight as unknown as { val: tf.Variable }).val
    );
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get batchesPerEpoch(): number {
    return this.batchesPerEpochCount;
  }

  play(): Promise<void> {
    if (this.playing || this.disposed) return Promise.resolve();
    this.playing = true;
    return this.loop();
  }

  pause(): void {
    this.playing = false;
  }

  async step(): Promise<void> {
    if (this.disposed) return;
    try {
      const batchXs = tf.tidy(() => {
        const indices = tf.tensor1d(this.sampleIndices(), 'int32');
        return tf.gather(this.data.xs, indices);
      });

      let batchLoss: number;
      try {
        const cost = this.optimizer.minimize(() => this.loss(batchXs), true, this.variables);
        batchLoss = cost ? (cost.dataSync()[0] as number) : 0;
        cost?.dispose();
        if (this.disposed) return;
      } finally {
        batchXs.dispose();
      }

      this.batchLosses.push(batchLoss);
      this.batch += 1;

      if (this.batch >= this.batchesPerEpochCount) {
        const mean =
          this.batchLosses.reduce((total, value) => total + value, 0) / this.batchLosses.length;
        this.epoch += 1;
        this.batch = 0;
        this.batchLosses = [];
        this.onStats({
          epoch: this.epoch,
          batch: 0,
          batchLoss,
          epochMeanLoss: mean,
          epochAccuracy: null
        });
        return;
      }

      this.onStats({
        epoch: this.epoch,
        batch: this.batch,
        batchLoss,
        epochMeanLoss: null,
        epochAccuracy: null
      });
    } catch (error) {
      this.playing = false;
      this.onError(error);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.playing = false;
    this.optimizer.dispose();
  }

  private loss(xs: tf.Tensor): tf.Scalar {
    const encoded = this.models.encoder.apply(xs, { training: true }) as tf.Tensor;
    const [mean, logVariance] = splitLatent(encoded, this.latentSize);
    const z = sampleLatent(mean, logVariance);
    const reconstruction = this.models.decoder.apply(z, { training: true }) as tf.Tensor;
    return tf.add(tf.losses.meanSquaredError(xs, reconstruction), klDivergence(mean, logVariance));
  }

  private async loop(): Promise<void> {
    try {
      while (this.playing && !this.disposed) {
        await this.step();
        if (!this.playing || this.disposed) break;
        await this.yieldFn();
        if (this.disposed) break;
      }
    } catch (error) {
      this.onError(error);
    } finally {
      this.playing = false;
    }
  }

  private sampleIndices(): number[] {
    const examples = this.data.xs.shape[0];
    const remaining = examples - this.batch * this.batchSize;
    const size = Math.min(this.batchSize, remaining);
    if (this.batch === 0) {
      this.pool = Array.from({ length: examples }, (_, index) => index);
      for (let i = this.pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [this.pool[i], this.pool[j]] = [this.pool[j], this.pool[i]];
      }
      this.poolPosition = 0;
    }
    const indices = this.pool.slice(this.poolPosition, this.poolPosition + size);
    this.poolPosition += indices.length;
    return indices;
  }
}
