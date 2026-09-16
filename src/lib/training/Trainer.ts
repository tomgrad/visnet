import * as tf from '@tensorflow/tfjs';

export interface TrainStats {
  epoch: number;
  batch: number;
  batchLoss: number;
  epochMeanLoss: number | null;
  epochAccuracy: number | null;
}

export type YieldFn = () => Promise<void>;

const nextFrame: YieldFn = () => tf.nextFrame();

export class Trainer {
  private readonly batchesPerEpochCount: number;
  private epoch = 0;
  private batch = 0;
  private batchLosses: number[] = [];
  private pool: number[] = [];
  private poolPosition = 0;
  private playing = false;
  private disposed = false;

  constructor(
    private readonly model: tf.LayersModel,
    private readonly data: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    private readonly batchSize: number,
    private readonly onStats: (stats: TrainStats) => void,
    private readonly yieldFn: YieldFn = nextFrame
  ) {
    const examples = data.xs.shape[0];
    this.batchesPerEpochCount = Math.max(1, Math.ceil(examples / batchSize));
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

    const indices = tf.tensor1d(this.sampleIndices(), 'int32');
    const batchXs = tf.gather(this.data.xs, indices);
    const batchYs = tf.gather(this.data.ys, indices);
    indices.dispose();

    let batchLoss: number;
    try {
      const result = await this.model.trainOnBatch(batchXs, batchYs);
      batchLoss = Array.isArray(result) ? result[0] : result;
    } finally {
      batchXs.dispose();
      batchYs.dispose();
    }

    this.batchLosses.push(batchLoss);
    this.batch += 1;

    if (this.batch >= this.batchesPerEpochCount) {
      const mean = this.batchLosses.reduce((total, value) => total + value, 0) / this.batchLosses.length;
      const accuracy = this.accuracy();
      this.epoch += 1;
      this.batch = 0;
      this.batchLosses = [];
      this.onStats({
        epoch: this.epoch,
        batch: 0,
        batchLoss,
        epochMeanLoss: mean,
        epochAccuracy: accuracy
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
  }

  dispose(): void {
    this.disposed = true;
    this.playing = false;
  }

  private async loop(): Promise<void> {
    while (this.playing && !this.disposed) {
      await this.step();
      if (!this.playing || this.disposed) break;
      await this.yieldFn();
    }
    this.playing = false;
  }

  private sampleIndices(): number[] {
    const examples = this.data.xs.shape[0];
    const size = Math.min(this.batchSize, examples);
    const indices: number[] = [];

    while (indices.length < size) {
      if (this.poolPosition >= this.pool.length) {
        this.pool = Array.from({ length: examples }, (_, index) => index);
        for (let i = this.pool.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [this.pool[i], this.pool[j]] = [this.pool[j], this.pool[i]];
        }
        this.poolPosition = 0;
      }
      indices.push(this.pool[this.poolPosition]);
      this.poolPosition += 1;
    }

    return indices;
  }

  private accuracy(): number {
    return tf.tidy(() => {
      const logits = this.model.predict(this.data.xs) as tf.Tensor;
      const predictions = tf.argMax(logits, 1);
      const labels = tf.argMax(this.data.ys, 1);
      const correct = tf.cast(tf.equal(predictions, labels), 'float32').mean();
      return correct.dataSync()[0];
    });
  }
}
