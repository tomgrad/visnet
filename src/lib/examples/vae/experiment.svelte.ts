import { onDestroy, onMount, untrack } from 'svelte';
import type * as tf from '@tensorflow/tfjs';
import type { TrainStats } from '../../training/Trainer';
import type { VaeNetwork } from '../../network/vae';
import { syncTraining } from '../../network/vae';
import { createStatsTracker, WEIGHTS_DISCARDED_NOTICE } from '../experimentStats';
import {
  loadRuntime,
  type ModelData,
  type Runtime,
  type VaeModels,
  type VaeTrainerHandle
} from '../runtime';
import type { VaeController } from './vaeController.svelte';

export interface VaeStorage {
  saveNetwork(vae: VaeNetwork): void;
  loadNetwork(): VaeNetwork | null;
  hasStoredNetwork(): boolean;
}

export interface VaeExperiment {
  runtime: Runtime | null;
  encoder: VaeModels['encoder'] | null;
  decoder: VaeModels['decoder'] | null;
  data: ModelData | null;
  latentSize: number;
  playing: boolean;
  stats: TrainStats | null;
  lossPoints: number[];
  banner: string | null;
  saving: boolean;
  redrawKey: number;
  setData(next: ModelData | null): void;
  announce(message: string): void;
  play(): Promise<void>;
  pause(): void;
  step(): void;
  resetModel(): void;
  save(): Promise<void>;
  load(): Promise<void>;
  dispose(): void;
  reconstruct(xs: tf.Tensor): tf.Tensor;
  generate(z: tf.Tensor): tf.Tensor;
}

export function createVaeExperiment(options: {
  controller: VaeController;
  weightsId: string;
  storage: VaeStorage | null;
}): VaeExperiment {
  const { controller, storage } = options;

  let runtime = $state.raw<Runtime | null>(null);
  let encoder = $state.raw<VaeModels['encoder'] | null>(null);
  let decoder = $state.raw<VaeModels['decoder'] | null>(null);
  let data = $state.raw<ModelData | null>(null);
  let playing = $state(false);
  const tracker = createStatsTracker();
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let redrawKey = $state(0);
  let builtSignature = $state('');
  let trained = $state(false);

  let trainer: VaeTrainerHandle | null = null;

  const architecture = $derived(
    JSON.stringify({
      encoder: controller.encoderStore.network.blocks.map((block) => ({
        ...block,
        colour: undefined
      })),
      decoder: controller.decoderStore.network.blocks.map((block) => ({
        ...block,
        colour: undefined
      })),
      latentSize: controller.latentSize
    })
  );
  const isValid = $derived(controller.isValid);
  const latentSize = $derived(controller.latentSize);
  const batchSize = $derived(controller.training.batchSize);
  const learningRate = $derived(controller.training.learningRate);

  function vaeNetwork(): VaeNetwork {
    return {
      encoder: controller.encoderStore.network,
      decoder: controller.decoderStore.network,
      latentSize: controller.latentSize,
      training: controller.training
    };
  }

  function syncStats(): void {
    stats = tracker.stats;
    lossPoints = tracker.lossPoints;
    redrawKey = tracker.redrawKey;
  }

  function resetStats(): void {
    tracker.reset();
    syncStats();
  }

  function handleStats(next: TrainStats): void {
    if (next.epochMeanLoss !== null) trained = true;
    tracker.track(next);
    syncStats();
  }

  function handleError(error: unknown): void {
    playing = false;
    console.error(error);
    banner = 'Training stopped. Reset the model and try again.';
  }

  function releaseTrainer(): void {
    trainer?.dispose();
    trainer = null;
    playing = false;
  }

  function setData(next: ModelData | null): void {
    releaseTrainer();
    const previous = untrack(() => data);
    runtime?.disposeData(previous);
    data = next;
  }

  $effect(() => {
    const api = runtime;
    const currentEncoder = encoder;
    const currentDecoder = decoder;
    const currentData = data;
    const currentLatentSize = latentSize;
    const currentBatchSize = batchSize;
    const currentLearningRate = learningRate;
    releaseTrainer();
    if (
      !api ||
      !currentEncoder ||
      !currentDecoder ||
      !currentData ||
      !isValid ||
      currentData.xs.shape[0] === 0
    ) {
      return;
    }
    trainer = api.createVaeTrainer(
      { encoder: currentEncoder, decoder: currentDecoder },
      currentData,
      {
        ...untrack(() => controller.training),
        batchSize: currentBatchSize,
        learningRate: currentLearningRate
      },
      currentLatentSize,
      handleStats,
      handleError
    );
  });

  $effect(() => {
    const api = runtime;
    if (!api || architecture === builtSignature) return;
    builtSignature = architecture;
    releaseTrainer();
    resetStats();
    const hadTrained = trained;
    trained = false;
    const previousEncoder = untrack(() => encoder);
    const previousDecoder = untrack(() => decoder);
    api.disposeModel(previousEncoder);
    api.disposeModel(previousDecoder);
    encoder = null;
    decoder = null;
    if (hadTrained) banner = WEIGHTS_DISCARDED_NOTICE;
    if (!isValid) return;
    try {
      const models = api.buildVaeModels(syncTraining(vaeNetwork()));
      encoder = models.encoder;
      decoder = models.decoder;
    } catch (error) {
      console.error(error);
      encoder = null;
      decoder = null;
      void import('../../tf/buildModel')
        .then(({ describeBuildError }) => {
          banner = describeBuildError(error).message;
        })
        .catch(() => {
          banner = 'The network could not be built.';
        });
    }
  });

  $effect(() => {
    const currentStorage = storage;
    if (!currentStorage) return;
    const vae = vaeNetwork();
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(vae);
      } catch {
        banner = 'Your work could not be saved. The browser storage may be full.';
      }
    }, 500);
    return () => clearTimeout(timer);
  });

  onMount(async () => {
    if (storage) {
      const saved = storage.loadNetwork();
      if (saved) {
        controller.encoderStore.load(saved.encoder);
        controller.decoderStore.load(saved.decoder);
      } else if (storage.hasStoredNetwork()) {
        banner = 'The saved network could not be read, so a fresh one has been loaded.';
      }
    } else {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
    }
    runtime = await loadRuntime(options.weightsId, { vae: true });
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeModel(encoder);
    runtime?.disposeModel(decoder);
  });

  return {
    get runtime() {
      return runtime;
    },
    get encoder() {
      return encoder;
    },
    get decoder() {
      return decoder;
    },
    get data() {
      return data;
    },
    get latentSize() {
      return controller.latentSize;
    },
    get playing() {
      return playing;
    },
    get stats() {
      return stats;
    },
    get lossPoints() {
      return lossPoints;
    },
    get banner() {
      return banner;
    },
    set banner(value: string | null) {
      banner = value;
    },
    get saving() {
      return saving;
    },
    get redrawKey() {
      return redrawKey;
    },
    setData,
    announce(message) {
      banner = message;
    },
    async play() {
      if (!trainer) return;
      banner = null;
      playing = true;
      await trainer.play();
      playing = false;
    },
    pause() {
      trainer?.pause();
      playing = false;
    },
    step() {
      if (!trainer) return;
      banner = null;
      void trainer.step();
    },
    resetModel() {
      releaseTrainer();
      banner = null;
      runtime?.disposeModel(encoder);
      runtime?.disposeModel(decoder);
      encoder = null;
      decoder = null;
      resetStats();
      trained = false;
      builtSignature = '';
    },
    async save() {
      const api = runtime;
      const currentEncoder = encoder;
      const currentDecoder = decoder;
      if (!api || !currentEncoder || !currentDecoder) return;
      saving = true;
      try {
        await api.saveVaeWeights({ encoder: currentEncoder, decoder: currentDecoder });
        banner = 'Model saved in this browser.';
      } catch {
        banner = 'The model could not be saved in this browser.';
      } finally {
        saving = false;
      }
    },
    async load() {
      const api = runtime;
      const currentEncoder = encoder;
      const currentDecoder = decoder;
      if (!api || !currentEncoder || !currentDecoder) return;
      saving = true;
      try {
        const loaded = await api.loadVaeWeightsInto({
          encoder: currentEncoder,
          decoder: currentDecoder
        });
        banner = loaded
          ? 'Saved weights loaded.'
          : 'No saved weights match this network. Train and save again.';
      } finally {
        saving = false;
      }
    },
    dispose() {
      releaseTrainer();
      runtime?.disposeData(data);
      runtime?.disposeModel(encoder);
      runtime?.disposeModel(decoder);
    },
    reconstruct(xs) {
      const api = runtime;
      const currentEncoder = encoder;
      const currentDecoder = decoder;
      if (!api || !currentEncoder || !currentDecoder) {
        throw new Error('The VAE models are not ready yet.');
      }
      return api.reconstruct(
        { encoder: currentEncoder, decoder: currentDecoder },
        xs,
        controller.latentSize
      );
    },
    generate(z) {
      const api = runtime;
      const currentEncoder = encoder;
      const currentDecoder = decoder;
      if (!api || !currentEncoder || !currentDecoder) {
        throw new Error('The VAE models are not ready yet.');
      }
      return api.generate({ encoder: currentEncoder, decoder: currentDecoder }, z);
    }
  };
}
