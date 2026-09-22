import { onDestroy, onMount, untrack } from 'svelte';
import type { NetworkStore } from '../editor/networkStore.svelte';
import type { NetworkStorage } from '../persist/storage';
import type { TrainStats } from '../training/Trainer';
import { createStatsTracker, WEIGHTS_DISCARDED_NOTICE } from './experimentStats';
import {
  loadRuntime,
  type Model,
  type ModelData,
  type Runtime,
  type TrainerHandle
} from './runtime';

export interface Experiment {
  runtime: Runtime | null;
  model: Model | null;
  data: ModelData | null;
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
}

export function createExperiment(options: {
  store: NetworkStore;
  weightsId: string;
  storage: NetworkStorage | null;
}): Experiment {
  const { store, storage } = options;

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
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

  let trainer: TrainerHandle | null = null;
  let compiledTraining = '';

  const architecture = $derived(
    JSON.stringify(store.network.blocks.map((block) => ({ ...block, colour: undefined })))
  );
  const trainingSignature = $derived(JSON.stringify(store.network.training));
  const trainingBatchSize = $derived(store.network.training.batchSize);

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
    const currentModel = model;
    const currentData = data;
    const batchSize = trainingBatchSize;
    releaseTrainer();
    if (!api || !currentModel || !currentData || !store.isValid || currentData.xs.shape[0] === 0) {
      return;
    }
    trainer = api.createTrainer(
      currentModel,
      currentData,
      batchSize,
      handleStats,
      handleError,
      store.task !== 'reconstruction'
    );
  });

  $effect(() => {
    const api = runtime;
    if (!api || architecture === builtSignature) return;
    builtSignature = architecture;
    compiledTraining = trainingSignature;
    releaseTrainer();
    resetStats();
    const hadTrained = trained;
    trained = false;
    api.disposeModel(model);
    model = null;
    if (hadTrained) banner = WEIGHTS_DISCARDED_NOTICE;
    if (!store.isValid) return;
    try {
      model = api.buildModel(store.network);
    } catch (error) {
      console.error(error);
      model = null;
      void import('../tf/buildModel')
        .then(({ describeBuildError }) => {
          banner = describeBuildError(error).message;
        })
        .catch(() => {
          banner = 'The network could not be built.';
        });
    }
  });

  $effect(() => {
    const api = runtime;
    const training = trainingSignature;
    if (!api || !model || training === compiledTraining) return;
    compiledTraining = training;
    api.compileModel(model, store.network.training);
  });

  $effect(() => {
    const currentStorage = storage;
    if (!currentStorage) return;
    const net = store.network;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(net);
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
        store.load(saved);
      } else if (storage.hasStoredNetwork()) {
        banner = 'The saved network could not be read, so a fresh one has been loaded.';
      }
    } else {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
    }
    runtime = await loadRuntime(options.weightsId);
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeModel(model);
  });

  return {
    get runtime() {
      return runtime;
    },
    get model() {
      return model;
    },
    get data() {
      return data;
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
    announce(message: string) {
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
      runtime?.disposeModel(model);
      model = null;
      resetStats();
      trained = false;
      builtSignature = '';
    },
    async save() {
      const api = runtime;
      if (!api || !model) return;
      saving = true;
      try {
        await api.saveWeights(model);
        banner = 'Model saved in this browser.';
      } catch {
        banner = 'The model could not be saved in this browser.';
      } finally {
        saving = false;
      }
    },
    async load() {
      const api = runtime;
      if (!api || !model) return;
      saving = true;
      try {
        const loaded = await api.loadWeightsInto(model);
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
      runtime?.disposeModel(model);
    }
  };
}
