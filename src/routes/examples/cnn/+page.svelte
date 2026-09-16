<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import SampleGrid from '$lib/components/SampleGrid.svelte';
  import StatsReadout from '$lib/components/StatsReadout.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import type { ImageDataset } from '$lib/data/images';
  import { loadMnistData } from '$lib/data/mnist';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import {
    CNN_PALETTE,
    CNN_STORAGE_KEYS,
    CNN_WEIGHTS_ID,
    createCnnNetwork,
    defaultSampleIndices
  } from '$lib/examples/cnn/example';
  import {
    loadRuntime,
    type Model,
    type ModelData,
    type Runtime,
    type TrainerHandle
  } from '$lib/examples/runtime';
  import { createBrowserStorage, type NetworkStorage } from '$lib/persist/storage';
  import type { TrainStats } from '$lib/training/Trainer';

  const store = new NetworkStore(createCnnNetwork());
  store.expectedClasses = 10;
  store.expectedInputShape = [28, 28, 1];

  const sampleIndices = defaultSampleIndices();

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
  let testData = $state.raw<ImageDataset | null>(null);
  let sampleData = $state.raw<ModelData | null>(null);
  let playing = $state(false);
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let builtSignature = $state('');
  let trainData = $state.raw<ImageDataset | null>(null);
  let loadState = $state<'loading' | 'ready' | 'unavailable'>('loading');
  let redrawKey = $state(0);
  let trainCount = $state(0);

  let currentModel: Model | null = null;
  let data: ModelData | null = null;
  let currentSampleData: ModelData | null = null;
  let trainer: TrainerHandle | null = null;
  let storage: NetworkStorage | null = null;
  let compiledTraining = '';

  const architecture = $derived(JSON.stringify(store.network.blocks));
  const trainingSignature = $derived(JSON.stringify(store.network.training));

  function handleStats(next: TrainStats): void {
    stats = next;
    redrawKey += 1;
    if (next.epochMeanLoss !== null) {
      lossPoints = [...lossPoints, next.epochMeanLoss].slice(-200);
    }
  }

  function handleError(error: unknown): void {
    playing = false;
    console.error(error);
    banner = 'Training stopped because the model changed. Press Reset model and try again.';
  }

  function releaseTrainer(): void {
    trainer?.dispose();
    trainer = null;
    playing = false;
  }

  onMount(async () => {
    runtime = await loadRuntime(CNN_WEIGHTS_ID);
    storage = createBrowserStorage(CNN_STORAGE_KEYS);
    if (storage) {
      const saved = storage.loadNetwork();
      if (saved) store.load(saved);
      else if (storage.hasStoredNetwork()) {
        banner = 'The saved network could not be read, so a fresh one has been loaded.';
      }
    } else {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
    }

    try {
      const mnist = await loadMnistData();
      trainData = mnist.train;
      trainCount = mnist.train.count;
      testData = mnist.test;
      loadState = 'ready';
    } catch (error) {
      console.error(error);
      loadState = 'unavailable';
    }
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeData(currentSampleData);
    runtime?.disposeModel(currentModel);
    data = null;
    currentSampleData = null;
    currentModel = null;
    model = null;
  });

  $effect(() => {
    const api = runtime;
    if (!api) return;
    if (architecture === builtSignature) return;
    builtSignature = architecture;
    compiledTraining = trainingSignature;

    releaseTrainer();
    stats = null;
    lossPoints = [];
    api.disposeModel(currentModel);
    currentModel = null;
    model = null;

    if (!store.isValid) return;
    try {
      const built = api.buildModel(store.network);
      currentModel = built;
      model = built;
    } catch (error) {
      console.error(error);
      currentModel = null;
      model = null;
      banner = 'This network could not be built. Check the problems listed in the editor.';
    }
  });

  $effect(() => {
    const api = runtime;
    const training = trainingSignature;
    if (!api || !currentModel || training === compiledTraining) return;
    compiledTraining = training;
    api.compileModel(currentModel, store.network.training);
  });

  $effect(() => {
    const api = runtime;
    const train = trainData;
    const currentModelRef = model;
    if (!api || !train) return;

    const next = api.imagesToTensors(train);
    api.disposeData(data);
    data = next;
    releaseTrainer();

    if (!currentModelRef || !store.isValid || next.xs.shape[0] === 0) return;
    trainer = api.createTrainer(
      currentModelRef,
      next,
      store.network.training.batchSize,
      handleStats,
      handleError
    );
  });

  $effect(() => {
    const api = runtime;
    const test = testData;
    if (!api || !test) return;

    const next = api.imagesToTensors(test, sampleIndices);
    api.disposeData(currentSampleData);
    currentSampleData = next;
    sampleData = next;
  });

  $effect(() => {
    const net = store.network;
    const currentStorage = storage;
    if (!currentStorage) return;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(net);
      } catch {
        banner = 'Your work could not be saved. The browser storage may be full.';
      }
    }, 500);
    return () => clearTimeout(timer);
  });

  async function play(): Promise<void> {
    if (!trainer) return;
    playing = true;
    await trainer.play();
    playing = false;
  }

  function pause(): void {
    trainer?.pause();
    playing = false;
  }

  function step(): void {
    void trainer?.step();
  }

  function resetModel(): void {
    releaseTrainer();
    runtime?.disposeModel(currentModel);
    currentModel = null;
    model = null;
    stats = null;
    lossPoints = [];
    builtSignature = '';
  }

  async function save(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      await api.saveWeights(currentModel);
      banner = 'Model saved in this browser.';
    } catch {
      banner = 'The model could not be saved in this browser.';
    } finally {
      saving = false;
    }
  }

  async function load(): Promise<void> {
    const api = runtime;
    if (!api || !currentModel) return;
    saving = true;
    try {
      const loaded = await api.loadWeightsInto(currentModel);
      banner = loaded
        ? 'Saved weights loaded.'
        : 'No saved weights match this network. Train and save again.';
    } finally {
      saving = false;
    }
  }
</script>

<ExampleLayout
  title="Handwritten digits"
  intro="Build a small convolutional network, train it on handwritten digits, and watch it get them right."
>
  {#snippet editor()}
    <NetworkEditor {store} palette={CNN_PALETTE} onsave={save} onload={load} {saving} />
  {/snippet}

  {#snippet experiment()}
    {#if banner}
      <p class="banner" role="status">{banner}</p>
    {/if}

    {#if loadState === 'loading'}
      <p class="note">Loading the digit images…</p>
    {:else if loadState === 'unavailable'}
      <div class="note">
        <h2>The digit images are not prepared</h2>
        <p>
          Run <code>npm run data:mnist</code> in the project, then reload this page. The images
          are downloaded locally and are never part of the repository.
        </p>
      </div>
    {:else}
      <p class="note">Training on {trainCount} digits, checking against {testData?.count}.</p>
      <SampleGrid
        {model}
        dataset={testData}
        indices={sampleIndices}
        sampleXs={sampleData?.xs ?? null}
        {redrawKey}
        onerror={(message) => (banner = message)}
      />
    {/if}

    <TrainingPanel
      {store}
      {playing}
      disabled={!store.isValid || loadState !== 'ready'}
      onplay={play}
      onpause={pause}
      onstep={step}
      onreset={resetModel}
    />

    <LossChart points={lossPoints} />
    <StatsReadout {stats} />
  {/snippet}
</ExampleLayout>

<style>
  .banner {
    margin: 0;
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-warning) 12%, var(--color-surface));
    border: 1px solid var(--color-warning);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .note {
    display: grid;
    gap: var(--space-2);
    margin: 0;
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  .note h2 {
    margin: 0;
    font-size: var(--text-base);
    color: var(--color-text);
  }

  .note p {
    margin: 0;
  }
</style>
