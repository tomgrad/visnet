<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import DecisionBoundary from '$lib/components/DecisionBoundary.svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import StatsReadout from '$lib/components/StatsReadout.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import { GENERATOR_DESCRIPTIONS, GENERATOR_NAMES } from '$lib/data/points';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import { DatasetStore } from '$lib/examples/mlp/datasetStore.svelte';
  import {
    CLASS_LABELS,
    MLP_PALETTE,
    MLP_STORAGE_KEYS,
    MLP_WEIGHTS_ID
  } from '$lib/examples/mlp/example';
  import { weightsDiscardedNotice } from '$lib/examples/notices';
  import {
    loadRuntime,
    type Model,
    type ModelData,
    type Runtime,
    type TrainerHandle
  } from '$lib/examples/runtime';
  import { createBrowserStorage, type NetworkStorage } from '$lib/persist/storage';
  import type { TrainStats } from '$lib/training/Trainer';

  const store = new NetworkStore();
  const datasetStore = new DatasetStore();
  store.expectedClasses = 2;

  let runtime = $state.raw<Runtime | null>(null);
  let model = $state.raw<Model | null>(null);
  let playing = $state(false);
  let redrawKey = $state(0);
  let stats = $state<TrainStats | null>(null);
  let lossPoints = $state<number[]>([]);
  let banner = $state<string | null>(null);
  let saving = $state(false);
  let builtSignature = $state('');
  let trained = $state(false);

  let currentModel: Model | null = null;
  let data: ModelData | null = null;
  let trainer: TrainerHandle | null = null;
  let storage: NetworkStorage | null = null;
  let compiledTraining = '';

  const architecture = $derived(JSON.stringify(store.network.blocks));
  const trainingSignature = $derived(JSON.stringify(store.network.training));

  function handleStats(next: TrainStats): void {
    redrawKey += 1;
    stats = next;
    if (next.epochMeanLoss !== null) {
      trained = true;
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
    runtime = await loadRuntime(MLP_WEIGHTS_ID);
    storage = createBrowserStorage(MLP_STORAGE_KEYS);
    if (!storage) {
      banner =
        'This browser will not let the app save your work, so changes last only until you reload.';
      return;
    }
    const savedNetwork = storage.loadNetwork();
    if (savedNetwork) {
      store.load(savedNetwork);
    } else if (storage.hasStoredNetwork()) {
      banner = 'The saved network could not be read, so a fresh one has been loaded.';
    }
    const savedDataset = storage.loadDataset();
    if (savedDataset) datasetStore.dataset = savedDataset;
  });

  onDestroy(() => {
    releaseTrainer();
    runtime?.disposeData(data);
    runtime?.disposeModel(currentModel);
    data = null;
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
    const hadTrained = trained;
    trained = false;
    api.disposeModel(currentModel);
    currentModel = null;
    model = null;
    if (hadTrained) banner = weightsDiscardedNotice(true);

    if (!store.isValid) return;
    try {
      const built = api.buildModel(store.network);
      currentModel = built;
      model = built;
    } catch (error) {
      console.error(error);
      api.disposeModel(currentModel);
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
    const dataset = datasetStore.dataset;
    const currentModelRef = model;
    if (!api) return;

    const next = api.toTensors(dataset);
    releaseTrainer();
    api.disposeData(data);
    data = next;

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
    const net = store.network;
    const dataset = datasetStore.dataset;
    const currentStorage = storage;
    if (!currentStorage) return;
    const timer = setTimeout(() => {
      try {
        currentStorage.saveNetwork(net);
        currentStorage.saveDataset(dataset);
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
    trained = false;
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
  title="Points in 2D"
  intro="Build a small network, train it on coloured points, and watch the boundary between the two classes take shape."
>
  {#snippet editor()}
    <NetworkEditor {store} palette={MLP_PALETTE} onsave={save} onload={load} {saving} />
  {/snippet}

  {#snippet experiment()}
    {#if banner}
      <p class="banner" role="status">{banner}</p>
    {/if}

    <div class="dataset">
      <h2>Data</h2>
      <label>
        <span>Pattern</span>
        <select
          data-testid="dataset-generator"
          value={datasetStore.generator}
          onchange={(event) => {
            datasetStore.generator = event.currentTarget.value as (typeof GENERATOR_NAMES)[number];
            datasetStore.regenerate();
          }}
        >
          {#each GENERATOR_NAMES as name (name)}
            <option value={name}>{name}</option>
          {/each}
        </select>
        <small>{GENERATOR_DESCRIPTIONS[datasetStore.generator]}</small>
      </label>

      <label>
        <span>Points</span>
        <input
          type="number"
          min="10"
          max="2000"
          data-testid="dataset-count"
          value={datasetStore.pointCount}
          onchange={(event) => {
            const value = Number(event.currentTarget.value);
            if (Number.isFinite(value) && value >= 10) {
              datasetStore.pointCount = value;
              datasetStore.regenerate();
            }
          }}
        />
      </label>

      <div class="dataset-actions">
        <button type="button" onclick={() => datasetStore.reseed()}>New sample</button>
        <button type="button" onclick={() => datasetStore.clear()}>Clear points</button>
      </div>

      <fieldset>
        <legend>Point to add when clicking</legend>
        {#each [0, 1] as label (label)}
          <label class="inline">
            <input
              type="radio"
              name="label"
              value={label}
              checked={datasetStore.selectedLabel === label}
              onchange={() => datasetStore.selectLabel(label as 0 | 1)}
            />
            <span>{CLASS_LABELS[label]}</span>
          </label>
        {/each}
      </fieldset>
    </div>

    <DecisionBoundary
      {model}
      {redrawKey}
      dataset={datasetStore.dataset}
      selectedLabel={datasetStore.selectedLabel}
      onaddpoint={(x, y) => datasetStore.addPoint(x, y)}
      caption={!runtime
        ? 'Loading the network. The boundary appears in a moment.'
        : store.isValid
          ? 'Each coloured area is the class the network predicts at that spot. Click to add a point.'
          : 'Fix the problems listed in the editor before the boundary can be drawn.'}
    />

    <TrainingPanel
      {store}
      {playing}
      disabled={!store.isValid}
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

  .dataset {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  label.inline {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  select,
  input[type='number'] {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  fieldset {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2);
    display: flex;
    gap: var(--space-3);
    margin: 0;
  }

  legend {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }

  .dataset-actions {
    display: flex;
    gap: var(--space-2);
  }

  .dataset-actions button {
    flex: 1;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }
</style>
