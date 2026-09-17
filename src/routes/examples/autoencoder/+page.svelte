<script lang="ts">
  import { onMount } from 'svelte';
  import CodeScatter from '$lib/components/CodeScatter.svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import FeatureMaps from '$lib/components/FeatureMaps.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import ReconstructionGrid from '$lib/components/ReconstructionGrid.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import type { ImageDataset } from '$lib/data/images';
  import { loadMnistData } from '$lib/data/mnist';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import {
    AUTOENCODER_PALETTE,
    AUTOENCODER_STORAGE_KEYS,
    AUTOENCODER_WEIGHTS_ID,
    PRESETS,
    RECONSTRUCTION_COUNT,
    SCATTER_COUNT,
    TRAIN_COUNT,
    presetFor
  } from '$lib/examples/autoencoder/example';
  import { createExperiment } from '$lib/examples/experiment.svelte';
  import { probeTargetFor } from '$lib/network/probe';
  import { createBrowserStorage } from '$lib/persist/storage';

  function firstIndices(count: number, size: number): number[] {
    return Array.from({ length: Math.max(0, Math.min(count, size)) }, (_, index) => index);
  }

  const store = new NetworkStore(PRESETS[0].create());
  store.task = 'reconstruction';
  store.expectedInputShape = [28, 28, 1];
  const storage = createBrowserStorage(AUTOENCODER_STORAGE_KEYS);
  const session = createExperiment({ store, weightsId: AUTOENCODER_WEIGHTS_ID, storage });

  let testData = $state.raw<ImageDataset | null>(null);
  let trainData = $state.raw<ImageDataset | null>(null);
  let loadState = $state<'loading' | 'ready' | 'unavailable'>('loading');
  let trainCount = $state(0);

  const selectedPreset = $derived(presetFor(store.network));
  const target = $derived(probeTargetFor(store.network, store.selectedBlockId));
  const rank = $derived(target ? target.dims.length : 0);
  const reconstructionIndices = $derived(
    testData ? firstIndices(testData.count, RECONSTRUCTION_COUNT) : []
  );
  const scatterIndices = $derived(testData ? firstIndices(testData.count, SCATTER_COUNT) : []);

  onMount(async () => {
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

  $effect(() => {
    const api = session.runtime;
    const train = trainData;
    if (!api || !train) return;
    session.setData(api.imagesToReconstruction(train, firstIndices(train.count, TRAIN_COUNT)));
  });

  function choosePreset(event: Event): void {
    const id = (event.currentTarget as HTMLSelectElement).value;
    const preset = PRESETS.find((candidate) => candidate.id === id);
    if (preset) store.load(preset.create());
  }
</script>

<ExampleLayout
  title="Autoencoder"
  intro="Build a network that compresses a handwritten digit and rebuilds it, then watch the reconstructions sharpen as it trains."
>
  {#snippet editor()}
    <NetworkEditor
      {store}
      palette={AUTOENCODER_PALETTE}
      onsave={session.save}
      onload={session.load}
      saving={session.saving}
    />
  {/snippet}

  {#snippet experiment()}
    {#if session.banner}
      <p class="banner" role="status">{session.banner}</p>
    {/if}

    {#if loadState === 'loading'}
      <p class="note">Loading the digit images…</p>
    {:else if loadState === 'unavailable'}
      <div class="note">
        <h2>The digit images are not prepared</h2>
        <p>
          Run <code>npm run data:mnist</code> in the project, then reload this page. The images are downloaded
          locally and are never part of the repository.
        </p>
      </div>
    {:else}
      <p class="note">Training on the first {Math.min(trainCount, TRAIN_COUNT)} digits.</p>

      <label class="preset">
        <span>Network</span>
        <select
          data-testid="autoencoder-preset"
          value={selectedPreset ?? 'custom'}
          onchange={choosePreset}
        >
          {#each PRESETS as preset (preset.id)}
            <option value={preset.id}>{preset.label}</option>
          {/each}
          {#if selectedPreset === null}
            <option value="custom">Custom</option>
          {/if}
        </select>
      </label>

      <div class="scroll">
        <ReconstructionGrid
          model={session.model}
          dataset={testData}
          indices={reconstructionIndices}
          redrawKey={session.redrawKey}
        />
      </div>

      {#if !session.model}
        <p class="note" data-testid="layer-message">
          Fix the problems listed in the editor before the layer view can be drawn.
        </p>
      {:else if target === null}
        <p class="note" data-testid="layer-message">
          Select a block to see its code scatter or feature maps.
        </p>
      {:else if rank === 1}
        <CodeScatter
          model={session.model}
          {store}
          dataset={testData}
          indices={scatterIndices}
          redrawKey={session.redrawKey}
        />
      {:else if rank === 3}
        <FeatureMaps
          model={session.model}
          {store}
          dataset={testData}
          indices={scatterIndices}
          redrawKey={session.redrawKey}
        />
      {:else}
        <p class="note" data-testid="layer-message">
          This layer's output cannot be plotted here. Select the code layer or a convolution layer.
        </p>
      {/if}
    {/if}

    <TrainingPanel
      {store}
      playing={session.playing}
      disabled={!store.isValid || loadState !== 'ready'}
      showAccuracy={false}
      stats={session.stats}
      onplay={session.play}
      onpause={session.pause}
      onstep={session.step}
      onreset={session.resetModel}
    />

    <LossChart points={session.lossPoints} />
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

  .preset {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  .preset select {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  .scroll {
    overflow-x: auto;
    padding-bottom: var(--space-1);
  }
</style>
