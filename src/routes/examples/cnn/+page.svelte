<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import SampleGrid from '$lib/components/SampleGrid.svelte';
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
  import { createExperiment } from '$lib/examples/experiment.svelte';
  import type { ModelData } from '$lib/examples/runtime';
  import { createBrowserStorage } from '$lib/persist/storage';

  const store = new NetworkStore(createCnnNetwork());
  store.expectedClasses = 10;
  store.expectedInputShape = [28, 28, 1];
  const storage = createBrowserStorage(CNN_STORAGE_KEYS);
  const session = createExperiment({ store, weightsId: CNN_WEIGHTS_ID, storage });

  let testData = $state.raw<ImageDataset | null>(null);
  let sampleData = $state.raw<ModelData | null>(null);
  let trainData = $state.raw<ImageDataset | null>(null);
  let loadState = $state<'loading' | 'ready' | 'unavailable'>('loading');
  let trainCount = $state(0);
  let currentSampleData: ModelData | null = null;
  const sampleIndices = $derived(testData ? defaultSampleIndices(testData.count) : []);

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

  onDestroy(() => {
    session.runtime?.disposeData(currentSampleData);
    currentSampleData = null;
  });

  $effect(() => {
    const api = session.runtime;
    if (!api || !trainData) return;
    session.setData(api.imagesToTensors(trainData));
  });

  $effect(() => {
    const api = session.runtime;
    const test = testData;
    if (!api || !test) return;
    const next = api.imagesToTensors(test, sampleIndices);
    api.disposeData(currentSampleData);
    currentSampleData = next;
    sampleData = next;
  });
</script>

<ExampleLayout
  title="Handwritten digits"
  intro="Build a small convolutional network, train it on handwritten digits, and watch it get them right."
>
  {#snippet editor()}
    <NetworkEditor
      {store}
      palette={CNN_PALETTE}
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
      <p class="note">Training on {trainCount} digits, checking against {testData?.count}.</p>
      <SampleGrid
        model={session.model}
        dataset={testData}
        indices={sampleIndices}
        sampleXs={sampleData?.xs ?? null}
        redrawKey={session.redrawKey}
        onerror={(message) => (session.banner = message)}
      />
    {/if}

    <TrainingPanel
      {store}
      playing={session.playing}
      disabled={!store.isValid || loadState !== 'ready'}
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
</style>
