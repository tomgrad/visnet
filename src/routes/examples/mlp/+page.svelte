<script lang="ts">
  import { onMount } from 'svelte';
  import DecisionBoundary from '$lib/components/DecisionBoundary.svelte';
  import ExampleLayout from '$lib/components/ExampleLayout.svelte';
  import LatentSpace from '$lib/components/LatentSpace.svelte';
  import LossChart from '$lib/components/LossChart.svelte';
  import NetworkEditor from '$lib/components/NetworkEditor.svelte';
  import TrainingPanel from '$lib/components/TrainingPanel.svelte';
  import { GENERATOR_DESCRIPTIONS, GENERATOR_NAMES } from '$lib/data/points';
  import { NetworkStore } from '$lib/editor/networkStore.svelte';
  import { createExperiment } from '$lib/examples/experiment.svelte';
  import { DatasetStore } from '$lib/examples/mlp/datasetStore.svelte';
  import { CLASS_LABELS, MLP_STORAGE_KEYS, MLP_WEIGHTS_ID } from '$lib/examples/mlp/example';
  import { createBrowserStorage } from '$lib/persist/storage';

  const store = new NetworkStore();
  const datasetStore = new DatasetStore();
  store.expectedClasses = 2;
  const storage = createBrowserStorage(MLP_STORAGE_KEYS);
  const session = createExperiment({ store, weightsId: MLP_WEIGHTS_ID, storage });

  onMount(() => {
    if (!storage) return;
    const savedDataset = storage.loadDataset();
    if (savedDataset) datasetStore.dataset = savedDataset;
  });

  $effect(() => {
    const api = session.runtime;
    if (!api) return;
    session.setData(api.toTensors(datasetStore.dataset));
  });

  $effect(() => {
    const dataset = datasetStore.dataset;
    if (!storage) return;
    const timer = setTimeout(() => {
      try {
        storage.saveDataset(dataset);
      } catch {
        session.announce('Your work could not be saved. The browser storage may be full.');
      }
    }, 500);
    return () => clearTimeout(timer);
  });
</script>

<ExampleLayout
  title="Points in 2D"
  intro="Build a small network, train it on coloured points, and watch the boundary between the two classes take shape."
>
  {#snippet editor()}
    <NetworkEditor {store} onsave={session.save} onload={session.load} saving={session.saving} />
  {/snippet}

  {#snippet experiment()}
    {#if session.banner}
      <p class="banner" role="status">{session.banner}</p>
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
      model={session.model}
      redrawKey={session.redrawKey}
      dataset={datasetStore.dataset}
      selectedLabel={datasetStore.selectedLabel}
      onaddpoint={(x, y) => datasetStore.addPoint(x, y)}
      caption={!session.runtime
        ? 'Loading the network. The boundary appears in a moment.'
        : store.isValid
          ? 'Each coloured area is the class the network predicts at that spot. Click to add a point.'
          : 'Fix the problems listed in the editor before the boundary can be drawn.'}
    />

    <LatentSpace
      model={session.model}
      {store}
      dataset={datasetStore.dataset}
      redrawKey={session.redrawKey}
    />

    <TrainingPanel
      {store}
      playing={session.playing}
      disabled={!store.isValid}
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
