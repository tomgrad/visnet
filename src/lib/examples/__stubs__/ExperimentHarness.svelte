<script lang="ts">
  import { untrack } from 'svelte';
  import type { NetworkStore } from '../../editor/networkStore.svelte';
  import type { NetworkStorage } from '../../persist/storage';
  import { createExperiment, type Experiment } from '../experiment.svelte';

  let {
    store,
    storage = null,
    onready
  }: {
    store: NetworkStore;
    storage?: NetworkStorage | null;
    onready: (session: Experiment) => void;
  } = $props();

  const session = untrack(() => createExperiment({ store, weightsId: 'test', storage }));
  untrack(() => onready(session));
</script>
