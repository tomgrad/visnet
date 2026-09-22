<script lang="ts">
  import { untrack } from 'svelte';
  import type { VaeController } from '../vaeController.svelte';
  import { createVaeExperiment, type VaeExperiment } from '../experiment.svelte';

  let {
    controller,
    onready
  }: {
    controller: VaeController;
    onready: (session: VaeExperiment) => void;
  } = $props();

  const session = untrack(() =>
    createVaeExperiment({ controller, weightsId: 'vae', storage: null })
  );
  untrack(() => onready(session));
</script>
