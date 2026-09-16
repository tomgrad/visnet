<script lang="ts">
  import { useSvelteFlow } from '@xyflow/svelte';

  let {
    viewport = $bindable()
  }: {
    viewport: {
      screenToFlowPosition: (point: { x: number; y: number }) => { x: number; y: number };
    } | null;
  } = $props();

  const flow = useSvelteFlow();
  $effect(() => {
    if (viewport !== flow) {
      viewport = flow;
    }
  });
</script>
