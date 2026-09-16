<script lang="ts">
  import type * as tf from '@tensorflow/tfjs';
  import type { PointDataset } from '../data/points';
  import DecisionBoundary from './DecisionBoundary.svelte';

  let {
    model = null,
    dataset,
    selectedLabel = 0,
    onaddpoint
  }: {
    model?: tf.LayersModel | null;
    dataset: PointDataset;
    selectedLabel?: 0 | 1;
    onaddpoint: (x: number, y: number) => void;
  } = $props();

  let redrawKey = $state(0);

  export function bump(): void {
    redrawKey += 1;
  }
</script>

<DecisionBoundary {model} {dataset} {selectedLabel} {onaddpoint} {redrawKey} />
