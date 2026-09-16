<script lang="ts">
  import type * as tf from '@tensorflow/tfjs';
  import type { ImageDataset } from '../../data/images';
  import SampleGrid from '../SampleGrid.svelte';

  let {
    model,
    dataset,
    indices,
    sampleXs,
    onerror
  }: {
    model: tf.LayersModel | null;
    dataset: ImageDataset | null;
    indices: number[];
    sampleXs: tf.Tensor | null;
    onerror?: (message: string) => void;
  } = $props();

  let redrawKey = $state(0);

  export function bump(): void {
    redrawKey += 1;
  }
</script>

<SampleGrid {model} {dataset} {indices} {sampleXs} {redrawKey} {onerror} />
