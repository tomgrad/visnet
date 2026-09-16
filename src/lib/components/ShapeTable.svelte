<script lang="ts">
  import { shapeLabel } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let { store }: { store: NetworkStore } = $props();
</script>

<table class="shapes" data-testid="shape-table">
  <caption>The shape of the data at every step</caption>
  <thead>
    <tr>
      <th scope="col">#</th>
      <th scope="col">Block</th>
      <th scope="col">In</th>
      <th scope="col">Out</th>
      <th scope="col">Parameters</th>
    </tr>
  </thead>
  <tbody>
    {#each store.network.blocks as block, index (block.id)}
      {@const info = store.shapes.perBlock[index]}
      <tr data-testid="shape-row">
        <td>{index + 1}</td>
        <td>{block.kind}</td>
        <td>{shapeLabel(info.inShape) ?? '—'}</td>
        <td>{shapeLabel(info.outShape) ?? '—'}</td>
        <td>{info.paramCount ?? '—'}</td>
      </tr>
    {/each}
  </tbody>
  <tfoot>
    <tr>
      <th scope="row" colspan="4">Total parameters</th>
      <td>{store.shapes.totalParamCount}</td>
    </tr>
  </tfoot>
</table>

<style>
  .shapes {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-xs);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  caption {
    text-align: left;
    padding: var(--space-2) var(--space-3);
    color: var(--color-text-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  th,
  td {
    text-align: left;
    padding: var(--space-1) var(--space-3);
    border-top: 1px solid var(--color-border);
    font-family: var(--font-mono);
  }

  thead th {
    color: var(--color-text-muted);
    font-family: var(--font-sans);
    font-weight: 600;
  }

  tfoot th,
  tfoot td {
    font-weight: 600;
  }
</style>
