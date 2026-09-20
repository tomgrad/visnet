<script lang="ts">
  import { getContext } from 'svelte';
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { blockColourHex } from '../editor/colours';
  import { EDITOR_NODE_ACTIONS, type EditorNodeActions } from '../editor/context';
  import { NODE_HEIGHT, NODE_WIDTH, shapeLabel } from '../editor/flow';
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockColour, BlockKind } from '../network/types';

  interface NodeData {
    kind: BlockKind;
    colour?: BlockColour;
    inShape: number[] | null;
    outShape: number[] | null;
    paramCount: number | null;
    index: number;
    removable: boolean;
  }

  let { id, data, selected }: NodeProps = $props();
  const actions = getContext<EditorNodeActions | undefined>(EDITOR_NODE_ACTIONS);
  const info = $derived(data as unknown as NodeData);
  const inLabel = $derived(shapeLabel(info.inShape) ?? '—');
  const outLabel = $derived(shapeLabel(info.outShape) ?? '—');
  const tooltip = $derived(
    `${BLOCK_DESCRIPTIONS[info.kind]} Input ${inLabel}, output ${outLabel}, ${info.paramCount ?? 0} trainable numbers.`
  );
</script>

<div
  class="block"
  class:selected
  style="width: {NODE_WIDTH}px; min-height: {NODE_HEIGHT}px; background: {blockColourHex(
    info.colour
  )}"
  data-testid="block-node"
  title={tooltip}
>
  {#if info.kind !== 'input'}
    <Handle type="target" position={Position.Top} />
  {/if}

  <header>
    <span class="kind">{info.kind}</span>
    {#if info.removable}
      <button
        type="button"
        class="remove"
        aria-label={`Remove ${info.kind}`}
        data-testid="block-remove"
        onclick={() => actions?.removeBlock(id)}
      >
        ×
      </button>
    {/if}
  </header>

  <p class="shapes">{inLabel} → {outLabel}</p>
  <p class="params">{info.paramCount ?? 0} parameters</p>

  {#if info.kind !== 'output'}
    <Handle type="source" position={Position.Bottom} />
  {/if}
</div>

<style>
  .block {
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .block.selected {
    border-color: var(--color-accent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--color-accent) 30%, transparent);
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }

  .kind {
    font-weight: 600;
  }

  .remove {
    border: none;
    background: none;
    color: var(--color-text-muted);
    cursor: pointer;
    font-size: var(--text-base);
    line-height: 1;
    padding: 0 var(--space-1);
  }

  .remove:hover {
    color: var(--color-error);
  }

  .shapes,
  .params {
    margin: var(--space-1) 0 0;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
