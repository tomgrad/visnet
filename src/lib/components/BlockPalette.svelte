<script lang="ts">
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockKind } from '../network/types';

  let {
    palette,
    onadd,
    ondragstart
  }: {
    palette: BlockKind[];
    onadd: (kind: BlockKind) => void;
    ondragstart?: (kind: BlockKind, event: DragEvent) => void;
  } = $props();
</script>

<div class="palette">
  <h2>Blocks</h2>
  <ul>
    {#each palette as kind (kind)}
      <li>
        <button
          type="button"
          draggable="true"
          data-testid={`palette-${kind}`}
          title={BLOCK_DESCRIPTIONS[kind]}
          onclick={() => onadd(kind)}
          ondragstart={(event) => ondragstart?.(kind, event)}
        >
          <span class="kind">{kind}</span>
          <span class="description">{BLOCK_DESCRIPTIONS[kind]}</span>
        </button>
      </li>
    {/each}
  </ul>
</div>

<style>
  .palette h2 {
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
    margin: 0 0 var(--space-2);
  }

  .palette ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }

  .palette button {
    width: 100%;
    text-align: left;
    display: grid;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: grab;
  }

  .palette button:hover {
    border-color: var(--color-accent);
  }

  .kind {
    font-weight: 600;
  }

  .description {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
  }
</style>
