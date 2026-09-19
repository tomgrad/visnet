<script lang="ts">
  import { BLOCK_CATEGORIES } from '../network/categories';
  import { BLOCK_DESCRIPTIONS } from '../network/descriptions';
  import type { BlockKind } from '../network/types';

  let {
    ondragstart
  }: {
    ondragstart?: (kind: BlockKind, event: DragEvent) => void;
  } = $props();

  let collapsed = $state<Record<string, boolean>>({});

  function toggle(id: string): void {
    collapsed = { ...collapsed, [id]: !collapsed[id] };
  }
</script>

<div class="palette">
  <h2>Blocks</h2>
  {#each BLOCK_CATEGORIES as category (category.id)}
    <section class="category">
      <button
        type="button"
        class="category-header"
        data-testid={`palette-category-${category.id}`}
        aria-expanded={!collapsed[category.id]}
        onclick={() => toggle(category.id)}
      >
        {category.label}
      </button>
      {#if !collapsed[category.id]}
        <ul>
          {#each category.kinds as kind (kind)}
            <li>
              <button
                type="button"
                draggable="true"
                data-testid={`palette-${kind}`}
                title={BLOCK_DESCRIPTIONS[kind]}
                ondragstart={(event) => ondragstart?.(kind, event)}
              >
                <span class="kind">{kind}</span>
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
  {/each}
</div>

<style>
  .palette h2 {
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
    margin: 0 0 var(--space-2);
  }

  .category {
    display: grid;
    gap: var(--space-2);
  }

  .category + .category {
    margin-top: var(--space-3);
  }

  .category-header {
    width: 100%;
    text-align: left;
    padding: var(--space-1) 0;
    background: none;
    border: none;
    border-bottom: 1px solid var(--color-border);
    cursor: pointer;
    font: inherit;
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .category ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }

  .palette li button {
    width: 100%;
    text-align: left;
    padding: var(--space-1) var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: grab;
  }

  .palette li button:hover {
    border-color: var(--color-accent);
  }

  .kind {
    font-weight: 600;
  }
</style>
