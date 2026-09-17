<script lang="ts">
  import { onMount } from 'svelte';
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let {
    store,
    onfit,
    onsave,
    onload,
    saving = false
  }: {
    store: NetworkStore;
    onfit?: () => void;
    onsave?: () => void;
    onload?: () => void;
    saving?: boolean;
  } = $props();

  function isEditingText(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
  }

  function handleKeydown(event: KeyboardEvent): void {
    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier || event.key.toLowerCase() !== 'z') return;
    if (isEditingText(event.target)) return;
    event.preventDefault();
    if (event.shiftKey) store.redo();
    else store.undo();
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  });
</script>

<div class="toolbar">
  <button type="button" data-testid="undo" disabled={!store.canUndo} onclick={() => store.undo()}>
    Undo
  </button>
  <button type="button" data-testid="redo" disabled={!store.canRedo} onclick={() => store.redo()}>
    Redo
  </button>
  <button type="button" data-testid="reset-network" onclick={() => store.reset()}>
    Reset network
  </button>

  <button
    type="button"
    data-testid="tidy-up"
    disabled={Object.keys(store.network.positions).length === 0}
    onclick={() => store.clearPositions()}
  >
    Tidy up
  </button>

  {#if onfit}
    <button type="button" data-testid="fit-view" onclick={onfit}>Fit view</button>
  {/if}

  {#if onsave}
    <button type="button" data-testid="save-model" disabled={saving} onclick={onsave}>
      Save model
    </button>
  {/if}

  {#if onload}
    <button type="button" data-testid="load-model" disabled={saving} onclick={onload}>
      Load model
    </button>
  {/if}
</div>

<style>
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  button {
    padding: var(--space-1) var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
    font-size: var(--text-sm);
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
