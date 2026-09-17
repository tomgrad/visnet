<script lang="ts">
  import { onMount, type Component } from 'svelte';
  import type { BlockKind } from '../network/types';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import BlockPalette from './BlockPalette.svelte';
  import EditorToolbar from './EditorToolbar.svelte';
  import InspectorPanel from './InspectorPanel.svelte';
  import IssuesPanel from './IssuesPanel.svelte';
  import ShapeTable from './ShapeTable.svelte';

  type CanvasProps = {
    store: NetworkStore;
    palette: BlockKind[];
  };

  let {
    store,
    palette,
    onsave,
    onload,
    saving = false
  }: {
    store: NetworkStore;
    palette: BlockKind[];
    onsave?: () => void;
    onload?: () => void;
    saving?: boolean;
  } = $props();

  let Canvas = $state<Component<CanvasProps> | null>(null);

  onMount(async () => {
    const module = await import('./BlockCanvas.svelte');
    Canvas = module.default as Component<CanvasProps>;
  });
</script>

<section class="editor" data-testid="network-editor">
  <EditorToolbar {store} {onsave} {onload} {saving} />

  <div class="body">
    <aside class="left">
      <BlockPalette
        {palette}
        ondragstart={(kind, event) => {
          event.dataTransfer?.setData('application/visnet-block', kind);
        }}
      />
    </aside>

    <div class="middle">
      {#if Canvas}
        <Canvas {store} {palette} />
      {/if}
    </div>

    <aside class="right">
      {#if store.announcements.length > 0}
        <div class="announcements" data-testid="announcements" role="status" aria-live="polite">
          <ul>
            {#each store.announcements as announcement, index (index)}
              <li>{announcement}</li>
            {/each}
          </ul>
          <button type="button" onclick={() => store.dismissAnnouncements()}>Dismiss</button>
        </div>
      {/if}
      <InspectorPanel {store} />
      <ShapeTable {store} />
    </aside>
  </div>

  <IssuesPanel {store} />
</section>

<style>
  .editor {
    display: grid;
    gap: var(--space-3);
    container: editor / inline-size;
  }

  .body {
    display: grid;
    gap: var(--space-3);
    grid-template-columns: minmax(180px, 220px) minmax(320px, 1fr) minmax(260px, 300px);
    align-items: start;
  }

  @container editor (max-width: 790px) {
    .body {
      grid-template-columns: 1fr;
    }
  }

  .left,
  .right {
    display: grid;
    gap: var(--space-3);
  }

  .middle {
    min-height: 480px;
    height: 640px;
  }

  .announcements {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-accent) 12%, var(--color-surface));
    border: 1px solid var(--color-accent);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .announcements ul {
    margin: 0;
    padding-left: var(--space-4);
  }

  .announcements button {
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    cursor: pointer;
  }
</style>
