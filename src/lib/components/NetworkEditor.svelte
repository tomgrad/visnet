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

</style>
