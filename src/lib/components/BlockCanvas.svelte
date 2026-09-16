<script lang="ts">
  import { browser } from '$app/environment';
  import { onDestroy, setContext } from 'svelte';
  import {
    Background,
    Controls,
    MarkerType,
    SvelteFlow,
    type Edge,
    type Node,
    type NodeTypes
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { EDITOR_NODE_ACTIONS, type EditorNodeActions } from '../editor/context';
  import { connectionToIntent, nodeCentre, toFlow } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import { dropIndexFor } from '../editor/placement';
  import type { BlockKind } from '../network/types';
  import BlockNode from './BlockNode.svelte';
  import CanvasViewport from './CanvasViewport.svelte';

  let {
    store,
    palette,
    ondragover
  }: {
    store: NetworkStore;
    palette: BlockKind[];
    ondragover: (kind: BlockKind | null) => void;
  } = $props();

  setContext<EditorNodeActions>(EDITOR_NODE_ACTIONS, {
    removeBlock: (id) => store.removeBlock(id)
  });

  const nodeTypes: NodeTypes = { block: BlockNode as NodeTypes[string] };

  const flow = $derived(toFlow(store.network, store.shapes));

  const nodes = $derived<Node[]>(
    flow.nodes.map((node) => ({
      id: node.id,
      type: 'block',
      position: node.position,
      selected: node.id === store.selectedBlockId,
      data: node.data
    }))
  );

  const edges = $derived<Edge[]>(
    flow.edges.map((edge, index) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? undefined,
      markerEnd: MarkerType.ArrowClosed,
      animated: dropIndex !== null && index === dropIndex - 1,
      style:
        dropIndex !== null && index === dropIndex - 1
          ? 'stroke: var(--color-accent); stroke-width: 3px'
          : undefined
    }))
  );

  let wrapper: HTMLDivElement | null = $state(null);
  let viewport = $state.raw<{
    screenToFlowPosition: (p: { x: number; y: number }) => { x: number; y: number };
  } | null>(null);
  let dropIndex = $state<number | null>(null);
  let connectionHint = $state<string | null>(null);
  let hintTimer: ReturnType<typeof setTimeout> | null = null;

  const DRAG_TYPE = 'application/visnet-block';

  function isBlockDrag(event: DragEvent): boolean {
    return Array.from(event.dataTransfer?.types ?? []).includes(DRAG_TYPE);
  }

  const centres = $derived(flow.nodes.map((node) => nodeCentre(node.position)));

  function dropPoint(event: DragEvent): { x: number; y: number } {
    if (!viewport || !wrapper) return { x: 0, y: 0 };
    return viewport.screenToFlowPosition({ x: event.clientX, y: event.clientY });
  }

  function handleDragOver(event: DragEvent): void {
    event.preventDefault();
    if (!isBlockDrag(event)) return;
    dropIndex = dropIndexFor(dropPoint(event), centres);
  }

  function handleDragLeave(event: DragEvent): void {
    const related = event.relatedTarget;
    if (related instanceof Node && wrapper?.contains(related)) return;
    dropIndex = null;
  }

  function showConnectionHint(): void {
    connectionHint =
      'That connection is not allowed. Blocks form a single chain from Input to Output.';
    if (hintTimer) clearTimeout(hintTimer);
    hintTimer = setTimeout(() => {
      connectionHint = null;
      hintTimer = null;
    }, 4000);
  }

  onDestroy(() => {
    if (hintTimer) clearTimeout(hintTimer);
  });

  function handleConnect(connection: { source: string; target: string }): void {
    const intent = connectionToIntent(connection, store.network);
    if (!intent) {
      showConnectionHint();
      return;
    }
    store.moveBlock(intent.from, intent.to);
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    const kind = event.dataTransfer?.getData(DRAG_TYPE) as BlockKind | undefined;
    const point = dropPoint(event);
    const index = dropIndex ?? dropIndexFor(point, centres);
    dropIndex = null;
    ondragover(null);
    if (!kind || !palette.includes(kind)) return;

    store.addBlock(kind, index, point);
  }
</script>

{#if browser}
  <div
    class="canvas"
    role="region"
    aria-label="Network canvas"
    bind:this={wrapper}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
    data-testid="canvas"
  >
    {#if connectionHint}
      <p class="connection-hint" role="status" data-testid="connection-hint">{connectionHint}</p>
    {/if}
    <SvelteFlow
      {nodes}
      {edges}
      {nodeTypes}
      fitView
      nodesDraggable
      nodesConnectable
      onconnect={handleConnect}
      onnodedragstop={({ targetNode }) => {
        if (targetNode) store.setPosition(targetNode.id, targetNode.position);
      }}
      onnodeclick={({ node }) => store.select(node.id)}
      onpaneclick={() => store.select(null)}
    >
      <Background />
      <Controls />
      <CanvasViewport bind:viewport />
    </SvelteFlow>
  </div>
{/if}

<style>
  .canvas {
    position: relative;
    height: 100%;
    min-height: 320px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg);
  }

  .connection-hint {
    position: absolute;
    z-index: 1;
    top: var(--space-2);
    left: var(--space-2);
    right: var(--space-2);
    margin: 0;
    padding: var(--space-1) var(--space-2);
    background: color-mix(in srgb, var(--color-warning) 12%, var(--color-surface));
    border: 1px solid var(--color-warning);
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    pointer-events: none;
  }
</style>
