<script lang="ts">
  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';
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
  import { connectionToIntent, NODE_GAP, NODE_WIDTH, toFlow } from '../editor/flow';
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

  const nodeTypes: NodeTypes = { block: BlockNode as NodeTypes[string] };

  const flow = $derived(toFlow(store.network, store.shapes));

  const nodes = $derived<Node[]>(
    flow.nodes.map((node) => ({
      id: node.id,
      type: 'block',
      position: node.position,
      selected: node.id === store.selectedBlockId,
      data: { ...node.data, onremove: () => store.removeBlock(node.id) }
    }))
  );

  const edges = $derived<Edge[]>(
    flow.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? undefined,
      markerEnd: MarkerType.ArrowClosed,
      animated: false
    }))
  );

  let wrapper: HTMLDivElement | null = $state(null);
  let viewport = $state<{
    screenToFlowPosition: (p: { x: number; y: number }) => { x: number; y: number };
  } | null>(null);
  let connectionHint = $state<string | null>(null);
  let hintTimer: ReturnType<typeof setTimeout> | null = null;

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
    const kind = event.dataTransfer?.getData('application/visnet-block') as BlockKind | undefined;
    ondragover(null);
    if (!kind || !palette.includes(kind)) return;

    const x =
      viewport && wrapper
        ? viewport.screenToFlowPosition({ x: event.clientX, y: event.clientY }).x
        : 0;
    const index = dropIndexFor(x, store.network.blocks.length, NODE_WIDTH, NODE_GAP);
    store.addBlock(kind, index);
  }
</script>

{#if browser}
  <div
    class="canvas"
    role="region"
    aria-label="Network canvas"
    bind:this={wrapper}
    ondragover={(event) => event.preventDefault()}
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
      nodesDraggable={false}
      nodesConnectable
      onconnect={handleConnect}
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
