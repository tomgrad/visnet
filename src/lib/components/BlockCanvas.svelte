<script lang="ts">
  import { browser } from '$app/environment';
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

  function handleConnect(connection: { source: string; target: string }): void {
    const intent = connectionToIntent(connection, store.network);
    if (!intent) return;
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
    height: 100%;
    min-height: 320px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg);
  }
</style>
