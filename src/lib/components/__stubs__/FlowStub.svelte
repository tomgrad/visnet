<script lang="ts">
  import type { Component, Snippet } from 'svelte';
  import { recordEdges, recordNodes } from './flowProbe';

  interface StubNode {
    id: string;
    type?: string;
    data: unknown;
    selected?: boolean;
  }

  type StubNodeComponent = Component<{ id: string; data: unknown; selected: boolean }>;

  let {
    nodes = [],
    edges = [],
    nodeTypes = {},
    children
  }: {
    nodes?: StubNode[];
    edges?: unknown[];
    nodeTypes?: Record<string, StubNodeComponent>;
    children?: Snippet;
  } = $props();

  $effect(() => {
    recordNodes(nodes);
  });

  $effect(() => {
    recordEdges(edges);
  });
</script>

{#each nodes as node (node.id)}
  {@const NodeComponent = nodeTypes[node.type ?? 'default']}
  {#if NodeComponent}
    <NodeComponent id={node.id} data={node.data} selected={node.selected ?? false} />
  {/if}
{/each}

{#if children}{@render children()}{/if}
