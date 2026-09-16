<script lang="ts">
  import type { Component, Snippet } from 'svelte';
  import { recordNodes } from './flowProbe';

  interface StubNode {
    id: string;
    type?: string;
    data: unknown;
    selected?: boolean;
  }

  type StubNodeComponent = Component<{ id: string; data: unknown; selected: boolean }>;

  let {
    nodes = [],
    nodeTypes = {},
    children
  }: {
    nodes?: StubNode[];
    nodeTypes?: Record<string, StubNodeComponent>;
    children?: Snippet;
  } = $props();

  $effect(() => {
    recordNodes(nodes);
  });
</script>

{#each nodes as node (node.id)}
  {@const NodeComponent = nodeTypes[node.type ?? 'default']}
  {#if NodeComponent}
    <NodeComponent id={node.id} data={node.data} selected={node.selected ?? false} />
  {/if}
{/each}

{#if children}{@render children()}{/if}
