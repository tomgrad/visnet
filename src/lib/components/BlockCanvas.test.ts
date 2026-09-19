import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import { autoPosition } from '../editor/flow';
import BlockCanvas from './BlockCanvas.svelte';
import {
  capturedEdges,
  capturedHandler,
  capturedNodes,
  capturedProOptions,
  resetCaptured
} from './__stubs__/flowProbe';

vi.mock('@xyflow/svelte', async () => {
  const FlowStub = (await import('./__stubs__/FlowStub.svelte')).default;
  return {
    SvelteFlow: FlowStub,
    Background: FlowStub,
    Controls: FlowStub,
    Handle: FlowStub,
    MarkerType: { ArrowClosed: 'arrowclosed' },
    Position: { Left: 'left', Right: 'right', Top: 'top', Bottom: 'bottom' },
    useSvelteFlow: () => ({
      screenToFlowPosition: (point: { x: number; y: number }) => point
    })
  };
});

interface CapturedEdge {
  id: string;
  animated?: boolean;
  style?: string;
}

function highlighted(): CapturedEdge[] {
  return (capturedEdges() as CapturedEdge[]).filter((edge) => edge.style !== undefined);
}

function canvas() {
  const store = new NetworkStore();
  render(BlockCanvas, { props: { store } });
  return store;
}

function blockDragEvent(
  type: string,
  init: { clientY?: number; relatedTarget?: EventTarget | null; kind?: string } = {}
): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(event, {
    clientX: 10,
    clientY: init.clientY ?? 0,
    relatedTarget: init.relatedTarget ?? null,
    dataTransfer: {
      types: ['application/visnet-block'],
      getData: () => init.kind ?? 'linear'
    }
  });
  return event;
}

async function dragOver(y: number): Promise<void> {
  await fireEvent(screen.getByTestId('canvas'), blockDragEvent('dragover', { clientY: y }));
  await tick();
}

async function dragLeave(relatedTarget: EventTarget | null): Promise<void> {
  await fireEvent(screen.getByTestId('canvas'), blockDragEvent('dragleave', { relatedTarget }));
  await tick();
}

async function drop(y: number): Promise<void> {
  await fireEvent(screen.getByTestId('canvas'), blockDragEvent('drop', { clientY: y }));
  await tick();
}

beforeEach(() => {
  resetCaptured();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('BlockCanvas', () => {
  it('hands the viewport helper to the parent without a reactive identity warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    canvas();
    await tick();
    await tick();

    const mismatches = warn.mock.calls.filter((call) =>
      String(call[0]).includes('state_proxy_equality_mismatch')
    );
    expect(mismatches).toEqual([]);
  });

  it('passes nodes Svelte Flow can structured-clone', async () => {
    canvas();
    await tick();

    const nodes = capturedNodes();
    expect(nodes.length).toBeGreaterThan(0);
    expect(() => structuredClone(nodes[0])).not.toThrow();
    expect(() => structuredClone(nodes)).not.toThrow();
  });

  it('removes a block when a node delete button is clicked', async () => {
    const store = canvas();
    await tick();

    const before = store.network.blocks.length;
    await userEvent.click(screen.getAllByTestId('block-remove')[0]);

    expect(store.network.blocks.length).toBe(before - 1);
  });

  it('ignores a dropped kind that is not in the palette', async () => {
    const store = canvas();
    const before = store.network.blocks.length;
    const addBlock = vi.spyOn(store, 'addBlock');
    const event = blockDragEvent('drop', { kind: 'bogus' });
    await fireEvent(screen.getByTestId('canvas'), event);
    await tick();
    expect(addBlock).not.toHaveBeenCalled();
    expect(store.network.blocks.length).toBe(before);
  });

  it('hides the Svelte Flow attribution', async () => {
    canvas();
    await tick();

    expect(capturedProOptions()).toMatchObject({ hideAttribution: true });
  });

  it('does not offer a delete button for the input or output block', async () => {
    canvas();
    await tick();

    expect(screen.getAllByTestId('block-remove')).toHaveLength(4);
  });
});

describe('drop indicator', () => {
  it('highlights nothing before a drag starts', async () => {
    canvas();
    await tick();
    expect(highlighted()).toEqual([]);
  });

  it('highlights the wire the block will land on', async () => {
    canvas();
    await tick();
    const edges = capturedEdges() as CapturedEdge[];

    await dragOver(400);

    const marked = highlighted();
    expect(marked).toHaveLength(1);
    expect(marked[0].animated).toBe(true);
    expect(marked[0].id).toBe(edges[2].id);
  });

  it('moves the highlight as the pointer moves down', async () => {
    canvas();
    await tick();
    const edges = capturedEdges() as CapturedEdge[];

    await dragOver(150);
    expect(highlighted()[0].id).toBe(edges[0].id);

    await dragOver(700);
    expect(highlighted()[0].id).toBe(edges[3].id);
  });

  it('clears the highlight when the drag leaves the canvas', async () => {
    canvas();
    await tick();

    await dragOver(400);
    expect(highlighted()).toHaveLength(1);

    await dragLeave(document.body);
    expect(highlighted()).toEqual([]);
  });

  it('keeps the highlight when the pointer moves between children', async () => {
    canvas();
    await tick();

    await dragOver(400);
    expect(highlighted()).toHaveLength(1);

    await dragLeave(screen.getAllByTestId('block-node')[0]);
    expect(highlighted()).toHaveLength(1);
  });

  it('clears the highlight once the block is dropped', async () => {
    canvas();
    await tick();

    await dragOver(400);
    expect(highlighted()).toHaveLength(1);

    await drop(400);
    expect(highlighted()).toEqual([]);
  });

  it('ignores a drag that is not a block', async () => {
    canvas();
    await tick();

    const event = new Event('dragover', { bubbles: true, cancelable: true });
    Object.assign(event, { clientY: 400, dataTransfer: { types: ['text/plain'] } });
    await fireEvent(screen.getByTestId('canvas'), event);
    await tick();

    expect(highlighted()).toEqual([]);
  });
});

describe('free positioning', () => {
  it('enables node dragging and commits the position when a drag stops', async () => {
    const store = canvas();
    await tick();

    const handler = capturedHandler('onnodedragstop');
    expect(handler).toBeDefined();

    const id = store.network.blocks[1].id;
    handler?.({ targetNode: { id, position: { x: 44, y: 88 } } });
    await tick();

    expect(store.network.positions[id]).toEqual({ x: 44, y: 88 });
  });

  it('ignores a drag stop with no target node', async () => {
    const store = canvas();
    await tick();

    capturedHandler('onnodedragstop')?.({ targetNode: null, nodes: [], event: null });
    await tick();

    expect(store.isTidy).toBe(true);
    expect(store.canUndo).toBe(false);
  });

  it('places a dropped block at the point it was dropped', async () => {
    const store = canvas();
    await tick();

    const before = new Set(store.network.blocks.map((block) => block.id));
    await drop(400);

    const added = store.network.blocks.find((block) => !before.has(block.id));
    expect(added).toBeDefined();
    expect(store.network.positions[added!.id]).toEqual({ x: 10, y: 400 });
  });

  it('gives a block added by click its auto slot', async () => {
    const store = canvas();
    await tick();

    const created = store.addBlock('linear');
    await tick();

    const index = store.network.blocks.findIndex((block) => block.id === created);
    expect(store.network.positions[created]).toEqual(autoPosition(index));
  });
});
