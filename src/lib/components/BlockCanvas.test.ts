import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import BlockCanvas from './BlockCanvas.svelte';
import { capturedNodes, resetCapturedNodes } from './__stubs__/flowProbe';

vi.mock('@xyflow/svelte', async () => {
  const FlowStub = (await import('./__stubs__/FlowStub.svelte')).default;
  return {
    SvelteFlow: FlowStub,
    Background: FlowStub,
    Controls: FlowStub,
    Handle: FlowStub,
    MarkerType: { ArrowClosed: 'arrowclosed' },
    Position: { Left: 'left', Right: 'right' },
    useSvelteFlow: () => ({
      screenToFlowPosition: (point: { x: number; y: number }) => point
    })
  };
});

function canvas() {
  const store = new NetworkStore();
  render(BlockCanvas, { props: { store, palette: ['linear'], ondragover: () => {} } });
  return store;
}

beforeEach(() => {
  resetCapturedNodes();
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

  it('does not offer a delete button for the input or output block', async () => {
    canvas();
    await tick();

    const removable = screen.getAllByTestId('block-remove').length;
    expect(removable).toBe(4);
  });
});
