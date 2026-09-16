import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import BlockCanvas from './BlockCanvas.svelte';

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

afterEach(() => {
  vi.restoreAllMocks();
});

describe('BlockCanvas', () => {
  it('hands the viewport helper to the parent without a reactive identity warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const store = new NetworkStore();

    render(BlockCanvas, { props: { store, palette: ['linear'], ondragover: () => {} } });
    await tick();
    await tick();

    const mismatches = warn.mock.calls.filter((call) =>
      String(call[0]).includes('state_proxy_equality_mismatch')
    );
    expect(mismatches).toEqual([]);
  });
});
