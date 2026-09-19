import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import EditorToolbar from './EditorToolbar.svelte';

function toolbar() {
  const store = new NetworkStore();
  render(EditorToolbar, { props: { store } });
  return store;
}

function pressCtrlZ(target: EventTarget): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    key: 'z',
    ctrlKey: true,
    bubbles: true,
    cancelable: true
  });
  target.dispatchEvent(event);
  return event;
}

describe('EditorToolbar keyboard shortcuts', () => {
  it('undoes the last change on Ctrl+Z', async () => {
    const store = toolbar();
    store.addBlock('relu');
    await tick();
    const before = store.network.blocks.length;

    pressCtrlZ(window);

    expect(store.network.blocks.length).toBe(before - 1);
  });

  it('leaves Ctrl+Z to the browser while an editable field is focused', async () => {
    const store = toolbar();
    store.addBlock('relu');
    await tick();
    const before = store.network.blocks.length;

    const input = document.createElement('input');
    document.body.appendChild(input);
    const event = pressCtrlZ(input);
    input.remove();

    expect(store.network.blocks.length).toBe(before);
    expect(event.defaultPrevented).toBe(false);
  });
});

describe('EditorToolbar export', () => {
  const originalCreate = URL.createObjectURL;
  const originalRevoke = URL.revokeObjectURL;

  afterEach(() => {
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
    vi.restoreAllMocks();
  });

  it('downloads the network as a TS module', async () => {
    const store = toolbar();
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => 'blob:mock');
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    await userEvent.click(screen.getByTestId('export-model'));

    expect(createObjectURL).toHaveBeenCalledTimes(1);
    const blob = createObjectURL.mock.calls[0][0] as unknown as Blob;
    expect(await blob.text()).toContain('satisfies Network');
    expect(await blob.text()).toContain(store.network.blocks[0].id);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock');
  });
});
