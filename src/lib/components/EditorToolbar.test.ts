import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
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
