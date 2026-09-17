import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import InspectorPanel from './InspectorPanel.svelte';

function storeWithSelection(index: number): NetworkStore {
  const store = new NetworkStore();
  store.select(store.network.blocks[index].id);
  return store;
}

describe('InspectorPanel', () => {
  it('prompts the user when nothing is selected', () => {
    render(InspectorPanel, { props: { store: new NetworkStore() } });
    expect(screen.getByTestId('inspector-empty')).toBeTruthy();
  });

  it('shows the selected block and its incoming shape', () => {
    render(InspectorPanel, { props: { store: storeWithSelection(1) } });
    expect(screen.getByTestId('inspector').textContent).toContain('linear');
    expect(screen.getByTestId('inspector-incoming').textContent).toContain('[2]');
  });

  it('edits a linear layer unit count', async () => {
    const store = storeWithSelection(1);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-units'), { target: { value: '16' } });
    expect(store.network.blocks[1]).toMatchObject({ units: 16 });
  });

  it('leaves the network alone when a numeric field is cleared', async () => {
    const store = storeWithSelection(1);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-units'), { target: { value: '' } });
    expect(store.network.blocks[1]).toMatchObject({ units: 8 });
  });

  it('restores the units field and explains a non-integer entry', async () => {
    const store = storeWithSelection(1);
    render(InspectorPanel, { props: { store } });
    const field = screen.getByTestId('param-units') as HTMLInputElement;
    await fireEvent.change(field, { target: { value: '2.5' } });
    expect(store.network.blocks[1]).toMatchObject({ units: 8 });
    expect(field.value).toBe('8');
    expect(screen.getByTestId('param-error').textContent).toContain('whole number');
  });

  it('rejects a shape entry with an invalid part instead of dropping it silently', async () => {
    const store = storeWithSelection(0);
    render(InspectorPanel, { props: { store } });
    const field = screen.getByTestId('param-shape') as HTMLInputElement;
    await fireEvent.change(field, { target: { value: '2, -1, x' } });
    expect(store.network.blocks[0]).toMatchObject({ shape: [2] });
    expect(field.value).toBe('2');
    expect(screen.getByTestId('param-error').textContent).toContain('positive whole numbers');
  });

  it('shows no unit control for an activation block', () => {
    render(InspectorPanel, { props: { store: storeWithSelection(2) } });
    expect(screen.queryByTestId('param-units')).toBeNull();
    expect(screen.getByTestId('inspector').textContent).toContain('relu');
  });

  it('limits convolution kernel choices to the incoming image size', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('conv2d', 1);
    store.updateBlock(store.network.blocks[1].id, { kernelSize: 3 });
    render(InspectorPanel, { props: { store } });

    const options = Array.from(
      screen.getByTestId('param-kernel-size').querySelectorAll('option')
    ).map((option) => option.getAttribute('value'));
    expect(options).toEqual(['1', '2', '3', '4']);
    expect(screen.getByTestId('param-kernel-size')).toBeTruthy();
  });

  it('describes what each padding choice does to the image size', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    store.addBlock('conv2d', 1);
    render(InspectorPanel, { props: { store } });

    const labels = Array.from(screen.getByTestId('param-padding').querySelectorAll('option')).map(
      (option) => option.textContent ?? ''
    );
    expect(labels.some((label) => label.includes('28×28'))).toBe(true);
    expect(labels.some((label) => label.includes('26×26'))).toBe(true);
  });

  it('moves the selected block and refuses to move the input or output', async () => {
    const store = storeWithSelection(1);
    const moving = store.network.blocks[1].id;
    render(InspectorPanel, { props: { store } });
    await userEvent.click(screen.getByTestId('move-right'));
    expect(store.network.blocks[2].id).toBe(moving);
  });

  it('disables the move buttons for the input and output blocks', () => {
    const store = storeWithSelection(0);
    render(InspectorPanel, { props: { store } });
    expect((screen.getByTestId('move-left') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('move-right') as HTMLButtonElement).disabled).toBe(true);
  });

  it('offers pool size choices limited to the incoming image', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('maxpool2d', 1);
    store.updateBlock(store.network.blocks[1].id, { poolSize: 3 });
    render(InspectorPanel, { props: { store } });

    const options = Array.from(
      screen.getByTestId('param-pool-size').querySelectorAll('option')
    ).map((option) => option.getAttribute('value'));
    expect(options).toEqual(['1', '2', '3', '4']);
  });

  it('changes the pool size', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [6, 4, 1] });
    store.addBlock('maxpool2d', 1);
    store.select(store.network.blocks[1].id);
    render(InspectorPanel, { props: { store } });
    await fireEvent.change(screen.getByTestId('param-pool-size'), { target: { value: '4' } });
    expect(store.network.blocks[1]).toMatchObject({ poolSize: 4 });
  });

  it('describes what each padding choice does to a pooled image', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    store.addBlock('maxpool2d', 1);
    store.updateBlock(store.network.blocks[1].id, { poolSize: 3 });
    render(InspectorPanel, { props: { store } });

    const labels = Array.from(screen.getByTestId('param-padding').querySelectorAll('option')).map(
      (option) => option.textContent ?? ''
    );
    expect(labels.some((label) => label.includes('14×14'))).toBe(true);
    expect(labels.some((label) => label.includes('13×13'))).toBe(true);
  });
});
