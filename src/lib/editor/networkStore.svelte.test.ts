import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { NetworkStore } from './networkStore.svelte';

function store(): NetworkStore {
  const instance = new NetworkStore();
  instance.expectedClasses = 2;
  return instance;
}

const ids = (instance: NetworkStore) => instance.network.blocks.map((block) => block.id);
const kinds = (instance: NetworkStore) => instance.network.blocks.map((block) => block.kind);

describe('selection', () => {
  it('selects and clears', () => {
    const instance = store();
    instance.select('x');
    expect(instance.selectedBlockId).toBe('x');
    instance.select(null);
    expect(instance.selectedBlockId).toBeNull();
  });
});

describe('addBlock', () => {
  it('inserts after the selection and selects the new block', () => {
    const instance = store();
    const anchor = instance.network.blocks[1].id;
    instance.select(anchor);
    const created = instance.addBlock('sigmoid');
    expect(kinds(instance)).toEqual([
      'input',
      'linear',
      'sigmoid',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
    expect(instance.network.blocks[2].id).toBe(created);
    expect(instance.selectedBlockId).toBe(created);
  });

  it('inserts before the output when nothing is selected', () => {
    const instance = store();
    instance.addBlock('flatten');
    expect(kinds(instance)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'flatten',
      'output'
    ]);
  });

  it('honours an explicit index', () => {
    const instance = store();
    instance.addBlock('sigmoid', 1);
    expect(kinds(instance)[1]).toBe('sigmoid');
  });
});

describe('removeBlock', () => {
  it('removes an interior block', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.removeBlock(target);
    expect(ids(instance)).not.toContain(target);
  });

  it('refuses to remove the input or output and records no history entry', () => {
    const instance = store();
    instance.removeBlock(instance.network.blocks[0].id);
    instance.removeBlock(instance.network.blocks[instance.network.blocks.length - 1].id);
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
    expect(instance.canUndo).toBe(false);
  });

  it('clears the selection when the selected block is removed', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.select(target);
    instance.removeBlock(target);
    expect(instance.selectedBlockId).toBeNull();
  });
});

describe('moveBlock', () => {
  it('reorders the chain', () => {
    const instance = store();
    const moving = instance.network.blocks[1].id;
    instance.moveBlock(1, 3);
    expect(ids(instance)[3]).toBe(moving);
  });

  it('records nothing for a no-op move', () => {
    const instance = store();
    instance.moveBlock(1, 1);
    expect(instance.canUndo).toBe(false);
  });

  it('moves the selection by an offset and ignores the ends', () => {
    const instance = store();
    const first = instance.network.blocks[1].id;
    instance.select(first);
    instance.moveSelectedBy(-1);
    expect(ids(instance)[1]).toBe(first);
    expect(instance.canUndo).toBe(false);

    instance.moveSelectedBy(1);
    expect(ids(instance)[2]).toBe(first);
    expect(instance.canUndo).toBe(true);
  });
});

describe('updateBlock', () => {
  it('applies a valid patch with no announcement', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.updateBlock(target, { units: 16 });
    expect(instance.network.blocks[1]).toMatchObject({ units: 16 });
    expect(instance.announcements).toEqual([]);
  });

  it('announces an automatic correction', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    instance.updateBlock(target, { units: 0 });
    expect(instance.network.blocks[1]).toMatchObject({ units: 1 });
    expect(instance.announcements).toEqual([
      'Units changed from 0 to 1. A layer must produce at least one number.'
    ]);
  });

  it('dismisses announcements', () => {
    const instance = store();
    instance.announce('something');
    instance.dismissAnnouncements();
    expect(instance.announcements).toEqual([]);
  });

  it('re-clamps downstream parameters when an upstream shape changes', () => {
    const instance = store();
    instance.addBlock('conv2d', 1);
    instance.updateBlock(instance.network.blocks[0].id, { shape: [28, 28, 1] });
    instance.updateBlock(instance.network.blocks[1].id, { kernelSize: 7 });
    instance.dismissAnnouncements();

    instance.updateBlock(instance.network.blocks[0].id, { shape: [4, 4, 1] });

    expect(instance.network.blocks[1]).toMatchObject({ kernelSize: 4 });
    expect(instance.announcements).toEqual([
      'Kernel size changed from 7 to 4 because the incoming data is 4×4.'
    ]);
  });
});

describe('updateTraining', () => {
  it('changes only the training configuration', () => {
    const instance = store();
    const before = ids(instance);
    instance.updateTraining({ optimizer: 'sgd', learningRate: 0.5 });
    expect(instance.network.training).toMatchObject({ optimizer: 'sgd', learningRate: 0.5 });
    expect(ids(instance)).toEqual(before);
  });
});

describe('history', () => {
  it('undoes and redoes a change and reports availability', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;

    expect(instance.canUndo).toBe(false);
    instance.updateBlock(target, { units: 32 });
    expect(instance.canUndo).toBe(true);

    instance.undo();
    expect(instance.network.blocks[1]).toMatchObject({ units: 8 });
    expect(instance.canRedo).toBe(true);

    instance.redo();
    expect(instance.network.blocks[1]).toMatchObject({ units: 32 });
  });

  it('does nothing when there is nothing to undo', () => {
    const instance = store();
    instance.undo();
    instance.redo();
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
  });

  it('clears history and selection on load', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[1].id, { units: 32 });
    instance.select(instance.network.blocks[1].id);
    instance.load(createEmptyNetwork());
    expect(instance.canUndo).toBe(false);
    expect(instance.canRedo).toBe(false);
    expect(instance.selectedBlockId).toBeNull();
    expect(instance.announcements).toEqual([]);
  });

  it('resets to a fresh default network', () => {
    const instance = store();
    instance.removeBlock(instance.network.blocks[1].id);
    instance.reset();
    expect(kinds(instance)).toEqual(['input', 'linear', 'relu', 'linear', 'softmax', 'output']);
    expect(instance.canUndo).toBe(false);
  });
});

describe('derived state', () => {
  it('reflects the network', () => {
    const instance = store();
    expect(instance.isValid).toBe(true);
    expect(instance.errors).toEqual([]);
    expect(instance.warnings).toEqual([]);
    expect(instance.paramCount).toBe(42);

    instance.removeBlock(instance.network.blocks[4].id);
    expect(instance.isValid).toBe(true);
    expect(instance.warnings.map((issue) => issue.title)).toContain(
      'Add a Softmax for probabilities'
    );
  });

  it('exposes the selected block', () => {
    const instance = store();
    const target = instance.network.blocks[1].id;
    expect(instance.selectedBlock).toBeNull();
    instance.select(target);
    expect(instance.selectedBlock).toMatchObject({ id: target, kind: 'linear' });
  });

  it('reports errors that block training', () => {
    const instance = store();
    instance.updateBlock(instance.network.blocks[0].id, { shape: [28, 28, 1] });
    expect(instance.isValid).toBe(false);
    expect(instance.errors.map((issue) => issue.title)).toContain('Linear layer needs a flat list');
  });
});
