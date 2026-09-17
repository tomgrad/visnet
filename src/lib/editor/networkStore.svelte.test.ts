import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { Network } from '../network/types';
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
  it('stores a unit count without an announcement', () => {
    const store = new NetworkStore();
    const id = store.network.blocks[1].id;
    store.updateBlock(id, { units: 4 });
    expect(store.network.blocks[1]).toMatchObject({ units: 4 });
  });

  it('stores an invalid patch without correcting it', () => {
    const store = new NetworkStore();
    const id = store.network.blocks[1].id;
    store.updateBlock(id, { units: 0 });
    expect(store.network.blocks[1]).toMatchObject({ units: 0 });
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

describe('task', () => {
  it('defaults to the classification task', () => {
    expect(new NetworkStore().task).toBe('classification');
  });

  it('suppresses the image-without-convolution warning for reconstruction', () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    expect(store.warnings.some((w) => w.title === 'Image input without a Convolution layer')).toBe(
      true
    );
    store.task = 'reconstruction';
    expect(store.warnings.some((w) => w.title === 'Image input without a Convolution layer')).toBe(
      false
    );
  });
});

describe('expected input shape', () => {
  it('surfaces a mismatch as a warning', () => {
    const instance = store();
    instance.expectedInputShape = [28, 28, 1];

    expect(instance.warnings.map((issue) => issue.title)).toContain(
      'Input shape does not match the data'
    );
  });

  it('says nothing when the shape matches', () => {
    const instance = store();
    instance.expectedInputShape = [2];

    expect(instance.warnings.map((issue) => issue.title)).not.toContain(
      'Input shape does not match the data'
    );
  });
});

describe('positions', () => {
  it('stores a moved position and undoes it', () => {
    const instance = store();
    const id = instance.network.blocks[1].id;

    expect(instance.network.positions).toEqual({});
    instance.setPosition(id, { x: 40, y: 90 });

    expect(instance.network.positions[id]).toEqual({ x: 40, y: 90 });
    expect(instance.canUndo).toBe(true);

    instance.undo();
    expect(instance.network.positions).toEqual({});
  });

  it('records nothing when the same position is set twice', () => {
    const instance = store();
    const id = instance.network.blocks[1].id;

    instance.setPosition(id, { x: 40, y: 90 });
    expect(instance.canUndo).toBe(true);

    instance.setPosition(id, { x: 40, y: 90 });
    instance.undo();

    expect(instance.network.positions).toEqual({});
    expect(instance.canUndo).toBe(false);
  });

  it('compares against the auto slot, not just stored positions', () => {
    const instance = store();
    const id = instance.network.blocks[1].id;

    instance.setPosition(id, { x: 0, y: 170 });
    expect(instance.canUndo).toBe(false);
    expect(instance.network.positions).toEqual({});
  });

  it('ignores an unknown block', () => {
    const instance = store();
    instance.setPosition('missing', { x: 1, y: 2 });
    expect(instance.network.positions).toEqual({});
    expect(instance.canUndo).toBe(false);
  });

  it('clears every position in one undoable step', () => {
    const instance = store();
    instance.setPosition(instance.network.blocks[1].id, { x: 10, y: 10 });
    instance.setPosition(instance.network.blocks[2].id, { x: 20, y: 20 });
    expect(Object.keys(instance.network.positions)).toHaveLength(2);

    instance.clearPositions();
    expect(instance.network.positions).toEqual({});

    instance.undo();
    expect(Object.keys(instance.network.positions)).toHaveLength(2);
  });

  it('records nothing when there is nothing to tidy', () => {
    const instance = store();
    instance.clearPositions();
    expect(instance.canUndo).toBe(false);
  });

  it('gives a dropped block the position it was dropped at', () => {
    const instance = store();
    const created = instance.addBlock('sigmoid', 2, { x: 30, y: 60 });
    expect(instance.network.positions[created]).toEqual({ x: 30, y: 60 });
    expect(instance.network.blocks[2].id).toBe(created);
  });

  it('leaves a clicked block unpositioned so it takes the auto slot', () => {
    const instance = store();
    const created = instance.addBlock('sigmoid');
    expect(instance.network.positions[created]).toBeUndefined();
  });

  it('drops a removed block position', () => {
    const instance = store();
    const id = instance.network.blocks[1].id;
    instance.setPosition(id, { x: 5, y: 5 });
    instance.removeBlock(id);
    expect(instance.network.positions).toEqual({});
  });
});

describe('initial network', () => {
  it('defaults to the MLP network', () => {
    const instance = new NetworkStore();
    expect(instance.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'linear',
      'relu',
      'linear',
      'softmax',
      'output'
    ]);
  });

  it('resets to the network it was constructed with', () => {
    const initial: Network = {
      version: 2,
      blocks: [
        { id: 'in', kind: 'input', shape: [4, 4, 1] },
        { id: 'conv', kind: 'conv2d', filters: 2, kernelSize: 2, stride: 1, padding: 'same' },
        { id: 'out', kind: 'output', shape: [3] }
      ],
      training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
      positions: {}
    };
    const instance = new NetworkStore(initial);

    instance.addBlock('relu', 1);
    expect(instance.network.blocks).toHaveLength(4);

    instance.reset();

    expect(instance.network.blocks.map((block) => block.kind)).toEqual([
      'input',
      'conv2d',
      'output'
    ]);
    expect(instance.canUndo).toBe(false);
  });
});
