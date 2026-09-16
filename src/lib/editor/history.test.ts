import { describe, expect, it } from 'vitest';
import { History } from './history';

describe('History', () => {
  it('starts empty', () => {
    const history = new History<string>();
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
    expect(history.undo('now')).toBeNull();
    expect(history.redo('now')).toBeNull();
  });

  it('undoes back through pushed states', () => {
    const history = new History<string>();
    history.push('one');
    history.push('two');

    expect(history.canUndo).toBe(true);
    expect(history.undo('three')).toBe('two');
    expect(history.undo('two')).toBe('one');
    expect(history.undo('one')).toBeNull();
    expect(history.canUndo).toBe(false);
  });

  it('redoes what was undone', () => {
    const history = new History<string>();
    history.push('one');
    const undone = history.undo('two');

    expect(undone).toBe('one');
    expect(history.canRedo).toBe(true);
    expect(history.redo('one')).toBe('two');
    expect(history.canRedo).toBe(false);
  });

  it('drops the redo stack when a new change is pushed', () => {
    const history = new History<string>();
    history.push('one');
    history.undo('two');
    history.push('one-bis');
    expect(history.canRedo).toBe(false);
    expect(history.redo('one-bis')).toBeNull();
  });

  it('keeps at most `limit` entries', () => {
    const history = new History<number>(2);
    history.push(1);
    history.push(2);
    history.push(3);
    expect(history.undo(4)).toBe(3);
    expect(history.undo(3)).toBe(2);
    expect(history.undo(2)).toBeNull();
  });

  it('clears both stacks', () => {
    const history = new History<string>();
    history.push('one');
    history.undo('two');
    history.clear();
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
  });
});
