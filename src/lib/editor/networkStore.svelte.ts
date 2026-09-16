import { insertAt, moveBlock, removeBlock, replaceBlock } from '../network/chain';
import { clampBlockPatch, clampNetwork } from '../network/constraints';
import { createBlock, createEmptyNetwork } from '../network/factory';
import { inferShapes } from '../network/inferShapes';
import type { Block, BlockKind, Network, TrainingConfig } from '../network/types';
import { validate } from '../network/validate';
import { History } from './history';
import { insertionIndexFor } from './placement';

const HISTORY_LIMIT = 50;

export class NetworkStore {
  network = $state<Network>(createEmptyNetwork());
  selectedBlockId = $state<string | null>(null);
  expectedClasses = $state<number | undefined>(undefined);
  announcements = $state<string[]>([]);
  canUndo = $state(false);
  canRedo = $state(false);

  #history = new History<Network>(HISTORY_LIMIT);

  issues = $derived(validate(this.network, { expectedClasses: this.expectedClasses }));
  errors = $derived(this.issues.filter((issue) => issue.severity === 'error'));
  warnings = $derived(this.issues.filter((issue) => issue.severity === 'warning'));
  shapes = $derived(inferShapes(this.network));
  isValid = $derived(this.errors.length === 0);
  paramCount = $derived(this.shapes.totalParamCount);
  selectedBlock = $derived(
    this.network.blocks.find((block) => block.id === this.selectedBlockId) ?? null
  );

  select(id: string | null): void {
    this.selectedBlockId = id;
  }

  addBlock(kind: BlockKind, index?: number): string {
    const block = createBlock(kind);
    const target = index ?? insertionIndexFor(this.network, this.selectedBlockId);
    this.#commit(insertAt(this.network, target, block));
    this.selectedBlockId = block.id;
    return block.id;
  }

  removeBlock(id: string): void {
    const next = removeBlock(this.network, id);
    if (next === this.network) return;
    this.#commit(next);
    if (this.selectedBlockId === id) this.selectedBlockId = null;
  }

  moveBlock(from: number, to: number): void {
    this.#commit(moveBlock(this.network, from, to));
  }

  moveSelectedBy(offset: number): void {
    const index = this.network.blocks.findIndex((block) => block.id === this.selectedBlockId);
    if (index === -1) return;
    this.moveBlock(index, index + offset);
  }

  updateBlock(id: string, patch: Partial<Block>): void {
    const { patch: clamped, announcement } = clampBlockPatch(this.network, id, patch);
    const result = clampNetwork(replaceBlock(this.network, id, clamped));
    this.#commit(result.network);
    const messages = announcement ? [announcement, ...result.announcements] : result.announcements;
    if (messages.length > 0) this.announcements = [...this.announcements, ...messages];
  }

  updateTraining(patch: Partial<TrainingConfig>): void {
    this.#commit({ ...this.network, training: { ...this.network.training, ...patch } });
  }

  undo(): void {
    const previous = this.#history.undo(this.network);
    if (previous === null) return;
    this.network = previous;
    this.#syncHistoryFlags();
  }

  redo(): void {
    const next = this.#history.redo(this.network);
    if (next === null) return;
    this.network = next;
    this.#syncHistoryFlags();
  }

  load(net: Network): void {
    this.#history.clear();
    this.network = net;
    this.selectedBlockId = null;
    this.announcements = [];
    this.#syncHistoryFlags();
  }

  reset(): void {
    this.load(createEmptyNetwork());
  }

  announce(message: string): void {
    this.announcements = [...this.announcements, message];
  }

  dismissAnnouncements(): void {
    this.announcements = [];
  }

  #commit(next: Network): void {
    if (next === this.network) return;
    this.#history.push(this.network);
    this.network = next;
    this.#syncHistoryFlags();
  }

  #syncHistoryFlags(): void {
    this.canUndo = this.#history.canUndo;
    this.canRedo = this.#history.canRedo;
  }
}
