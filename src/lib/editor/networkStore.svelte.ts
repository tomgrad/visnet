import { insertAt, moveBlock, removeBlock, replaceBlock } from '../network/chain';
import { clampBlockPatch, clampNetwork } from '../network/constraints';
import { cloneNetwork, createBlock, createEmptyNetwork } from '../network/factory';
import { inferShapes } from '../network/inferShapes';
import type { Block, BlockKind, Network, NodePosition, TrainingConfig } from '../network/types';
import { validate } from '../network/validate';
import { positionFor } from './flow';
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
  #initial: Network;

  issues = $derived(validate(this.network, { expectedClasses: this.expectedClasses }));
  errors = $derived(this.issues.filter((issue) => issue.severity === 'error'));
  warnings = $derived(this.issues.filter((issue) => issue.severity === 'warning'));
  shapes = $derived(inferShapes(this.network));
  isValid = $derived(this.errors.length === 0);
  paramCount = $derived(this.shapes.totalParamCount);
  selectedBlock = $derived(
    this.network.blocks.find((block) => block.id === this.selectedBlockId) ?? null
  );

  constructor(initial: Network = createEmptyNetwork()) {
    this.network = initial;
    this.#initial = cloneNetwork(initial);
  }

  select(id: string | null): void {
    this.selectedBlockId = id;
  }

  addBlock(kind: BlockKind, index?: number, position?: NodePosition): string {
    const block = createBlock(kind);
    const target = index ?? insertionIndexFor(this.network, this.selectedBlockId);
    const inserted = insertAt(this.network, target, block);
    this.#commitClamped(
      position ? { ...inserted, positions: { ...inserted.positions, [block.id]: position } } : inserted
    );
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
    this.#commitClamped(moveBlock(this.network, from, to));
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

  setPosition(id: string, position: NodePosition): void {
    const index = this.network.blocks.findIndex((block) => block.id === id);
    if (index === -1) return;
    const current = positionFor(this.network, index);
    if (current.x === position.x && current.y === position.y) return;
    this.#commit({ ...this.network, positions: { ...this.network.positions, [id]: position } });
  }

  clearPositions(): void {
    if (Object.keys(this.network.positions).length === 0) return;
    this.#commit({ ...this.network, positions: {} });
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
    const result = clampNetwork(net);
    this.#history.clear();
    this.network = result.network;
    this.selectedBlockId = null;
    this.announcements = [...result.announcements];
    this.#syncHistoryFlags();
  }

  reset(): void {
    this.load(this.#initial);
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

  #commitClamped(next: Network): void {
    const result = clampNetwork(next);
    this.#commit(result.network);
    if (result.announcements.length > 0) {
      this.announcements = [...this.announcements, ...result.announcements];
    }
  }

  #syncHistoryFlags(): void {
    this.canUndo = this.#history.canUndo;
    this.canRedo = this.#history.canRedo;
  }
}
