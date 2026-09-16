export class History<T> {
  private past: T[] = [];
  private future: T[] = [];

  constructor(private readonly limit: number = 50) {}

  push(previous: T): void {
    this.past.push(previous);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
  }

  undo(current: T): T | null {
    const previous = this.past.pop();
    if (previous === undefined) return null;
    this.future.push(current);
    return previous;
  }

  redo(current: T): T | null {
    const next = this.future.pop();
    if (next === undefined) return null;
    this.past.push(current);
    return next;
  }

  clear(): void {
    this.past = [];
    this.future = [];
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }
}
