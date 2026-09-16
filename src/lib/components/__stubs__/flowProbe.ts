let latestNodes: unknown[] = [];
let latestEdges: unknown[] = [];

export function recordNodes(nodes: unknown[]): void {
  latestNodes = nodes;
}

export function recordEdges(edges: unknown[]): void {
  latestEdges = edges;
}

export function capturedNodes(): unknown[] {
  return latestNodes;
}

export function capturedEdges(): unknown[] {
  return latestEdges;
}

export function resetCaptured(): void {
  latestNodes = [];
  latestEdges = [];
  handlers = {};
}

let handlers: Record<string, unknown> = {};

export function recordHandlers(next: Record<string, unknown>): void {
  handlers = next;
}

export function capturedHandler(name: string): ((argument: unknown) => void) | undefined {
  return handlers[name] as ((argument: unknown) => void) | undefined;
}
