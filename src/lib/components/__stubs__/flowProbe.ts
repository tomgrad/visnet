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
}
