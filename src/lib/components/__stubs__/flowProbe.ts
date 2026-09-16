let captured: unknown[] = [];

export function recordNodes(nodes: unknown[]): void {
  captured = nodes;
}

export function capturedNodes(): unknown[] {
  return captured;
}

export function resetCapturedNodes(): void {
  captured = [];
}
