import type { Network, NodePosition } from '../network/types';

const FIRST_INTERIOR = 1;

export function insertionIndexFor(net: Network, selectedBlockId: string | null): number {
  const last = net.blocks.length - 1;
  const index = net.blocks.findIndex((block) => block.id === selectedBlockId);
  if (index === -1) return last;
  return Math.min(index + 1, last);
}

function midpoint(a: NodePosition, b: NodePosition): NodePosition {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function distanceSquared(a: NodePosition, b: NodePosition): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

export function dropIndexFor(point: NodePosition, centres: NodePosition[]): number {
  const last = centres.length - 1;
  if (last < FIRST_INTERIOR) return FIRST_INTERIOR;

  let nearest = FIRST_INTERIOR;
  let best = Number.POSITIVE_INFINITY;

  for (let edge = 0; edge < centres.length - 1; edge++) {
    const distance = distanceSquared(point, midpoint(centres[edge], centres[edge + 1]));
    if (distance < best) {
      best = distance;
      nearest = edge + 1;
    }
  }

  return Math.min(Math.max(nearest, FIRST_INTERIOR), last);
}
