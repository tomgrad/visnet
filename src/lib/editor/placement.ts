import type { Network } from '../network/types';

const FIRST_INTERIOR = 1;

export function insertionIndexFor(net: Network, selectedBlockId: string | null): number {
  const last = net.blocks.length - 1;
  const index = net.blocks.findIndex((block) => block.id === selectedBlockId);
  if (index === -1) return last;
  return Math.min(index + 1, last);
}

export function dropIndexFor(
  flowY: number,
  blockCount: number,
  nodeHeight: number,
  gap: number
): number {
  const last = blockCount - 1;
  if (last < FIRST_INTERIOR) return FIRST_INTERIOR;

  const step = nodeHeight + gap;
  let index = FIRST_INTERIOR;
  for (let i = FIRST_INTERIOR; i < last; i++) {
    const centre = i * step + nodeHeight / 2;
    if (flowY >= centre) index = i + 1;
  }
  return Math.min(Math.max(index, FIRST_INTERIOR), last);
}
