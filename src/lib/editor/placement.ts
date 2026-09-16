import type { Network } from '../network/types';

const FIRST_INTERIOR = 1;

export function insertionIndexFor(net: Network, selectedBlockId: string | null): number {
  const last = net.blocks.length - 1;
  const index = net.blocks.findIndex((block) => block.id === selectedBlockId);
  if (index === -1) return last;
  return Math.min(index + 1, last);
}

export function dropIndexFor(
  flowX: number,
  blockCount: number,
  nodeWidth: number,
  gap: number
): number {
  if (blockCount < 4) return FIRST_INTERIOR;

  const last = blockCount - 1;
  const step = nodeWidth + gap;
  let index = FIRST_INTERIOR;
  for (let i = FIRST_INTERIOR; i < last; i++) {
    if (flowX > i * step - gap) index = i + 1;
  }
  return Math.min(Math.max(index, FIRST_INTERIOR), last);
}
