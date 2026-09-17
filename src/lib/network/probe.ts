import { inferShapes } from './inferShapes';
import type { Network } from './types';

export interface ProbeTarget {
  source: 'input' | number;
  dims: number[];
}

export function probeTargetFor(net: Network, blockId: string | null): ProbeTarget | null {
  if (blockId === null) return null;
  const index = net.blocks.findIndex((block) => block.id === blockId);
  if (index === -1) return null;

  const { perBlock } = inferShapes(net);
  const block = net.blocks[index];

  if (block.kind === 'input') {
    return { source: 'input', dims: perBlock[index].outShape ?? [] };
  }

  const realBlocks = net.blocks.filter(
    (candidate) => candidate.kind !== 'input' && candidate.kind !== 'output'
  );
  if (realBlocks.length === 0) return null;

  const realIndex =
    block.kind === 'output'
      ? realBlocks.length - 1
      : realBlocks.findIndex((candidate) => candidate.id === block.id);
  if (realIndex === -1) return null;

  const realBlockIndex = net.blocks.findIndex(
    (candidate) => candidate.id === realBlocks[realIndex].id
  );
  return { source: realIndex, dims: perBlock[realBlockIndex].outShape ?? [] };
}
