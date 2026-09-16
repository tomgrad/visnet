import type { Block, Network } from './types';

const FIRST_INTERIOR = 1;

export function insertAt(net: Network, index: number, block: Block): Network {
  const last = net.blocks.length - 1;
  const target = Math.min(Math.max(index, FIRST_INTERIOR), last);
  const blocks = [...net.blocks];
  blocks.splice(target, 0, block);
  return { ...net, blocks };
}

export function moveBlock(net: Network, fromIndex: number, toIndex: number): Network {
  const last = net.blocks.length - 2;
  if (fromIndex < FIRST_INTERIOR || fromIndex > last) return net;
  const target = Math.min(Math.max(toIndex, FIRST_INTERIOR), last);
  if (target === fromIndex) return net;

  const blocks = [...net.blocks];
  const [block] = blocks.splice(fromIndex, 1);
  blocks.splice(target, 0, block);
  return { ...net, blocks };
}

export function removeBlock(net: Network, id: string): Network {
  const index = net.blocks.findIndex((block) => block.id === id);
  if (index < FIRST_INTERIOR || index > net.blocks.length - 2) return net;
  const positions = { ...net.positions };
  delete positions[id];
  return {
    ...net,
    blocks: net.blocks.filter((block) => block.id !== id),
    positions
  };
}

export function replaceBlock(net: Network, id: string, patch: Partial<Block>): Network {
  let changed = false;
  const blocks = net.blocks.map((block) => {
    if (block.id !== id) return block;
    changed = true;
    return { ...block, ...patch } as Block;
  });
  return changed ? { ...net, blocks } : net;
}
