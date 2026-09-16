import type { ShapeResult } from '../network/inferShapes';
import type { BlockKind, Network } from '../network/types';

export const NODE_WIDTH = 200;
export const NODE_HEIGHT = 90;
export const NODE_GAP = 80;

export interface FlowNode {
  id: string;
  position: { x: number; y: number };
  data: {
    kind: BlockKind;
    inShape: number[] | null;
    outShape: number[] | null;
    paramCount: number | null;
    index: number;
    removable: boolean;
  };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  label: string | null;
}

export interface ChainMove {
  type: 'move';
  from: number;
  to: number;
}

export function shapeLabel(shape: number[] | null): string | null {
  return shape ? `[${shape.join(' × ')}]` : null;
}

export function toFlow(
  net: Network,
  shapes: ShapeResult
): { nodes: FlowNode[]; edges: FlowEdge[] } {
  const nodes: FlowNode[] = net.blocks.map((block, index) => {
    const info = shapes.perBlock[index];
    return {
      id: block.id,
      position: { x: 0, y: index * (NODE_HEIGHT + NODE_GAP) },
      data: {
        kind: block.kind,
        inShape: info.inShape,
        outShape: info.outShape,
        paramCount: info.paramCount,
        index,
        removable: block.kind !== 'input' && block.kind !== 'output'
      }
    };
  });

  const edges: FlowEdge[] = shapes.edges.map((edge) => ({
    id: `${edge.fromId}->${edge.toId}`,
    source: edge.fromId,
    target: edge.toId,
    label: shapeLabel(edge.shape)
  }));

  return { nodes, edges };
}

export function connectionToIntent(
  connection: { source: string; target: string },
  net: Network
): ChainMove | null {
  const from = net.blocks.findIndex((block) => block.id === connection.source);
  const to = net.blocks.findIndex((block) => block.id === connection.target);

  if (from === -1 || to === -1) return null;
  if (from === 0 || from === net.blocks.length - 1) return null;
  if (to === 0) return null;
  if (to === from + 1) return null;

  const destination = from < to ? to - 1 : to;
  if (destination === from) return null;

  return { type: 'move', from, to: destination };
}
