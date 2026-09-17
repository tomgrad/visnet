import { describe, expect, it } from 'vitest';
import { moveBlock } from '../network/chain';
import { inferShapes } from '../network/inferShapes';
import type { Network } from '../network/types';
import {
  NODE_GAP,
  NODE_HEIGHT,
  NODE_WIDTH,
  autoPosition,
  connectionToIntent,
  nodeCentre,
  positionFor,
  shapeLabel,
  toFlow
} from './flow';

function net(): Network {
  return {
    version: 2,
    blocks: [
      { id: 'in', kind: 'input', shape: [2] },
      { id: 'a', kind: 'linear', units: 8 },
      { id: 'b', kind: 'relu' },
      { id: 'c', kind: 'linear', units: 2 },
      { id: 'out', kind: 'output', shape: [2] }
    ],
    training: { loss: 'crossEntropy', optimizer: 'adam', learningRate: 0.01, batchSize: 32 },
    positions: {}
  };
}

const ids = (network: Network) => network.blocks.map((b) => b.id);

describe('shapeLabel', () => {
  it('renders a one-dimensional shape', () => {
    expect(shapeLabel([2])).toBe('[2]');
  });

  it('renders a multi-dimensional shape', () => {
    expect(shapeLabel([28, 28, 4])).toBe('[28 × 28 × 4]');
  });

  it('renders nothing for an unknown shape', () => {
    expect(shapeLabel(null)).toBeNull();
  });
});

describe('toFlow', () => {
  const network = net();
  const flow = toFlow(network, inferShapes(network));

  it('creates one node per block, positioned top to bottom', () => {
    expect(flow.nodes.map((node) => node.id)).toEqual(ids(network));
    expect(flow.nodes.map((node) => node.position.y)).toEqual([
      0,
      NODE_HEIGHT + NODE_GAP,
      2 * (NODE_HEIGHT + NODE_GAP),
      3 * (NODE_HEIGHT + NODE_GAP),
      4 * (NODE_HEIGHT + NODE_GAP)
    ]);
    expect(flow.nodes.every((node) => node.position.x === 0)).toBe(true);
  });

  it('marks only the interior blocks as removable', () => {
    expect(flow.nodes.map((node) => node.data.removable)).toEqual([false, true, true, true, false]);
  });

  it('attaches shapes and parameter counts to nodes', () => {
    expect(flow.nodes[1].data).toMatchObject({
      kind: 'linear',
      inShape: [2],
      outShape: [8],
      paramCount: 24,
      index: 1
    });
  });

  it('creates one labelled edge per adjacent pair', () => {
    expect(flow.edges.map((edge) => edge.id)).toEqual(['in->a', 'a->b', 'b->c', 'c->out']);
    expect(flow.edges.map((edge) => edge.label)).toEqual(['[2]', '[8]', '[8]', '[2]']);
  });
});

describe('connectionToIntent', () => {
  it('moves a block later when dropped on a later block', () => {
    const intent = connectionToIntent({ source: 'a', target: 'c' }, net());
    expect(intent).toEqual({ type: 'move', from: 1, to: 2 });
    expect(ids(moveBlock(net(), 1, 2))).toEqual(['in', 'b', 'a', 'c', 'out']);
  });

  it('moves a block earlier when dropped on an earlier block', () => {
    const intent = connectionToIntent({ source: 'c', target: 'a' }, net());
    expect(intent).toEqual({ type: 'move', from: 3, to: 1 });
    expect(ids(moveBlock(net(), 3, 1))).toEqual(['in', 'c', 'a', 'b', 'out']);
  });

  it('lets a block be dropped on the output to become the last layer', () => {
    const intent = connectionToIntent({ source: 'a', target: 'out' }, net());
    expect(intent).toEqual({ type: 'move', from: 1, to: 3 });
    expect(ids(moveBlock(net(), 1, 3))).toEqual(['in', 'b', 'c', 'a', 'out']);
  });

  it('ignores a self connection', () => {
    expect(connectionToIntent({ source: 'a', target: 'a' }, net())).toBeNull();
  });

  it('ignores connections from the input or the output', () => {
    expect(connectionToIntent({ source: 'in', target: 'c' }, net())).toBeNull();
    expect(connectionToIntent({ source: 'out', target: 'a' }, net())).toBeNull();
  });

  it('ignores connections into the input', () => {
    expect(connectionToIntent({ source: 'c', target: 'in' }, net())).toBeNull();
  });

  it('ignores connections between already adjacent blocks', () => {
    expect(connectionToIntent({ source: 'a', target: 'b' }, net())).toBeNull();
  });

  it('ignores unknown ids', () => {
    expect(connectionToIntent({ source: 'a', target: 'missing' }, net())).toBeNull();
    expect(connectionToIntent({ source: 'missing', target: 'a' }, net())).toBeNull();
  });
});

describe('stored positions', () => {
  it('uses the stored position when there is one', () => {
    const network = net();
    const moved = { ...network, positions: { [network.blocks[1].id]: { x: 42, y: 99 } } };
    const flow = toFlow(moved, inferShapes(moved));
    expect(flow.nodes[1].position).toEqual({ x: 42, y: 99 });
  });

  it('falls back to the auto layout for blocks with no stored position', () => {
    const network = net();
    const moved = { ...network, positions: { [network.blocks[1].id]: { x: 42, y: 99 } } };
    const flow = toFlow(moved, inferShapes(moved));

    expect(flow.nodes[0].position).toEqual(autoPosition(0));
    expect(flow.nodes[2].position).toEqual(autoPosition(2));
    expect(flow.nodes[4].position).toEqual(autoPosition(4));
  });

  it('ignores a position for a block that is not in the network', () => {
    const network = net();
    const moved = { ...network, positions: { ghost: { x: 1, y: 2 } } };
    const flow = toFlow(moved, inferShapes(moved));
    expect(flow.nodes.map((node) => node.position)).toEqual([
      autoPosition(0),
      autoPosition(1),
      autoPosition(2),
      autoPosition(3),
      autoPosition(4)
    ]);
  });
});

describe('positionFor', () => {
  it('returns the stored position or the auto slot', () => {
    const network = net();
    const id = network.blocks[2].id;
    const moved = { ...network, positions: { [id]: { x: 7, y: 8 } } };
    expect(positionFor(moved, 2)).toEqual({ x: 7, y: 8 });
    expect(positionFor(moved, 1)).toEqual(autoPosition(1));
  });
});

describe('nodeCentre', () => {
  it('offsets by half the node size', () => {
    expect(nodeCentre({ x: 0, y: 0 })).toEqual({ x: NODE_WIDTH / 2, y: NODE_HEIGHT / 2 });
    expect(nodeCentre({ x: 10, y: 20 })).toEqual({
      x: 10 + NODE_WIDTH / 2,
      y: 20 + NODE_HEIGHT / 2
    });
  });
});
