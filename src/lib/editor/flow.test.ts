import { describe, expect, it } from 'vitest';
import { moveBlock } from '../network/chain';
import { createEmptyNetwork } from '../network/factory';
import { inferShapes } from '../network/inferShapes';
import type { Network } from '../network/types';
import {
  NODE_GAP,
  NODE_HEIGHT,
  NODE_WIDTH,
  autoPosition,
  connectionToIntent,
  isTidyLayout,
  materializePositions,
  nodeCentre,
  positionFor,
  shapeLabel,
  toFlow
} from './flow';

function net(): Network {
  return {
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

  it('carries a block colour into the node data', () => {
    const net = createEmptyNetwork();
    net.blocks[1] = { ...net.blocks[1], colour: 'blue' };
    const { nodes } = toFlow(net, inferShapes(net));
    expect(nodes[1].data.colour).toBe('blue');
    expect(nodes[0].data.colour).toBeUndefined();
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

  it('returns a copy of the stored position, not the stored object', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[2].id;
    const moved = { ...net, positions: { [id]: { x: 5, y: 6 } } };

    const position = positionFor(moved, 2);

    expect(position).not.toBe(moved.positions[id]);
    position.x = 99;
    expect(moved.positions[id].x).toBe(5);
  });

  it('returns a stored position that equals the auto slot', () => {
    const network = net();
    const id = network.blocks[2].id;
    const auto = autoPosition(2);
    const moved = { ...network, positions: { [id]: { ...auto } } };

    expect(positionFor(moved, 2)).toEqual(auto);
  });
});

describe('materializePositions', () => {
  it('fills missing positions with the auto slot and keeps stored ones', () => {
    const net = createEmptyNetwork();
    const moved = net.blocks[1].id;
    const result = materializePositions({ ...net, positions: { [moved]: { x: 5, y: 6 } } });

    expect(result.positions[moved]).toEqual({ x: 5, y: 6 });
    expect(result.positions[net.blocks[0].id]).toEqual(autoPosition(0));
    expect(Object.keys(result.positions)).toHaveLength(net.blocks.length);
  });

  it('keeps a stored position that equals the auto slot, as a fresh copy', () => {
    const net = createEmptyNetwork();
    const id = net.blocks[1].id;
    const auto = autoPosition(1);
    const input = { ...net, positions: { [id]: { ...auto } } };

    const result = materializePositions(input);

    expect(result.positions[id]).toEqual(auto);
    expect(result.positions[id]).not.toBe(input.positions[id]);
  });

  it('drops positions for block ids that are not in the network', () => {
    const net = createEmptyNetwork();
    const result = materializePositions({ ...net, positions: { ghost: { x: 1, y: 2 } } });

    expect(result.positions).not.toHaveProperty('ghost');
    expect(Object.keys(result.positions)).toHaveLength(net.blocks.length);
  });
});

describe('isTidyLayout', () => {
  it('is true for an empty or auto layout', () => {
    const net = createEmptyNetwork();
    expect(isTidyLayout(net)).toBe(true);
    expect(isTidyLayout(materializePositions(net))).toBe(true);
  });

  it('is false once a node is moved', () => {
    const net = createEmptyNetwork();
    const moved = { ...net, positions: { [net.blocks[1].id]: { x: 5, y: 6 } } };
    expect(isTidyLayout(moved)).toBe(false);
  });

  it('is true when a stored position equals the auto slot', () => {
    const net = createEmptyNetwork();
    const tidy = { ...net, positions: { [net.blocks[1].id]: autoPosition(1) } };
    expect(isTidyLayout(tidy)).toBe(true);
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
