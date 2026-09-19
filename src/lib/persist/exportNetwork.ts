import type { Block, Network } from '../network/types';

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

function quote(value: string): string {
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

function key(name: string): string {
  return IDENTIFIER.test(name) ? name : quote(name);
}

function scalar(value: unknown): string {
  if (typeof value === 'string') return quote(value);
  if (Array.isArray(value)) return `[${value.map(scalar).join(', ')}]`;
  return String(value);
}

function inlineObject(entries: [string, unknown][]): string {
  const fields = entries.map(([name, value]) => `${key(name)}: ${scalar(value)}`);
  return `{ ${fields.join(', ')} }`;
}

function blockLiteral(block: Block, indent: string): string {
  const entries = Object.entries(block) as [string, unknown][];
  const inline = inlineObject(entries);
  if (indent.length + inline.length + 1 <= 100) return inline;
  const fields = entries.map(([name, value]) => `${indent}  ${key(name)}: ${scalar(value)}`);
  return `{\n${fields.join(',\n')}\n${indent}}`;
}

function positionsLiteral(positions: Network['positions']): string {
  const entries = Object.entries(positions);
  if (entries.length === 0) return '{}';
  const fields = entries.map(([id, point]) => `    ${key(id)}: { x: ${point.x}, y: ${point.y} }`);
  return `{\n${fields.join(',\n')}\n  }`;
}

export function toTypeScriptModule(net: Network): string {
  const blocks = net.blocks.map((block) => `    ${blockLiteral(block, '    ')}`).join(',\n');
  const training = inlineObject(Object.entries(net.training) as [string, unknown][]);
  return [
    "import type { Network } from '$lib/network/types';",
    '',
    'export const NETWORK = {',
    '  blocks: [',
    blocks,
    '  ],',
    `  training: ${training},`,
    `  positions: ${positionsLiteral(net.positions)}`,
    '} satisfies Network;',
    ''
  ].join('\n');
}
