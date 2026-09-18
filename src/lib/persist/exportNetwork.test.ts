import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import { toTypeScriptModule } from './exportNetwork';

const HEADER = "import type { Network } from '$lib/network/types';";
const DECLARATION = 'export const NETWORK = ';

describe('toTypeScriptModule', () => {
  it('wraps the network in an importable TS module', () => {
    const module = toTypeScriptModule(createEmptyNetwork());
    expect(module).toContain(HEADER);
    expect(module).toContain(DECLARATION);
    expect(module.trimEnd().endsWith('satisfies Network;')).toBe(true);
  });

  it('embeds the network as a valid object literal', () => {
    const net = createEmptyNetwork();
    const module = toTypeScriptModule(net);
    const start = module.indexOf(DECLARATION) + DECLARATION.length;
    const json = module.slice(start, module.lastIndexOf('}') + 1);
    expect(JSON.parse(json)).toEqual({
      blocks: net.blocks,
      training: net.training,
      positions: net.positions
    });
  });
});
