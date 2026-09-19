import { describe, expect, it } from 'vitest';
import { createEmptyNetwork } from '../network/factory';
import type { Network } from '../network/types';
import { toTypeScriptModule } from './exportNetwork';

const HEADER = "import type { Network } from '$lib/network/types';";

describe('toTypeScriptModule', () => {
  it('wraps the network in an importable TS module', () => {
    const module = toTypeScriptModule(createEmptyNetwork());
    expect(module).toContain(HEADER);
    expect(module).toContain('export const NETWORK = {');
    expect(module.trimEnd().endsWith('satisfies Network;')).toBe(true);
  });

  it('uses unquoted keys and single quotes', () => {
    const net = createEmptyNetwork();
    const module = toTypeScriptModule(net);
    expect(module).toContain('  blocks: [');
    expect(module).toContain("kind: 'input'");
    expect(module).toContain(`id: '${net.blocks[0].id}'`);
    expect(module).not.toContain('"blocks"');
  });

  it('keeps short blocks on one line', () => {
    const module = toTypeScriptModule(createEmptyNetwork());
    expect(module).toContain("    { id: '");
    expect(module).toContain("kind: 'input', shape: [2] }");
  });

  it('wraps a block whose one-line form would exceed the print width', () => {
    const net: Network = {
      blocks: [
        { id: 'input', kind: 'input', shape: [28, 28, 1] },
        {
          id: 'conv-decode-first-layer',
          kind: 'conv2dtranspose',
          filters: 16,
          kernelSize: 3,
          stride: 2,
          padding: 'same'
        },
        { id: 'output', kind: 'output', shape: [28, 28, 1] }
      ],
      training: { loss: 'mse', optimizer: 'adam', learningRate: 0.001, batchSize: 32 },
      positions: {}
    };
    const module = toTypeScriptModule(net);
    expect(module).toContain("      id: 'conv-decode-first-layer'");
    expect(module).toContain('    }');
  });
});
