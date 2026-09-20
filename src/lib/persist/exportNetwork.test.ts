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

  it('includes a set colour and omits an unset one', () => {
    const net = createEmptyNetwork();
    const blocks = net.blocks.map((block, index) =>
      index === 1 ? { ...block, colour: 'blue' as const } : block
    );
    const module = toTypeScriptModule({ ...net, blocks });
    expect(module).toContain("colour: 'blue'");
    expect(module.split("colour: 'blue'").length - 1).toBe(1);
  });

  it('omits a colour that was explicitly cleared', () => {
    const net = createEmptyNetwork();
    const blocks = net.blocks.map((block, index) =>
      index === 1 ? { ...block, colour: undefined } : block
    );
    const module = toTypeScriptModule({ ...net, blocks });
    expect(module).not.toContain('colour');
  });

  it('exports a batchnorm block as a plain literal', () => {
    const net = createEmptyNetwork();
    const block = { id: 'bn', kind: 'batchnorm' as const };
    const module = toTypeScriptModule({ ...net, blocks: [...net.blocks, block] });
    expect(module).toContain("{ id: 'bn', kind: 'batchnorm' }");
  });
});
