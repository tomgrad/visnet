import type { Network } from '../network/types';

export function toTypeScriptModule(net: Network): string {
  const body = JSON.stringify(
    { blocks: net.blocks, training: net.training, positions: net.positions },
    null,
    2
  );
  return `import type { Network } from '$lib/network/types';\n\nexport const NETWORK = ${body} satisfies Network;\n`;
}
