import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const REQUIRED_TOKENS = [
  '--color-bg',
  '--color-surface',
  '--color-border',
  '--color-text',
  '--color-text-muted',
  '--color-accent',
  '--color-error',
  '--color-warning',
  '--color-success',
  '--class-0',
  '--class-1',
  '--space-1',
  '--space-6',
  '--radius-sm',
  '--radius-md',
  '--font-sans',
  '--font-mono',
  '--text-xs',
  '--text-xl'
];

describe('design tokens', () => {
  const css = readFileSync(fileURLToPath(new URL('./tokens.css', import.meta.url)), 'utf8');

  it.each(REQUIRED_TOKENS)('defines %s', (token) => {
    expect(css).toContain(`${token}:`);
  });
});
