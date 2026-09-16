import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit()],
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'engine',
          environment: 'node',
          include: ['src/**/*.test.ts', 'scripts/**/*.test.ts'],
          exclude: ['src/**/*.svelte.test.ts', 'src/lib/components/**/*.test.ts']
        }
      },
      {
        extends: true,
        resolve: { conditions: ['browser'] },
        test: {
          name: 'ui',
          environment: 'jsdom',
          setupFiles: ['@testing-library/svelte/vitest'],
          include: ['src/**/*.svelte.test.ts', 'src/lib/components/**/*.test.ts']
        }
      }
    ]
  }
});
