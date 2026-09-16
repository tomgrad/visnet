import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import prettier from 'eslint-config-prettier';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['build/', '.svelte-kit/', 'node_modules/', 'coverage/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...svelte.configs['flat/recommended'],
  prettier,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } }
  }
);
