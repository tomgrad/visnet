<script lang="ts">
  import type { NetworkStore } from '../editor/networkStore.svelte';

  let { store }: { store: NetworkStore } = $props();
</script>

<div class="issues" data-testid="issues-panel">
  <h2>Problems</h2>

  {#if store.issues.length === 0}
    <p class="ok" data-testid="issues-empty">This network looks good.</p>
  {:else}
    <ul>
      {#each store.issues as issue, index (index)}
        <li
          class="issue"
          class:error={issue.severity === 'error'}
          class:warning={issue.severity === 'warning'}
        >
          <button
            type="button"
            onclick={() => issue.blockId && store.select(issue.blockId)}
            disabled={!issue.blockId}
            data-testid="issue"
          >
            <span class="severity">{issue.severity === 'error' ? 'Blocks training' : 'Heads up'}</span>
            <span class="title">{issue.title}</span>
            <span class="message">{issue.message}</span>
            <span class="fix" data-testid="issue-fix">{issue.fix}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .issues {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .ok {
    margin: 0;
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }

  button {
    width: 100%;
    display: grid;
    gap: var(--space-1);
    text-align: left;
    padding: var(--space-2) var(--space-3);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-left-width: 3px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font: inherit;
  }

  button:disabled {
    cursor: default;
  }

  .error button {
    border-left-color: var(--color-error);
  }

  .warning button {
    border-left-color: var(--color-warning);
  }

  .severity {
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  .title {
    font-weight: 600;
    font-size: var(--text-sm);
  }

  .message {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  .fix {
    font-size: var(--text-sm);
    color: var(--color-text);
  }
</style>
