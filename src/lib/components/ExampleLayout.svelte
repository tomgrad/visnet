<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    title,
    intro,
    editor,
    experiment
  }: {
    title: string;
    intro: string;
    editor: Snippet;
    experiment: Snippet;
  } = $props();
</script>

<main class="example">
  <header>
    <h1>{title}</h1>
    <p>{intro}</p>
  </header>

  <div class="columns">
    <section class="editor">{@render editor()}</section>
    <section class="experiment">{@render experiment()}</section>
  </div>
</main>

<style>
  .example {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-5);
    max-width: 1500px;
    margin: 0 auto;
  }

  header h1 {
    margin: 0 0 var(--space-2);
    font-size: var(--text-xl);
  }

  header p {
    margin: 0;
    color: var(--color-text-muted);
    max-width: 70ch;
  }

  .columns {
    display: grid;
    gap: var(--space-4);
    grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
    align-items: start;
  }

  @media (max-width: 1200px) {
    .columns {
      grid-template-columns: 1fr;
    }
  }

  .experiment {
    display: grid;
    gap: var(--space-3);
    position: sticky;
    top: var(--space-4);
  }
</style>
