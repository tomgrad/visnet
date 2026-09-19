<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    title,
    intro,
    editor,
    training,
    focus,
    experiment
  }: {
    title: string;
    intro: string;
    editor: Snippet;
    training: Snippet;
    focus?: Snippet;
    experiment: Snippet;
  } = $props();
</script>

<main class="example">
  <header>
    <h1>{title}</h1>
    <p>{intro}</p>
  </header>

  <section class="editor">{@render editor()}</section>

  <div class="lower" class:has-focus={focus !== undefined}>
    {#if focus}
      <section class="focus">{@render focus()}</section>
    {/if}
    <section class="experiment">{@render experiment()}</section>
    <section class="training">{@render training()}</section>
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

  .lower {
    display: grid;
    gap: var(--space-4);
    grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
    align-items: start;
  }

  .lower.has-focus {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(280px, 340px);
  }

  .training {
    position: sticky;
    top: var(--space-4);
  }

  .experiment {
    display: grid;
    gap: var(--space-3);
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    align-items: start;
  }

  @media (max-width: 1400px) {
    .lower.has-focus {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }

    .lower.has-focus .training {
      grid-column: 1 / -1;
      position: static;
    }
  }

  @media (max-width: 700px) {
    .lower.has-focus {
      grid-template-columns: 1fr;
    }

    .lower.has-focus .training {
      grid-column: auto;
    }
  }

  @media (max-width: 1100px) {
    .lower {
      grid-template-columns: 1fr;
    }

    .training {
      position: static;
    }
  }
</style>
