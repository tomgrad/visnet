<script lang="ts">
  import { resolve } from '$app/paths';
</script>

<main>
  <h1>VisNet</h1>
  <p>Build neural networks from visual blocks and watch them learn.</p>

  <ul>
    <li>
      <a href={resolve('/examples/mlp')}>Points in 2D</a>
      <span>Classify coloured points and see the decision boundary.</span>
    </li>
    <li>
      <a href={resolve('/examples/cnn')}>Handwritten digits</a>
      <span>Coming next: convolutional networks on digit images.</span>
    </li>
  </ul>
</main>

<style>
  main {
    max-width: 70ch;
    margin: 0 auto;
    padding: var(--space-5);
  }

  h1 {
    font-size: var(--text-xl);
  }

  p {
    color: var(--color-text-muted);
  }

  ul {
    list-style: none;
    padding: 0;
    display: grid;
    gap: var(--space-3);
  }

  li {
    display: grid;
    gap: var(--space-1);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  a {
    color: var(--color-accent);
    font-weight: 600;
  }

  li span {
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
</style>
