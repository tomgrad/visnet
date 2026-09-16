<script lang="ts">
  import { resolve } from '$app/paths';
</script>

<main>
  <h1>Handwritten digits (coming next)</h1>
  <p>
    This example will train a small convolutional network on MNIST-style digit images. The engine
    already supports convolution, flatten, and the shape inference the editor needs; what is missing
    is this page.
  </p>
  <p><a href={resolve('/examples/mlp')}>Try the 2D points example instead</a></p>
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
</style>
