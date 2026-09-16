<script lang="ts">
  import { parameterBounds } from '../network/constraints';
  import { BLOCK_DESCRIPTIONS, PARAM_DESCRIPTIONS } from '../network/descriptions';
  import { shapeLabel } from '../editor/flow';
  import type { NetworkStore } from '../editor/networkStore.svelte';
  import type { Block, InputBlock } from '../network/types';

  let { store }: { store: NetworkStore } = $props();

  const block = $derived(store.selectedBlock);
  const index = $derived(
    store.network.blocks.findIndex((candidate) => candidate.id === store.selectedBlockId)
  );
  const info = $derived(index === -1 ? null : store.shapes.perBlock[index]);
  const previousKind = $derived(index > 0 ? store.network.blocks[index - 1].kind : null);
  const inShape = $derived(info?.inShape ?? null);
  const bounds = $derived(parameterBounds(inShape));
  const kernelChoices = $derived(
    bounds && block?.kind === 'conv2d' ? bounds.kernelSize : block?.kind === 'conv2d' ? [block.kernelSize] : []
  );
  const strideChoices = $derived(
    bounds && block?.kind === 'conv2d' ? bounds.stride : block?.kind === 'conv2d' ? [block.stride] : []
  );
  const canMove = $derived(block !== null && block.kind !== 'input' && block.kind !== 'output');
  let shapeError = $state<string | null>(null);

  function patch(next: Partial<Block>): void {
    if (!block) return;
    store.updateBlock(block.id, next);
  }

  function commitShape(event: Event): void {
    if (!block || block.kind !== 'input') return;
    const raw = (event.currentTarget as HTMLInputElement).value;
    const parsed = raw
      .split(',')
      .map((part) => Number(part.trim()))
      .filter((value) => Number.isFinite(value) && value > 0)
      .map((value) => Math.floor(value));

    if (parsed.length === 0) {
      shapeError = 'Enter at least one positive number, separated by commas.';
      return;
    }
    shapeError = null;
    patch({ shape: parsed } as Partial<InputBlock>);
  }

  function commitNumber(event: Event, key: 'units' | 'filters'): void {
    const raw = (event.currentTarget as HTMLInputElement).value;
    if (raw.trim() === '') return;
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    patch({ [key]: value } as Partial<Block>);
  }

  function outputSizeFor(padding: 'same' | 'valid'): string {
    if (!inShape || inShape.length !== 3 || !block || block.kind !== 'conv2d') return '';
    const [height, width] = inShape;
    const { kernelSize, stride } = block;
    const compute = (dimension: number): number =>
      padding === 'same'
        ? Math.ceil(dimension / stride)
        : Math.floor((dimension - kernelSize) / stride) + 1;
    const result = `${compute(height)}×${compute(width)}`;
    if (padding === 'same' && result === `${height}×${width}`) return `stays ${result}`;
    return `becomes ${result}`;
  }
</script>

{#if !block}
  <div class="inspector" data-testid="inspector-empty">
    <p>Select a block to see and change its settings.</p>
  </div>
{:else}
  <div class="inspector" data-testid="inspector">
    <h2>{block.kind}</h2>
    <p class="description">{BLOCK_DESCRIPTIONS[block.kind]}</p>

    {#if block.kind !== 'input'}
      <p class="incoming" data-testid="inspector-incoming">
        Input {shapeLabel(inShape) ?? '—'}{previousKind ? ` from ${previousKind}` : ''}
      </p>
    {/if}

    {#if block.kind === 'input'}
      <label>
        <span>Shape</span>
        <input
          type="text"
          data-testid="param-shape"
          title={PARAM_DESCRIPTIONS.inputShape}
          value={block.shape.join(', ')}
          onchange={commitShape}
        />
        <small>{PARAM_DESCRIPTIONS.inputShape}</small>
      </label>
      {#if shapeError}<p class="error">{shapeError}</p>{/if}
    {/if}

    {#if block.kind === 'linear' || block.kind === 'output'}
      <label>
        <span>Units</span>
        <input
          type="number"
          min="1"
          data-testid="param-units"
          title={PARAM_DESCRIPTIONS.units}
          value={block.units}
          onchange={(event) => commitNumber(event, 'units')}
        />
        <small>{PARAM_DESCRIPTIONS.units}</small>
      </label>
    {/if}

    {#if block.kind === 'conv2d'}
      <label>
        <span>Filters</span>
        <input
          type="number"
          min="1"
          data-testid="param-filters"
          title={PARAM_DESCRIPTIONS.filters}
          value={block.filters}
          onchange={(event) => commitNumber(event, 'filters')}
        />
        <small>{PARAM_DESCRIPTIONS.filters}</small>
      </label>

      <label>
        <span>Kernel size</span>
        <select
          data-testid="param-kernel-size"
          title={PARAM_DESCRIPTIONS.kernelSize}
          value={String(block.kernelSize)}
          onchange={(event) => patch({ kernelSize: Number(event.currentTarget.value) } as Partial<Block>)}
        >
          {#each kernelChoices as choice (choice)}
            <option value={String(choice)}>{choice}×{choice}</option>
          {/each}
        </select>
        <small>{PARAM_DESCRIPTIONS.kernelSize}</small>
      </label>

      <label>
        <span>Stride</span>
        <select
          data-testid="param-stride"
          title={PARAM_DESCRIPTIONS.stride}
          value={String(block.stride)}
          onchange={(event) => patch({ stride: Number(event.currentTarget.value) } as Partial<Block>)}
        >
          {#each strideChoices as choice (choice)}
            <option value={String(choice)}>{choice}</option>
          {/each}
        </select>
        <small>{PARAM_DESCRIPTIONS.stride}</small>
      </label>

      <label>
        <span>Padding</span>
        <select
          data-testid="param-padding"
          title={PARAM_DESCRIPTIONS.padding}
          value={block.padding}
          onchange={(event) => patch({ padding: event.currentTarget.value as 'same' | 'valid' } as Partial<Block>)}
        >
          <option value="same">same — {outputSizeFor('same')}</option>
          <option value="valid">valid — {outputSizeFor('valid')}</option>
        </select>
        <small>{PARAM_DESCRIPTIONS.padding}</small>
      </label>
    {/if}

    <dl class="facts">
      <dt>Output</dt>
      <dd>{shapeLabel(info?.outShape ?? null) ?? '—'}</dd>
      <dt>Parameters</dt>
      <dd>{info?.paramCount ?? 0}</dd>
    </dl>

    <div class="moves">
      <button
        type="button"
        data-testid="move-left"
        disabled={!canMove}
        onclick={() => store.moveSelectedBy(-1)}
      >
        Move left
      </button>
      <button
        type="button"
        data-testid="move-right"
        disabled={!canMove}
        onclick={() => store.moveSelectedBy(1)}
      >
        Move right
      </button>
    </div>
  </div>
{/if}

<style>
  .inspector {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-3);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }

  h2 {
    margin: 0;
    font-size: var(--text-base);
  }

  .description,
  .incoming {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }

  label {
    display: grid;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }

  input,
  select {
    font: inherit;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  small {
    color: var(--color-text-muted);
    font-size: var(--text-xs);
  }

  .error {
    margin: 0;
    color: var(--color-error);
    font-size: var(--text-xs);
  }

  .facts {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-1) var(--space-2);
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }

  .facts dt {
    color: var(--color-text-muted);
  }

  .facts dd {
    margin: 0;
  }

  .moves {
    display: flex;
    gap: var(--space-2);
  }

  .moves button {
    flex: 1;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    cursor: pointer;
  }

  .moves button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
