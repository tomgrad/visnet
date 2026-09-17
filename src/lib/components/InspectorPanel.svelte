<script lang="ts">
  import { BLOCK_DESCRIPTIONS, PARAM_DESCRIPTIONS } from '../network/descriptions';
  import { spatialOutputSize } from '../network/inferShapes';
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
  const limit = $derived(inShape && inShape.length === 3 ? Math.min(inShape[0], inShape[1]) : null);
  const spatialSize = $derived(
    block?.kind === 'conv2d'
      ? block.kernelSize
      : block?.kind === 'maxpool2d'
        ? block.poolSize
        : null
  );
  const spatialStride = $derived(
    block?.kind === 'conv2d' || block?.kind === 'maxpool2d' ? block.stride : null
  );
  const paddingValue = $derived(
    block?.kind === 'conv2d' || block?.kind === 'maxpool2d' ? block.padding : 'valid'
  );
  function choicesFor(current: number | null, highestAllowed: number | null): number[] {
    if (current === null) return [];
    if (highestAllowed === null || !Number.isFinite(highestAllowed) || highestAllowed < 1) {
      return [current];
    }
    const highest = Math.max(Math.floor(highestAllowed), Math.ceil(current));
    return Array.from({ length: highest }, (_, index) => index + 1);
  }

  const sizeChoices = $derived(choicesFor(spatialSize, limit));
  const strideChoices = $derived(choicesFor(spatialStride, limit));
  const canMove = $derived(block !== null && block.kind !== 'input' && block.kind !== 'output');
  let paramError = $state<string | null>(null);

  function patch(next: Partial<Block>): void {
    if (!block) return;
    store.updateBlock(block.id, next);
  }

  function currentNumber(key: 'units' | 'filters' | 'size'): number | null {
    if (!block) return null;
    if (key === 'units' && block.kind === 'linear') return block.units;
    if (key === 'filters' && block.kind === 'conv2d') return block.filters;
    if (key === 'size' && block.kind === 'upsampling2d') return block.size;
    return null;
  }

  function commitShape(event: Event): void {
    if (!block || block.kind !== 'input') return;
    const input = event.currentTarget as HTMLInputElement;
    const parts = input.value.split(',').map((part) => part.trim());
    const parsed = parts.map((part) => Number(part));
    const valid =
      parts.length > 0 &&
      parts.every((part) => part !== '') &&
      parsed.every((value) => Number.isInteger(value) && value > 0);

    if (!valid) {
      input.value = block.shape.join(', ');
      paramError = 'Enter positive whole numbers, separated by commas.';
      return;
    }
    paramError = null;
    patch({ shape: parsed } as Partial<InputBlock>);
  }

  function commitNumber(event: Event, key: 'units' | 'filters' | 'size'): void {
    const input = event.currentTarget as HTMLInputElement;
    const value = Number(input.value);
    if (input.value.trim() === '' || !Number.isInteger(value) || value <= 0) {
      const current = currentNumber(key);
      if (current !== null) input.value = String(current);
      paramError = 'Enter a positive whole number.';
      return;
    }
    paramError = null;
    patch({ [key]: value } as Partial<Block>);
  }

  function outputSizeFor(padding: 'same' | 'valid'): string {
    if (!inShape || inShape.length !== 3 || spatialSize === null || spatialStride === null) {
      return '';
    }
    const [height, width] = inShape;
    const compute = (dimension: number): number =>
      spatialOutputSize(dimension, spatialSize, spatialStride, padding);
    const result = `${compute(height)}×${compute(width)}`;
    if (padding === 'same' && result === `${height}×${width}`) return `stays ${result}`;
    return `becomes ${result}`;
  }
</script>

{#snippet spatialControls(
  testId: string,
  label: string,
  key: 'kernelSize' | 'poolSize',
  value: number,
  description: string
)}
  <label>
    <span>{label}</span>
    <select
      data-testid={testId}
      title={description}
      value={String(value)}
      onchange={(event) => patch({ [key]: Number(event.currentTarget.value) } as Partial<Block>)}
    >
      {#each sizeChoices as choice (choice)}
        <option value={String(choice)}>{choice}×{choice}</option>
      {/each}
    </select>
    <small>{description}</small>
  </label>

  <label>
    <span>Stride</span>
    <select
      data-testid="param-stride"
      title={PARAM_DESCRIPTIONS.stride}
      value={String(spatialStride)}
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
      value={paddingValue}
      onchange={(event) =>
        patch({ padding: event.currentTarget.value as 'same' | 'valid' } as Partial<Block>)}
    >
      <option value="same">same — {outputSizeFor('same')}</option>
      <option value="valid">valid — {outputSizeFor('valid')}</option>
    </select>
    <small>{PARAM_DESCRIPTIONS.padding}</small>
  </label>
{/snippet}

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
    {/if}

    {#if block.kind === 'linear'}
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
    {/if}

    {#if block.kind === 'conv2d'}
      {@render spatialControls(
        'param-kernel-size',
        'Kernel size',
        'kernelSize',
        block.kernelSize,
        PARAM_DESCRIPTIONS.kernelSize
      )}
    {:else if block.kind === 'maxpool2d'}
      {@render spatialControls(
        'param-pool-size',
        'Pool size',
        'poolSize',
        block.poolSize,
        PARAM_DESCRIPTIONS.poolSize
      )}
    {:else if block.kind === 'upsampling2d'}
      <label>
        <span>Size</span>
        <input
          type="number"
          min="1"
          data-testid="param-upsample-size"
          title={PARAM_DESCRIPTIONS.size}
          value={block.size}
          onchange={(event) => commitNumber(event, 'size')}
        />
        <small>{PARAM_DESCRIPTIONS.size}</small>
      </label>
    {/if}

    {#if paramError}
      <p class="error" data-testid="param-error">{paramError}</p>
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
