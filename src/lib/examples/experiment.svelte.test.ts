import { render } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import ExperimentHarness from './__stubs__/ExperimentHarness.svelte';
import type { Experiment } from './experiment.svelte';

const runtime = vi.hoisted(() => ({
  buildModel: vi.fn(() => ({ tag: 'model' })),
  compileModel: vi.fn(),
  disposeModel: vi.fn(),
  disposeData: vi.fn(),
  createTrainer: vi.fn(),
  toTensors: vi.fn(),
  imagesToTensors: vi.fn(),
  saveWeights: vi.fn(async () => {}),
  loadWeightsInto: vi.fn(async () => false)
}));

vi.mock('./runtime', () => ({
  loadRuntime: vi.fn(async () => runtime)
}));

function mountExperiment(): { store: NetworkStore; session: Experiment } {
  const store = new NetworkStore();
  let session: Experiment | null = null;
  render(ExperimentHarness, {
    props: { store, onready: (created: Experiment) => (session = created) }
  });
  return { store, session: session as unknown as Experiment };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('createExperiment', () => {
  it('does not build a model for an invalid network', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    let session: Experiment | null = null;
    render(ExperimentHarness, {
      props: { store, onready: (created: Experiment) => (session = created) }
    });

    await vi.waitFor(() => expect((session as Experiment | null)?.runtime).not.toBeNull());
    expect(store.isValid).toBe(false);
    expect(runtime.buildModel).not.toHaveBeenCalled();
    expect((session as unknown as Experiment).model).toBeNull();
  });

  it('builds a model for a valid network', async () => {
    const { session } = mountExperiment();

    await vi.waitFor(() => expect(session.model).not.toBeNull());
    expect(runtime.buildModel).toHaveBeenCalledTimes(1);
  });

  it('replaces the model when the architecture changes', async () => {
    const { store, session } = mountExperiment();
    await vi.waitFor(() => expect(session.model).not.toBeNull());
    const first = session.model;

    store.updateBlock(store.network.blocks[1].id, { units: 16 });

    await vi.waitFor(() => expect(session.model).not.toBe(first));
    expect(runtime.buildModel).toHaveBeenCalledTimes(2);
    expect(runtime.disposeModel).toHaveBeenCalledWith(first);
  });
});
