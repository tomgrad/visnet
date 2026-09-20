import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import ExperimentHarness from './__stubs__/ExperimentHarness.svelte';
import type { Experiment } from './experiment.svelte';
import type { ModelData } from './runtime';

const runtime = vi.hoisted(() => ({
  buildModel: vi.fn(() => ({ tag: 'model' })),
  compileModel: vi.fn(),
  disposeModel: vi.fn(),
  disposeData: vi.fn(),
  createTrainer: vi.fn(() => ({
    play: vi.fn(async () => {}),
    pause: vi.fn(),
    step: vi.fn(async () => {}),
    dispose: vi.fn()
  })),
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

  it('does not rebuild the model when only a colour changes', async () => {
    const { store, session } = mountExperiment();
    await vi.waitFor(() => expect(session.model).not.toBeNull());

    store.updateBlock(store.network.blocks[1].id, { colour: 'blue' });
    await tick();
    expect(runtime.buildModel).toHaveBeenCalledTimes(1);

    store.updateBlock(store.network.blocks[1].id, { units: 16 });
    await vi.waitFor(() => expect(runtime.buildModel).toHaveBeenCalledTimes(2));
  });

  it('does not recreate the trainer when only a colour or position changes', async () => {
    const { store, session } = mountExperiment();
    await vi.waitFor(() => expect(session.model).not.toBeNull());
    session.setData({ xs: { shape: [4, 2] }, ys: {} } as unknown as ModelData);
    await vi.waitFor(() => expect(runtime.createTrainer).toHaveBeenCalledTimes(1));

    store.updateBlock(store.network.blocks[1].id, { colour: 'blue' });
    store.setPosition(store.network.blocks[1].id, { x: 5, y: 6 });
    await tick();

    expect(runtime.createTrainer).toHaveBeenCalledTimes(1);
  });

  it('reads accuracy tracking from the store task', async () => {
    const store = new NetworkStore();
    store.task = 'reconstruction';
    let session: Experiment | null = null;
    render(ExperimentHarness, {
      props: { store, onready: (created: Experiment) => (session = created) }
    });

    await vi.waitFor(() => expect((session as unknown as Experiment).model).not.toBeNull());
    (session as unknown as Experiment).setData({
      xs: { shape: [2, 2] },
      ys: {}
    } as unknown as ModelData);

    await vi.waitFor(() => expect(runtime.createTrainer).toHaveBeenCalled());
    const lastCall = runtime.createTrainer.mock.calls.at(-1) as unknown[] | undefined;
    expect(lastCall?.[5]).toBe(false);
  });

  it('clears the banner when training starts or steps', async () => {
    const { session } = mountExperiment();
    await vi.waitFor(() => expect(session.model).not.toBeNull());
    session.setData({ xs: { shape: [4, 2] }, ys: {} } as unknown as ModelData);
    await vi.waitFor(() => expect(runtime.createTrainer).toHaveBeenCalledTimes(1));

    session.banner = 'The network changed, so training restarted with fresh weights.';
    await session.play();
    expect(session.banner).toBeNull();

    session.banner = 'The network changed, so training restarted with fresh weights.';
    session.step();
    expect(session.banner).toBeNull();
  });

  it('clears the banner when the model is reset', async () => {
    const { session } = mountExperiment();
    await vi.waitFor(() => expect(session.model).not.toBeNull());
    session.setData({ xs: { shape: [4, 2] }, ys: {} } as unknown as ModelData);
    await vi.waitFor(() => expect(runtime.createTrainer).toHaveBeenCalledTimes(1));

    session.banner = 'Training stopped. Reset the model and try again.';
    session.resetModel();

    expect(session.banner).toBeNull();
  });
});
