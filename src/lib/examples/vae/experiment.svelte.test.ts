import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import VaeExperimentHarness from './__stubs__/VaeExperimentHarness.svelte';
import { VaeController } from './vaeController.svelte';
import type { VaeExperiment } from './experiment.svelte';
import type { ModelData } from '../runtime';

const runtime = vi.hoisted(() => ({
  buildVaeModels: vi.fn(() => ({ encoder: { tag: 'encoder' }, decoder: { tag: 'decoder' } })),
  disposeModel: vi.fn(),
  disposeData: vi.fn(),
  createVaeTrainer: vi.fn(() => ({
    play: vi.fn(async () => {}),
    pause: vi.fn(),
    step: vi.fn(async () => {}),
    dispose: vi.fn()
  })),
  saveVaeWeights: vi.fn(async () => {}),
  loadVaeWeightsInto: vi.fn(async () => false)
}));

vi.mock('../runtime', () => ({
  loadRuntime: vi.fn(async () => runtime)
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe('createVaeExperiment', () => {
  it('builds both models for a valid VAE', async () => {
    const controller = new VaeController();
    let session: VaeExperiment | null = null;
    render(VaeExperimentHarness, {
      props: {
        controller,
        onready: (created: VaeExperiment) => (session = created)
      }
    });
    await vi.waitFor(() => expect(runtime.buildVaeModels).toHaveBeenCalled());
    expect((session as VaeExperiment | null)?.encoder).not.toBeNull();
    expect((session as VaeExperiment | null)?.decoder).not.toBeNull();
  });

  it('does not build models for an invalid VAE', async () => {
    const controller = new VaeController();
    controller.encoderStore.updateBlock(controller.encoderStore.network.blocks[0].id, {
      shape: [2]
    });
    let session: VaeExperiment | null = null;
    render(VaeExperimentHarness, {
      props: {
        controller,
        onready: (created: VaeExperiment) => (session = created)
      }
    });
    await vi.waitFor(() => expect((session as VaeExperiment | null)?.runtime).not.toBeNull());
    expect(controller.isValid).toBe(false);
    expect(runtime.buildVaeModels).not.toHaveBeenCalled();
    expect((session as VaeExperiment | null)?.encoder).toBeNull();
  });

  it('does not recreate the trainer when only a colour or position changes', async () => {
    const controller = new VaeController();
    let session: VaeExperiment | null = null;
    render(VaeExperimentHarness, {
      props: {
        controller,
        onready: (created: VaeExperiment) => (session = created)
      }
    });
    await vi.waitFor(() => expect(runtime.buildVaeModels).toHaveBeenCalled());
    (session as VaeExperiment | null)?.setData({
      xs: { shape: [4, 2] },
      ys: {}
    } as unknown as ModelData);
    await vi.waitFor(() => expect(runtime.createVaeTrainer).toHaveBeenCalledTimes(1));

    const block = controller.encoderStore.network.blocks[1];
    controller.encoderStore.updateBlock(block.id, { colour: 'blue' });
    controller.encoderStore.setPosition(block.id, { x: 5, y: 6 });
    await tick();

    expect(runtime.createVaeTrainer).toHaveBeenCalledTimes(1);
  });
});
