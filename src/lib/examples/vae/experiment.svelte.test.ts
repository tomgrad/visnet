import { render } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import VaeExperimentHarness from './__stubs__/VaeExperimentHarness.svelte';
import { VaeController } from './vaeController.svelte';
import type { VaeExperiment } from './experiment.svelte';

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
});
