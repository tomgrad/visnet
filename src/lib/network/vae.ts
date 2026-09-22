import { findProblems, type Problem } from './problems';
import type { Block, Network, TrainingConfig } from './types';

export interface VaeNetwork {
  encoder: Network;
  decoder: Network;
  latentSize: number;
  training: TrainingConfig;
}

const DEFAULT_TRAINING: TrainingConfig = {
  loss: 'mse',
  optimizer: 'adam',
  learningRate: 0.001,
  batchSize: 32
};

export function createVaeNetwork(latentSize = 2): VaeNetwork {
  const encoder: Network = {
    blocks: [
      { id: 'input', kind: 'input', shape: [64, 64, 3] },
      {
        id: 'conv-encode-1',
        kind: 'conv2d',
        filters: 16,
        kernelSize: 3,
        stride: 2,
        padding: 'same'
      },
      { id: 'relu-encode-1', kind: 'relu' },
      {
        id: 'conv-encode-2',
        kind: 'conv2d',
        filters: 32,
        kernelSize: 3,
        stride: 2,
        padding: 'same'
      },
      { id: 'relu-encode-2', kind: 'relu' },
      { id: 'flatten', kind: 'flatten' },
      { id: 'latent', kind: 'linear', units: latentSize },
      { id: 'latent-output', kind: 'output', shape: [latentSize] }
    ],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
  const decoder: Network = {
    blocks: [
      { id: 'latent-input', kind: 'input', shape: [latentSize] },
      { id: 'expand', kind: 'linear', units: 16 * 16 * 32 },
      { id: 'reshape', kind: 'reshape', shape: [16, 16, 32] },
      { id: 'upsample-1', kind: 'upsampling2d', size: 2 },
      {
        id: 'conv-decode-1',
        kind: 'conv2d',
        filters: 16,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      },
      { id: 'relu-decode-1', kind: 'relu' },
      { id: 'upsample-2', kind: 'upsampling2d', size: 2 },
      {
        id: 'conv-decode-2',
        kind: 'conv2d',
        filters: 3,
        kernelSize: 3,
        stride: 1,
        padding: 'same'
      },
      { id: 'sigmoid', kind: 'sigmoid' },
      { id: 'output', kind: 'output', shape: [64, 64, 3] }
    ],
    training: { ...DEFAULT_TRAINING },
    positions: {}
  };
  return { encoder, decoder, latentSize, training: { ...DEFAULT_TRAINING } };
}

export function syncLatentSize(vae: VaeNetwork, latentSize: number): VaeNetwork {
  const latentIndex = vae.encoder.blocks.reduce(
    (found, block, index) => (block.kind === 'linear' ? index : found),
    -1
  );
  return {
    ...vae,
    latentSize,
    encoder: {
      ...vae.encoder,
      blocks: vae.encoder.blocks.map((block, index) => {
        if (index === latentIndex) return { ...block, units: latentSize } as Block;
        if (block.kind === 'output') return { ...block, shape: [latentSize] };
        return block;
      })
    },
    decoder: {
      ...vae.decoder,
      blocks: vae.decoder.blocks.map((block) =>
        block.kind === 'input' ? { ...block, shape: [latentSize] } : block
      )
    }
  };
}

export function syncTraining(vae: VaeNetwork): VaeNetwork {
  return {
    ...vae,
    encoder: { ...vae.encoder, training: { ...vae.training } },
    decoder: { ...vae.decoder, training: { ...vae.training } }
  };
}

const DECODER_SUPPRESSED_WARNING = 'Convolution layer without image input';

function error(problem: Omit<Problem, 'severity'>): Problem {
  return { severity: 'error', ...problem };
}

function boundaryShape(block: Block | undefined): number[] | null {
  if (!block || (block.kind !== 'input' && block.kind !== 'output')) return null;
  return block.shape;
}

function matchesLatent(shape: number[] | null, latentSize: number): boolean {
  return shape !== null && shape.length === 1 && shape[0] === latentSize;
}

export function vaeProblems(vae: VaeNetwork): Problem[] {
  const problems: Problem[] = [];

  if (!Number.isInteger(vae.latentSize) || vae.latentSize < 1) {
    problems.push(
      error({
        title: 'Latent size must be a whole number',
        message: `The latent size is ${vae.latentSize}. It must be a whole number of at least 1.`,
        fix: 'Set the Latent size to a whole number such as 2 or 16.'
      })
    );
  }

  const encoderOutput = vae.encoder.blocks.find((block) => block.kind === 'output');
  if (encoderOutput && !matchesLatent(boundaryShape(encoderOutput), vae.latentSize)) {
    problems.push(
      error({
        title: 'The encoder does not produce the latent size',
        message: `The encoder ends with ${JSON.stringify(boundaryShape(encoderOutput))}, but the latent size is ${vae.latentSize}.`,
        fix: 'Set the Latent size to match the encoder, or change the encoder to end at the latent size.',
        blockId: encoderOutput.id
      })
    );
  }

  const decoderInput = vae.decoder.blocks.find((block) => block.kind === 'input');
  if (decoderInput && !matchesLatent(boundaryShape(decoderInput), vae.latentSize)) {
    problems.push(
      error({
        title: 'The decoder does not start from the latent size',
        message: `The decoder starts with ${JSON.stringify(boundaryShape(decoderInput))}, but the latent size is ${vae.latentSize}.`,
        fix: 'Set the Latent size to match the decoder, or change the decoder to start from the latent size.',
        blockId: decoderInput.id
      })
    );
  }

  problems.push(...findProblems(vae.encoder, { task: 'reconstruction' }));
  problems.push(
    ...findProblems(vae.decoder, { task: 'reconstruction' }).filter(
      (problem) => problem.title !== DECODER_SUPPRESSED_WARNING
    )
  );

  return problems;
}
