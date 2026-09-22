import type { VaeNetwork } from '../network/vae';
import type { TrainingConfig } from '../network/types';
import { decodeNetwork } from './networkCodec';

export function encodeVaeNetwork(vae: VaeNetwork): string {
  return JSON.stringify({
    encoder: vae.encoder,
    decoder: vae.decoder,
    latentSize: vae.latentSize,
    training: vae.training
  });
}

function isTraining(value: unknown): value is TrainingConfig {
  if (typeof value !== 'object' || value === null) return false;
  const training = value as Record<string, unknown>;
  return (
    (training.loss === 'mse' || training.loss === 'crossEntropy') &&
    (training.optimizer === 'sgd' || training.optimizer === 'adam') &&
    typeof training.learningRate === 'number' &&
    typeof training.batchSize === 'number'
  );
}

export function decodeVaeNetwork(raw: string): VaeNetwork | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null) return null;
  const payload = parsed as Record<string, unknown>;
  const encoder = decodeNetwork(JSON.stringify(payload.encoder));
  const decoder = decodeNetwork(JSON.stringify(payload.decoder));
  if (!encoder || !decoder) return null;
  if (typeof payload.latentSize !== 'number' || !Number.isInteger(payload.latentSize)) return null;
  if (!isTraining(payload.training)) return null;
  return { encoder, decoder, latentSize: payload.latentSize, training: payload.training };
}
