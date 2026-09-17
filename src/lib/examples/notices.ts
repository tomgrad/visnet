export const WEIGHTS_DISCARDED_NOTICE =
  'The network changed, so training restarted with fresh weights.';

export function weightsDiscardedNotice(hadTrainedModel: boolean): string | null {
  return hadTrainedModel ? WEIGHTS_DISCARDED_NOTICE : null;
}
