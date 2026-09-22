import type { TrainStats } from '../training/Trainer';

export const WEIGHTS_DISCARDED_NOTICE =
  'The network changed, so training restarted with fresh weights.';

export function createStatsTracker() {
  let stats: TrainStats | null = null;
  let lossPoints: number[] = [];
  let redrawKey = 0;

  return {
    track(next: TrainStats): void {
      redrawKey += 1;
      stats = next;
      if (next.epochMeanLoss !== null) {
        lossPoints = [...lossPoints, next.epochMeanLoss].slice(-200);
      }
    },
    reset(): void {
      stats = null;
      lossPoints = [];
      redrawKey = 0;
    },
    get stats() {
      return stats;
    },
    get lossPoints() {
      return lossPoints;
    },
    get redrawKey() {
      return redrawKey;
    }
  };
}
