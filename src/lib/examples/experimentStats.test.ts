import { describe, expect, it } from 'vitest';
import { createStatsTracker } from './experimentStats';

describe('createStatsTracker', () => {
  it('keeps the latest stats and appends an epoch loss', () => {
    const tracker = createStatsTracker();
    tracker.track({ epoch: 1, batch: 2, batchLoss: 0.5, epochMeanLoss: null, epochAccuracy: null });
    expect(tracker.stats?.epochMeanLoss).toBeNull();
    expect(tracker.lossPoints).toEqual([]);

    tracker.track({ epoch: 2, batch: 0, batchLoss: 0.4, epochMeanLoss: 0.4, epochAccuracy: null });
    expect(tracker.stats?.epochMeanLoss).toBe(0.4);
    expect(tracker.lossPoints).toEqual([0.4]);
    expect(tracker.redrawKey).toBe(2);
  });

  it('resets the stats and points', () => {
    const tracker = createStatsTracker();
    tracker.track({ epoch: 1, batch: 0, batchLoss: 0.4, epochMeanLoss: 0.4, epochAccuracy: null });
    tracker.reset();
    expect(tracker.stats).toBeNull();
    expect(tracker.lossPoints).toEqual([]);
  });
});
