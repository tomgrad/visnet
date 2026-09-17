import { describe, expect, it } from 'vitest';
import { WEIGHTS_DISCARDED_NOTICE, weightsDiscardedNotice } from './notices';

describe('weightsDiscardedNotice', () => {
  it('explains that changing the network discarded trained weights', () => {
    expect(weightsDiscardedNotice(true)).toBe(WEIGHTS_DISCARDED_NOTICE);
  });

  it('says nothing when the model had not been trained', () => {
    expect(weightsDiscardedNotice(false)).toBeNull();
  });
});
