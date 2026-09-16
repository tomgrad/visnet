import { describe, expect, it } from 'vitest';
import { gridCellRect, predictionsFromLogits } from './grid';

describe('predictionsFromLogits', () => {
  it('takes the argmax and compares it with the label', () => {
    const logits = [
      [0.1, 0.7, 0.2],
      [0.6, 0.3, 0.1],
      [0.2, 0.2, 0.6]
    ];
    const labels = Uint8Array.from([1, 0, 2]);

    expect(predictionsFromLogits(logits, labels, [0, 1, 2])).toEqual([
      { predicted: 1, correct: true },
      { predicted: 0, correct: true },
      { predicted: 2, correct: true }
    ]);
  });

  it('marks a wrong prediction', () => {
    const result = predictionsFromLogits([[0.1, 0.9]], Uint8Array.from([0]), [0]);
    expect(result).toEqual([{ predicted: 1, correct: false }]);
  });

  it('reads the label through the index list', () => {
    const labels = Uint8Array.from([9, 9, 4]);
    const result = predictionsFromLogits([[0.1, 0.2, 0.3, 0.4, 0.5]], labels, [2]);
    expect(result).toEqual([{ predicted: 4, correct: true }]);
  });

  it('breaks a tie towards the lowest class', () => {
    const result = predictionsFromLogits([[0.5, 0.5, 0.5]], Uint8Array.from([0]), [0]);
    expect(result[0].predicted).toBe(0);
  });

  it('returns one entry per logits row', () => {
    expect(predictionsFromLogits([], Uint8Array.from([]), [])).toEqual([]);
  });
});

describe('gridCellRect', () => {
  const width = 36;
  const height = 48;
  const gap = 4;

  it('places the first cell at the origin', () => {
    expect(gridCellRect(0, 8, width, height, gap)).toEqual({ x: 0, y: 0, width, height });
  });

  it('advances across a row, then wraps', () => {
    expect(gridCellRect(7, 8, width, height, gap)).toEqual({
      x: 7 * (width + gap),
      y: 0,
      width,
      height
    });
    expect(gridCellRect(8, 8, width, height, gap)).toEqual({
      x: 0,
      y: height + gap,
      width,
      height
    });
  });

  it('places the last of forty cells in the final row', () => {
    expect(gridCellRect(39, 8, width, height, gap)).toEqual({
      x: 7 * (width + gap),
      y: 4 * (height + gap),
      width,
      height
    });
  });
});
