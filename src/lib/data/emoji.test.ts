import { describe, expect, it } from 'vitest';
import { emojiAt, parseEmoji } from './emoji';

const ROWS = 2;
const COLS = 2;
const CHANNELS = 3;

function buffer(count = 1, mutate?: (bytes: Uint8Array, view: DataView) => void): ArrayBuffer {
  const pixelBytes = count * ROWS * COLS * CHANNELS;
  const result = new ArrayBuffer(16 + pixelBytes);
  const view = new DataView(result);
  const bytes = new Uint8Array(result);
  bytes.set([0x56, 0x53, 0x4e, 0x45], 0);
  view.setUint8(4, 1);
  view.setUint8(5, ROWS);
  view.setUint8(6, COLS);
  view.setUint8(7, CHANNELS);
  view.setUint32(8, count, true);
  for (let i = 0; i < pixelBytes; i++) bytes[16 + i] = i % 256;
  mutate?.(bytes, view);
  return result;
}

describe('parseEmoji', () => {
  it('parses a valid buffer', () => {
    const dataset = parseEmoji(buffer(2));
    expect(dataset).toMatchObject({ count: 2, rows: ROWS, cols: COLS, channels: CHANNELS });
    expect(dataset?.pixels.length).toBe(2 * ROWS * COLS * CHANNELS);
  });

  it('rejects a wrong magic', () => {
    expect(parseEmoji(buffer(1, (bytes) => bytes.set([0x58], 0)))).toBeNull();
  });

  it('rejects a wrong version', () => {
    expect(parseEmoji(buffer(1, (_bytes, view) => view.setUint8(4, 2)))).toBeNull();
  });

  it('rejects a wrong length', () => {
    expect(parseEmoji(buffer(1).slice(0, 20))).toBeNull();
  });

  it('rejects a zero count', () => {
    expect(parseEmoji(buffer(0))).toBeNull();
  });
});

describe('emojiAt', () => {
  it('returns the bytes for one image', () => {
    const dataset = parseEmoji(buffer(2))!;
    expect(emojiAt(dataset, 1).length).toBe(ROWS * COLS * CHANNELS);
  });

  it('throws for an out-of-range index', () => {
    const dataset = parseEmoji(buffer(1))!;
    expect(() => emojiAt(dataset, 1)).toThrow(RangeError);
  });
});
