import { describe, expect, it } from 'vitest';
import { parseSplit } from '../src/lib/data/images';
import { encodeSplit, parseIdxImages, parseIdxLabels } from './prepare-mnist.mjs';

const ROWS = 2;
const COLS = 3;

function idxImages(count: number, rows = ROWS, cols = COLS): ArrayBuffer {
  const buffer = new ArrayBuffer(16 + count * rows * cols);
  const view = new DataView(buffer);
  view.setUint32(0, 0x00000803, false);
  view.setUint32(4, count, false);
  view.setUint32(8, rows, false);
  view.setUint32(12, cols, false);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < count * rows * cols; i++) bytes[16 + i] = i % 256;
  return buffer;
}

function idxLabels(labels: number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(8 + labels.length);
  const view = new DataView(buffer);
  view.setUint32(0, 0x00000801, false);
  view.setUint32(4, labels.length, false);
  const bytes = new Uint8Array(buffer);
  labels.forEach((label, i) => (bytes[8 + i] = label));
  return buffer;
}

describe('parseIdxImages', () => {
  it('reads the big-endian header and pixels', () => {
    const parsed = parseIdxImages(idxImages(2));
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS });
    expect(parsed.pixels).toHaveLength(2 * ROWS * COLS);
    expect(parsed.pixels[0]).toBe(0);
  });

  it('rejects a wrong magic', () => {
    const buffer = new ArrayBuffer(16);
    new DataView(buffer).setUint32(0, 0x00000801, false);
    expect(() => parseIdxImages(buffer)).toThrow(/image/i);
  });

  it('rejects a truncated body', () => {
    const buffer = idxImages(2);
    expect(() => parseIdxImages(buffer.slice(0, buffer.byteLength - 1))).toThrow(/truncated/i);
  });
});

describe('parseIdxLabels', () => {
  it('reads the big-endian header and labels', () => {
    expect(Array.from(parseIdxLabels(idxLabels([3, 7])))).toEqual([3, 7]);
  });

  it('rejects a wrong magic', () => {
    expect(() => parseIdxLabels(idxImages(1))).toThrow(/label/i);
  });

  it('rejects a truncated body', () => {
    const buffer = idxLabels([1, 2]);
    expect(() => parseIdxLabels(buffer.slice(0, buffer.byteLength - 1))).toThrow(/truncated/i);
  });
});

describe('encodeSplit', () => {
  it('round-trips through the parser the browser uses', () => {
    const encoded = encodeSplit({
      count: 2,
      rows: ROWS,
      cols: COLS,
      numClasses: 3,
      pixels: Uint8Array.from({ length: 2 * ROWS * COLS }, (_, i) => i % 256),
      labels: Uint8Array.from([0, 2])
    });

    const parsed = parseSplit(encoded);
    expect(parsed).not.toBeNull();
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS, numClasses: 3 });
    expect(Array.from(parsed?.labels ?? [])).toEqual([0, 2]);
    expect(parsed?.pixels[0]).toBe(0);
    expect(parsed?.pixels[5]).toBe(5);
  });
});
