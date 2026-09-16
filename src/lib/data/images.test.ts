import { describe, expect, it } from 'vitest';
import { HEADER_BYTES, MAGIC, parseSplit, imageAt, type ImageDataset } from './images';

const ROWS = 2;
const COLS = 3;
const CLASSES = 3;

function build(
  options: {
    count?: number;
    rows?: number;
    cols?: number;
    numClasses?: number;
    magic?: string;
    version?: number;
    reserved?: number;
    extraBytes?: number;
    labels?: number[];
  } = {}
): ArrayBuffer {
  const count = options.count ?? 2;
  const rows = options.rows ?? ROWS;
  const cols = options.cols ?? COLS;
  const numClasses = options.numClasses ?? CLASSES;
  const labels = options.labels ?? Array.from({ length: count }, (_, i) => i % numClasses);
  const pixelBytes = count * rows * cols;

  const buffer = new ArrayBuffer(HEADER_BYTES + pixelBytes + count + (options.extraBytes ?? 0));
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  const magic = options.magic ?? MAGIC;
  for (let i = 0; i < 4; i++) view.setUint8(i, magic.charCodeAt(i) ?? 0);
  view.setUint8(4, options.version ?? 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, options.reserved ?? 0, true);

  for (let i = 0; i < pixelBytes; i++) bytes[HEADER_BYTES + i] = i % 256;
  for (let i = 0; i < count; i++) bytes[HEADER_BYTES + pixelBytes + i] = labels[i];
  return buffer;
}

function dataset(overrides: Partial<ImageDataset> = {}): ImageDataset {
  const parsed = parseSplit(build());
  if (!parsed) throw new Error('fixture failed to parse');
  return { ...parsed, ...overrides };
}

describe('parseSplit', () => {
  it('reads the header and the pixel and label bytes', () => {
    const parsed = parseSplit(build());
    expect(parsed).not.toBeNull();
    expect(parsed).toMatchObject({ count: 2, rows: ROWS, cols: COLS, numClasses: CLASSES });
    expect(parsed?.pixels).toHaveLength(2 * ROWS * COLS);
    expect(Array.from(parsed?.labels ?? [])).toEqual([0, 1]);
  });

  it('copies the bytes rather than aliasing the buffer', () => {
    const buffer = build();
    const parsed = parseSplit(buffer);
    new Uint8Array(buffer)[HEADER_BYTES] = 99;
    expect(parsed?.pixels[0]).toBe(0);
  });

  it('rejects a buffer shorter than the header', () => {
    expect(parseSplit(new ArrayBuffer(HEADER_BYTES - 1))).toBeNull();
  });

  it('rejects a wrong magic', () => {
    expect(parseSplit(build({ magic: 'NOPE' }))).toBeNull();
  });

  it('rejects an unsupported version', () => {
    expect(parseSplit(build({ version: 2 }))).toBeNull();
  });

  it('rejects a non-zero reserved field', () => {
    expect(parseSplit(build({ reserved: 7 }))).toBeNull();
  });

  it('rejects zero dimensions, classes or count', () => {
    expect(parseSplit(build({ rows: 0 }))).toBeNull();
    expect(parseSplit(build({ cols: 0 }))).toBeNull();
    expect(parseSplit(build({ numClasses: 0 }))).toBeNull();
    expect(parseSplit(build({ count: 0, labels: [] }))).toBeNull();
  });

  it('rejects trailing bytes', () => {
    expect(parseSplit(build({ extraBytes: 4 }))).toBeNull();
  });

  it('rejects a label beyond the class count', () => {
    expect(parseSplit(build({ numClasses: 2, labels: [0, 5] }))).toBeNull();
  });
});

describe('imageAt', () => {
  it('returns that image slice', () => {
    const parsed = dataset();
    expect(Array.from(imageAt(parsed, 1))).toEqual([6, 7, 8, 9, 10, 11]);
  });

  it('rejects an out-of-range or non-integer index', () => {
    const parsed = dataset();
    expect(() => imageAt(parsed, 2)).toThrow(RangeError);
    expect(() => imageAt(parsed, -1)).toThrow(RangeError);
    expect(() => imageAt(parsed, 0.5)).toThrow(RangeError);
  });
});
