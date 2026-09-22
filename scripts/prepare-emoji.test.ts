import { describe, expect, it } from 'vitest';
import {
  cacheName,
  compositeOnWhite,
  downscaleTo,
  encodeEmoji,
  isPreparedFile,
  select3dPaths
} from './prepare-emoji.mjs';

function rgba(
  width: number,
  height: number,
  pixels: number[][]
): { width: number; height: number; data: Uint8Array } {
  const data = new Uint8Array(width * height * 4);
  pixels.forEach((pixel, index) => {
    data[index * 4] = pixel[0];
    data[index * 4 + 1] = pixel[1];
    data[index * 4 + 2] = pixel[2];
    data[index * 4 + 3] = pixel[3];
  });
  return { width, height, data };
}

describe('select3dPaths', () => {
  it('keeps only the 3D PNG paths, sorted', () => {
    const tree = {
      tree: [
        { path: 'assets/Grinning face/3D/grinning_face_3d.png' },
        { path: 'assets/Grinning face/Color/grinning_face_color.svg' },
        { path: 'assets/Cat/Flat/cat_flat.svg' },
        { path: 'assets/Cat/3D/cat_3d.png' },
        { path: 'README.md' }
      ]
    };
    expect(select3dPaths(tree)).toEqual([
      'assets/Cat/3D/cat_3d.png',
      'assets/Grinning face/3D/grinning_face_3d.png'
    ]);
  });

  it('returns an empty list for a malformed tree', () => {
    expect(select3dPaths(null)).toEqual([]);
    expect(select3dPaths({})).toEqual([]);
  });

  it('skips a null entry instead of throwing', () => {
    const tree = {
      tree: [null, { path: 'assets/Cat/3D/cat_3d.png' }]
    };
    expect(select3dPaths(tree)).toEqual(['assets/Cat/3D/cat_3d.png']);
  });
});

describe('cacheName', () => {
  it('distinguishes paths that differ only by separator type', () => {
    expect(cacheName('assets/A b/3D/x_3d.png')).not.toBe(cacheName('assets/A_b/3D/x_3d.png'));
  });
});

function header(count = 2, mutate?: (bytes: Uint8Array) => void): Uint8Array {
  const rows = 2;
  const cols = 2;
  const channels = 3;
  const bytes = new Uint8Array(16);
  bytes.set([0x56, 0x53, 0x4e, 0x45], 0);
  bytes[4] = 1;
  bytes[5] = rows;
  bytes[6] = cols;
  bytes[7] = channels;
  new DataView(bytes.buffer).setUint32(8, count, true);
  mutate?.(bytes);
  return bytes;
}

describe('isPreparedFile', () => {
  it('accepts a valid header with the matching length', () => {
    expect(isPreparedFile(header(2), 16 + 2 * 2 * 2 * 3)).toBe(true);
  });

  it('rejects a wrong magic', () => {
    expect(
      isPreparedFile(
        header(2, (bytes) => (bytes[0] = 0x58)),
        16 + 2 * 2 * 2 * 3
      )
    ).toBe(false);
  });

  it('rejects a wrong version', () => {
    expect(
      isPreparedFile(
        header(2, (bytes) => (bytes[4] = 2)),
        16 + 2 * 2 * 2 * 3
      )
    ).toBe(false);
  });

  it('rejects a mismatched length', () => {
    expect(isPreparedFile(header(2), 16 + 2 * 2 * 2 * 3 + 1)).toBe(false);
  });
});

describe('compositeOnWhite', () => {
  it('leaves an opaque pixel unchanged', () => {
    const png = rgba(1, 1, [[10, 20, 30, 255]]);
    expect(Array.from(compositeOnWhite(png))).toEqual([10, 20, 30]);
  });

  it('turns a transparent pixel white', () => {
    const png = rgba(1, 1, [[10, 20, 30, 0]]);
    expect(Array.from(compositeOnWhite(png))).toEqual([255, 255, 255]);
  });

  it('blends a partly transparent pixel toward white', () => {
    const png = rgba(1, 1, [[0, 0, 0, 128]]);
    expect(Array.from(compositeOnWhite(png))).toEqual([127, 127, 127]);
  });
});

describe('downscaleTo', () => {
  it('averages a 2x2 block down to one pixel', () => {
    const rgb = new Uint8Array([0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30]);
    expect(Array.from(downscaleTo(rgb, 2, 2, 1))).toEqual([15, 15, 15]);
  });

  it('repeats the nearest source pixel when upscaling a smaller source', () => {
    const rgb = new Uint8Array([1, 2, 3, 4, 5, 6]);
    const row = [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6];
    expect(Array.from(downscaleTo(rgb, 2, 1, 4))).toEqual([...row, ...row, ...row, ...row]);
  });

  it('averages non-divisible 3x3 boxes down to 2x2', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    const rgb = new Uint8Array(values.flatMap((value) => [value, value, value]));
    expect(Array.from(downscaleTo(rgb, 3, 3, 2))).toEqual([
      10, 10, 10, 25, 25, 25, 55, 55, 55, 70, 70, 70
    ]);
  });
});

describe('encodeEmoji', () => {
  it('writes the VSNE header and pixel bytes', () => {
    const images = [new Uint8Array([1, 2, 3, 4, 5, 6])];
    const buffer = encodeEmoji(images, 2, 1, 3);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe('VSNE');
    expect(view.getUint8(4)).toBe(1);
    expect(view.getUint8(5)).toBe(2);
    expect(view.getUint8(6)).toBe(1);
    expect(view.getUint8(7)).toBe(3);
    expect(view.getUint32(8, true)).toBe(1);
    expect(view.getUint32(12, true)).toBe(0);
    expect(Array.from(bytes.slice(16))).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
