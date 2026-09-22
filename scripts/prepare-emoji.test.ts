import { describe, expect, it } from 'vitest';
import { compositeOnWhite, downscaleTo, encodeEmoji, select3dPaths } from './prepare-emoji.mjs';

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

  it('always produces the requested size, even from a smaller source', () => {
    const rgb = new Uint8Array([1, 2, 3, 4, 5, 6]);
    expect(downscaleTo(rgb, 2, 1, 4).length).toBe(4 * 4 * 3);
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
