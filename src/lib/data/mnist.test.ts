import { afterEach, describe, expect, it, vi } from 'vitest';
import { HEADER_BYTES, MAGIC } from './images';
import { ImageDataUnavailableError, loadMnistData } from './mnist';

function split(count = 2): ArrayBuffer {
  const rows = 2;
  const cols = 2;
  const numClasses = 3;
  const buffer = new ArrayBuffer(HEADER_BYTES + count * rows * cols + count);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < 4; i++) view.setUint8(i, MAGIC.charCodeAt(i));
  view.setUint8(4, 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true);
  for (let i = 0; i < count; i++) bytes[HEADER_BYTES + count * rows * cols + i] = i % numClasses;
  return buffer;
}

function ok(buffer: ArrayBuffer): Response {
  return { ok: true, arrayBuffer: async () => buffer } as unknown as Response;
}

function notFound(): Response {
  return { ok: false, status: 404, arrayBuffer: async () => split(2) } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadMnistData', () => {
  it('loads both splits', async () => {
    const fetchMock = vi.fn(async () => ok(split(2)));
    vi.stubGlobal('fetch', fetchMock);

    const data = await loadMnistData();

    expect(data.train.count).toBe(2);
    expect(data.test.count).toBe(2);
    expect(fetchMock).toHaveBeenCalledWith('/mnist/train.bin');
    expect(fetchMock).toHaveBeenCalledWith('/mnist/test.bin');
  });

  it('honours a custom base path', async () => {
    const fetchMock = vi.fn(async () => ok(split(2)));
    vi.stubGlobal('fetch', fetchMock);

    await loadMnistData('/assets/digits');

    expect(fetchMock).toHaveBeenCalledWith('/assets/digits/train.bin');
  });

  it('reports a missing asset with the prep command', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => notFound()));

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
    await expect(loadMnistData()).rejects.toThrow(/npm run data:mnist/);
  });

  it('reports a corrupt asset', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok(new ArrayBuffer(4))));

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
  });

  it('reports a failed request', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );

    await expect(loadMnistData()).rejects.toThrow(ImageDataUnavailableError);
  });
});
