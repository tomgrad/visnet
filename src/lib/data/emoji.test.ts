import { afterEach, describe, expect, it, vi } from 'vitest';
import { EmojiDataUnavailableError, emojiAt, loadEmojiData, parseEmoji } from './emoji';

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

function ok(body: ArrayBuffer): Response {
  return { ok: true, arrayBuffer: async () => body } as unknown as Response;
}

function notFound(): Response {
  return { ok: false, status: 404, arrayBuffer: async () => buffer(2) } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadEmojiData', () => {
  it('fetches /emoji/emoji.bin and returns the parsed dataset', async () => {
    const fetchMock = vi.fn(async () => ok(buffer(2)));
    vi.stubGlobal('fetch', fetchMock);

    const data = await loadEmojiData();

    expect(data).toMatchObject({ count: 2, rows: ROWS, cols: COLS, channels: CHANNELS });
    expect(data.pixels.length).toBe(2 * ROWS * COLS * CHANNELS);
    expect(fetchMock).toHaveBeenCalledWith('/emoji/emoji.bin');
  });

  it('honours a custom base path', async () => {
    const fetchMock = vi.fn(async () => ok(buffer(1)));
    vi.stubGlobal('fetch', fetchMock);

    await loadEmojiData('/assets/emoji');

    expect(fetchMock).toHaveBeenCalledWith('/assets/emoji/emoji.bin');
  });

  it('reports a missing asset with the prep command', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => notFound())
    );

    await expect(loadEmojiData()).rejects.toThrow(EmojiDataUnavailableError);
    await expect(loadEmojiData()).rejects.toThrow(/npm run data:emoji/);
  });

  it('reports a corrupt asset', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ok(new ArrayBuffer(4)))
    );

    await expect(loadEmojiData()).rejects.toThrow(EmojiDataUnavailableError);
  });

  it('reports a failed request', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );

    await expect(loadEmojiData()).rejects.toThrow(EmojiDataUnavailableError);
  });
});
