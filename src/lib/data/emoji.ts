export interface EmojiDataset {
  count: number;
  rows: number;
  cols: number;
  channels: number;
  pixels: Uint8Array;
}

export const EMOJI_MAGIC = 'VSNE';
export const EMOJI_FORMAT_VERSION = 1;
export const EMOJI_HEADER_BYTES = 16;

export class EmojiDataUnavailableError extends Error {
  constructor(message = 'The emoji images are not prepared. Run `npm run data:emoji`, then reload.') {
    super(message);
    this.name = 'EmojiDataUnavailableError';
  }
}

function magicAt(view: DataView): string {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
}

export function parseEmoji(buffer: ArrayBuffer): EmojiDataset | null {
  if (buffer.byteLength < EMOJI_HEADER_BYTES) return null;

  const view = new DataView(buffer);
  if (magicAt(view) !== EMOJI_MAGIC) return null;
  if (view.getUint8(4) !== EMOJI_FORMAT_VERSION) return null;

  const rows = view.getUint8(5);
  const cols = view.getUint8(6);
  const channels = view.getUint8(7);
  const count = view.getUint32(8, true);
  const reserved = view.getUint32(12, true);

  if (rows === 0 || cols === 0 || channels === 0 || count === 0) return null;
  if (reserved !== 0) return null;

  const pixelBytes = count * rows * cols * channels;
  if (buffer.byteLength !== EMOJI_HEADER_BYTES + pixelBytes) return null;

  return {
    count,
    rows,
    cols,
    channels,
    pixels: new Uint8Array(buffer, EMOJI_HEADER_BYTES, pixelBytes).slice()
  };
}

export function emojiAt(dataset: EmojiDataset, index: number): Uint8Array {
  if (!Number.isInteger(index) || index < 0 || index >= dataset.count) {
    throw new RangeError(`Emoji ${index} is out of range for a dataset of ${dataset.count}.`);
  }
  const size = dataset.rows * dataset.cols * dataset.channels;
  return dataset.pixels.subarray(index * size, (index + 1) * size);
}

export async function loadEmojiData(base = '/emoji'): Promise<EmojiDataset> {
  try {
    const response = await fetch(`${base}/emoji.bin`);
    if (!response.ok) throw new EmojiDataUnavailableError();
    const parsed = parseEmoji(await response.arrayBuffer());
    if (!parsed) throw new EmojiDataUnavailableError();
    return parsed;
  } catch {
    throw new EmojiDataUnavailableError();
  }
}
