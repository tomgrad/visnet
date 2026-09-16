export interface ImageDataset {
  count: number;
  rows: number;
  cols: number;
  numClasses: number;
  pixels: Uint8Array;
  labels: Uint8Array;
}

export const MAGIC = 'VSNT';
export const FORMAT_VERSION = 1;
export const HEADER_BYTES = 16;

function magicAt(view: DataView): string {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
}

export function parseSplit(buffer: ArrayBuffer): ImageDataset | null {
  if (buffer.byteLength < HEADER_BYTES) return null;

  const view = new DataView(buffer);
  if (magicAt(view) !== MAGIC) return null;
  if (view.getUint8(4) !== FORMAT_VERSION) return null;

  const rows = view.getUint8(5);
  const cols = view.getUint8(6);
  const numClasses = view.getUint8(7);
  const count = view.getUint32(8, true);
  const reserved = view.getUint32(12, true);

  if (rows === 0 || cols === 0 || numClasses === 0 || count === 0) return null;
  if (reserved !== 0) return null;

  const pixelBytes = count * rows * cols;
  if (buffer.byteLength !== HEADER_BYTES + pixelBytes + count) return null;

  const pixels = new Uint8Array(buffer, HEADER_BYTES, pixelBytes).slice();
  const labels = new Uint8Array(buffer, HEADER_BYTES + pixelBytes, count).slice();
  for (const label of labels) {
    if (label >= numClasses) return null;
  }

  return { count, rows, cols, numClasses, pixels, labels };
}

export function imageAt(dataset: ImageDataset, index: number): Uint8Array {
  if (!Number.isInteger(index) || index < 0 || index >= dataset.count) {
    throw new RangeError(`Image ${index} is out of range for a dataset of ${dataset.count}.`);
  }
  const size = dataset.rows * dataset.cols;
  return dataset.pixels.subarray(index * size, (index + 1) * size);
}
