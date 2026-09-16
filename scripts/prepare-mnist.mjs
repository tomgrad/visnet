#!/usr/bin/env node
import { access, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

const SOURCE = 'https://storage.googleapis.com/cvdf-datasets/mnist';
const OUT_DIR = fileURLToPath(new URL('../static/mnist/', import.meta.url));
const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz'
};
const MAGIC_IMAGES = 0x00000803;
const MAGIC_LABELS = 0x00000801;

export function parseIdxImages(buffer) {
  if (buffer.byteLength < 16) throw new Error('Truncated IDX image header.');
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== MAGIC_IMAGES) throw new Error('Not an IDX image file.');
  const count = view.getUint32(4, false);
  const rows = view.getUint32(8, false);
  const cols = view.getUint32(12, false);
  const expected = 16 + count * rows * cols;
  if (buffer.byteLength !== expected) {
    throw new Error(`Truncated image data: ${buffer.byteLength} of ${expected} bytes.`);
  }
  return { count, rows, cols, pixels: new Uint8Array(buffer, 16, count * rows * cols) };
}

export function parseIdxLabels(buffer) {
  if (buffer.byteLength < 8) throw new Error('Truncated IDX label header.');
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== MAGIC_LABELS) throw new Error('Not an IDX label file.');
  const count = view.getUint32(4, false);
  if (buffer.byteLength !== 8 + count) {
    throw new Error(`Truncated label data: ${buffer.byteLength} of ${8 + count} bytes.`);
  }
  return new Uint8Array(buffer, 8, count);
}

export function encodeSplit({ count, rows, cols, numClasses, pixels, labels }) {
  const pixelBytes = count * rows * cols;
  const buffer = new ArrayBuffer(16 + pixelBytes + count);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  bytes.set([0x56, 0x53, 0x4e, 0x54], 0);
  view.setUint8(4, 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, numClasses);
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true);
  bytes.set(pixels.subarray(0, pixelBytes), 16);
  bytes.set(labels.subarray(0, count), 16 + pixelBytes);
  return buffer;
}

async function download(name) {
  const response = await fetch(`${SOURCE}/${name}`);
  if (!response.ok) {
    throw new Error(`Could not download ${name}: HTTP ${response.status}.`);
  }
  const raw = gunzipSync(new Uint8Array(await response.arrayBuffer()));
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
}

function take(limit, label, images, labels) {
  const available = Math.min(images.count, labels.length);
  if (limit > available) {
    throw new Error(`Asked for ${limit} ${label} images but the file holds ${available}.`);
  }
  const size = images.rows * images.cols;
  return {
    count: limit,
    rows: images.rows,
    cols: images.cols,
    numClasses: 10,
    pixels: images.pixels.slice(0, limit * size),
    labels: labels.slice(0, limit)
  };
}

function flag(args, name, fallback) {
  const match = args.find((arg) => arg.startsWith(`${name}=`));
  if (!match) return fallback;
  const value = Number(match.slice(name.length + 1));
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} needs a positive whole number.`);
  }
  return value;
}

async function present(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const trainCount = flag(args, '--train', 1000);
  const testCount = flag(args, '--test', 200);
  const force = args.includes('--force');

  const trainPath = join(OUT_DIR, 'train.bin');
  const testPath = join(OUT_DIR, 'test.bin');

  if (!force && (await present(trainPath)) && (await present(testPath))) {
    console.log(`Already prepared at ${OUT_DIR}. Pass --force to download again.`);
    return;
  }

  console.log(`Downloading MNIST from ${SOURCE}`);
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    download(FILES.trainImages).then(parseIdxImages),
    download(FILES.trainLabels).then(parseIdxLabels),
    download(FILES.testImages).then(parseIdxImages),
    download(FILES.testLabels).then(parseIdxLabels)
  ]);

  const train = take(trainCount, 'training', trainImages, trainLabels);
  const test = take(testCount, 'test', testImages, testLabels);

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(trainPath, new Uint8Array(encodeSplit(train)));
  await writeFile(testPath, new Uint8Array(encodeSplit(test)));

  console.log(
    `Wrote ${train.count} training and ${test.count} test digits, ${train.rows}x${test.cols}.`
  );
  console.log(`  ${trainPath}`);
  console.log(`  ${testPath}`);
  console.log('MNIST is a derivative of the NIST Special Database 19.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
