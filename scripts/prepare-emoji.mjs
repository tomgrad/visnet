import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { PNG } from 'pngjs';

export function select3dPaths(tree) {
  const entries = tree && Array.isArray(tree.tree) ? tree.tree : [];
  return entries
    .map((entry) => entry.path)
    .filter((path) => typeof path === 'string' && /^assets\/.*\/3D\/.*_3d\.png$/.test(path))
    .sort();
}

export function compositeOnWhite(png) {
  const { width, height, data } = png;
  const out = new Uint8Array(width * height * 3);
  for (let index = 0; index < width * height; index++) {
    const alpha = data[index * 4 + 3] / 255;
    for (let channel = 0; channel < 3; channel++) {
      const value = data[index * 4 + channel];
      out[index * 3 + channel] = Math.round(value * alpha + 255 * (1 - alpha));
    }
  }
  return out;
}

export function downscaleTo(rgb, width, height, size) {
  const out = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    const y0 = Math.floor((y * height) / size);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / size));
    for (let x = 0; x < size; x++) {
      const x0 = Math.floor((x * width) / size);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / size));
      for (let channel = 0; channel < 3; channel++) {
        let total = 0;
        let count = 0;
        for (let sy = y0; sy < y1; sy++) {
          for (let sx = x0; sx < x1; sx++) {
            total += rgb[(sy * width + sx) * 3 + channel];
            count += 1;
          }
        }
        out[(y * size + x) * 3 + channel] = Math.round(total / count);
      }
    }
  }
  return out;
}

export function encodeEmoji(images, rows, cols, channels) {
  const count = images.length;
  const pixelBytes = count * rows * cols * channels;
  const buffer = new ArrayBuffer(16 + pixelBytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x56, 0x53, 0x4e, 0x45], 0);
  view.setUint8(4, 1);
  view.setUint8(5, rows);
  view.setUint8(6, cols);
  view.setUint8(7, channels);
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true);
  images.forEach((image, index) => bytes.set(image, 16 + index * rows * cols * channels));
  return buffer;
}

const REPO = 'microsoft/fluentui-emoji';
const TREE_URL = `https://api.github.com/repos/${REPO}/git/trees/main?recursive=1`;
const RAW = `https://raw.githubusercontent.com/${REPO}/main/`;
const OUT_DIR = fileURLToPath(new URL('../static/emoji/', import.meta.url));
const CACHE_DIR = fileURLToPath(new URL('../.cache/emoji/', import.meta.url));
const SIZE = 64;
const CHANNELS = 3;
const CONCURRENCY = 8;

function flag(args, name, fallback) {
  const match = args.find((arg) => arg.startsWith(`${name}=`));
  if (!match) return fallback;
  const value = Number(match.slice(name.length + 1));
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} needs a positive whole number.`);
  }
  return value;
}

function cacheName(path) {
  return path.replace(/[^A-Za-z0-9._-]+/g, '_');
}

async function fetchBuffer(url, attempts = 3) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      if (attempt >= attempts) throw new Error(`${url}: ${error.message}`, { cause: error });
      await new Promise((resolve) => setTimeout(resolve, attempt * 250));
    }
  }
}

async function mapLimit(items, limit, worker) {
  let next = 0;
  const run = async () => {
    while (next < items.length) {
      const index = next++;
      await worker(items[index], index);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
}

async function present(path) {
  try {
    await readFile(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const force = args.includes('--force');
  const limit = flag(args, '--limit', Infinity);

  const outputPath = join(OUT_DIR, 'emoji.bin');
  if (!force && (await present(outputPath))) {
    console.log(`Already prepared at ${outputPath}. Pass --force to download again.`);
    return;
  }

  console.log(`Listing 3D emoji from ${REPO}`);
  const tree = JSON.parse((await fetchBuffer(TREE_URL)).toString('utf8'));
  const all = select3dPaths(tree);
  if (all.length === 0) throw new Error('No 3D emoji PNGs were found in the repository tree.');
  const paths = all.slice(0, limit);
  console.log(`Preparing ${paths.length} of ${all.length} emoji`);

  await mkdir(CACHE_DIR, { recursive: true });
  await mkdir(OUT_DIR, { recursive: true });

  const images = new Array(paths.length);
  let done = 0;
  await mapLimit(paths, CONCURRENCY, async (path, index) => {
    const cached = join(CACHE_DIR, cacheName(path));
    let raw;
    if (!force && (await present(cached))) {
      raw = await readFile(cached);
    } else {
      raw = await fetchBuffer(`${RAW}${path.split('/').map(encodeURIComponent).join('/')}`);
      await writeFile(cached, raw);
    }
    const png = PNG.sync.read(raw);
    images[index] = downscaleTo(compositeOnWhite(png), png.width, png.height, SIZE);
    done += 1;
    if (done % 100 === 0) console.log(`  ${done}/${paths.length}`);
  });

  const buffer = encodeEmoji(images, SIZE, SIZE, CHANNELS);
  await writeFile(outputPath, new Uint8Array(buffer));
  console.log(`Wrote ${images.length} emoji, ${SIZE}x${SIZE}x${CHANNELS}.`);
  console.log(`  ${outputPath} (${buffer.byteLength} bytes)`);
  console.log('Fluent Emoji is MIT-licensed by Microsoft.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
