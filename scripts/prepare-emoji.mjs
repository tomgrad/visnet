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
