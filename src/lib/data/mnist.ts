import { parseSplit, type ImageDataset } from './images';

export interface MnistData {
  train: ImageDataset;
  test: ImageDataset;
}

export class ImageDataUnavailableError extends Error {
  constructor(
    message = 'The digit images are not prepared. Run `npm run data:mnist`, then reload.'
  ) {
    super(message);
    this.name = 'ImageDataUnavailableError';
  }
}

async function loadSplit(url: string): Promise<ImageDataset> {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new ImageDataUnavailableError();
    const parsed = parseSplit(await response.arrayBuffer());
    if (!parsed) throw new ImageDataUnavailableError();
    return parsed;
  } catch {
    throw new ImageDataUnavailableError();
  }
}

export async function loadMnistData(base = '/mnist'): Promise<MnistData> {
  const [train, test] = await Promise.all([
    loadSplit(`${base}/train.bin`),
    loadSplit(`${base}/test.bin`)
  ]);
  return { train, test };
}
