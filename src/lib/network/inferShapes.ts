import type { Block, Network } from './types';

export interface ShapeInfo {
  blockId: string;
  inShape: number[] | null;
  outShape: number[] | null;
  paramCount: number | null;
}

export interface EdgeShape {
  fromId: string;
  toId: string;
  shape: number[] | null;
}

export interface ShapeResult {
  perBlock: ShapeInfo[];
  edges: EdgeShape[];
  totalParamCount: number;
}

export function spatialOutputSize(
  size: number,
  window: number,
  stride: number,
  padding: 'same' | 'valid'
): number {
  if (padding === 'same') return Math.ceil(size / stride);
  return Math.floor((size - window) / stride) + 1;
}

function product(shape: number[]): number {
  return shape.reduce((total, dimension) => total * dimension, 1);
}

function outputShapeFor(block: Block, inShape: number[] | null): number[] | null {
  switch (block.kind) {
    case 'input':
      return [...block.shape];
    case 'linear':
      if (!inShape || inShape.length !== 1) return null;
      return [block.units];
    case 'conv2d': {
      if (!inShape || inShape.length !== 3) return null;
      const [height, width] = inShape;
      const outHeight = spatialOutputSize(height, block.kernelSize, block.stride, block.padding);
      const outWidth = spatialOutputSize(width, block.kernelSize, block.stride, block.padding);
      if (outHeight <= 0 || outWidth <= 0) return null;
      return [outHeight, outWidth, block.filters];
    }
    case 'maxpool2d': {
      if (!inShape || inShape.length !== 3) return null;
      const [height, width, channels] = inShape;
      const outHeight = spatialOutputSize(height, block.poolSize, block.stride, block.padding);
      const outWidth = spatialOutputSize(width, block.poolSize, block.stride, block.padding);
      if (outHeight <= 0 || outWidth <= 0) return null;
      return [outHeight, outWidth, channels];
    }
    case 'upsampling2d': {
      if (!inShape || inShape.length !== 3 || block.size < 1) return null;
      const [height, width, channels] = inShape;
      return [height * block.size, width * block.size, channels];
    }
    case 'flatten':
      if (!inShape || inShape.length < 1) return null;
      return [product(inShape)];
    case 'reshape':
      return inShape ? [...block.shape] : null;
    case 'relu':
    case 'sigmoid':
    case 'tanh':
    case 'softmax':
    case 'output':
      return inShape ? [...inShape] : null;
    default:
      return null;
  }
}

function paramCountFor(block: Block, inShape: number[] | null): number | null {
  switch (block.kind) {
    case 'linear':
      if (!inShape || inShape.length !== 1) return null;
      return inShape[0] * block.units + block.units;
    case 'conv2d':
      if (!inShape || inShape.length !== 3) return null;
      return block.kernelSize * block.kernelSize * inShape[2] * block.filters + block.filters;
    default:
      return 0;
  }
}

export function inferShapes(net: Network): ShapeResult {
  const perBlock: ShapeInfo[] = [];
  const edges: EdgeShape[] = [];
  let current: number[] | null = null;
  let broken = false;

  net.blocks.forEach((block, index) => {
    const inShape = current;
    const outShape = broken ? null : outputShapeFor(block, inShape);

    if (!broken && outShape === null) broken = true;

    perBlock.push({
      blockId: block.id,
      inShape,
      outShape,
      paramCount: paramCountFor(block, inShape)
    });

    if (index > 0) {
      edges.push({ fromId: net.blocks[index - 1].id, toId: block.id, shape: inShape });
    }

    current = outShape;
  });

  const totalParamCount = perBlock.reduce((total, info) => total + (info.paramCount ?? 0), 0);

  return { perBlock, edges, totalParamCount };
}
