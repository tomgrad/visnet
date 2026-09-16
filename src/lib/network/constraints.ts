import { replaceBlock } from './chain';
import { inferShapes } from './inferShapes';
import type { Block, Network } from './types';

export interface ParameterBounds {
  kernelSize: number[];
  stride: number[];
}

export interface ClampResult {
  patch: Partial<Block>;
  announcement: string | null;
}

export interface NetworkClampResult {
  network: Network;
  announcements: string[];
}

const MINIMUM_COUNT = 1;

export function parameterBounds(inShape: number[] | null): ParameterBounds | null {
  if (!inShape || inShape.length !== 3) return null;
  const limit = Math.min(inShape[0], inShape[1]);
  if (!Number.isFinite(limit) || limit < MINIMUM_COUNT) return null;
  const choices = Array.from({ length: Math.floor(limit) }, (_, index) => index + MINIMUM_COUNT);
  return { kernelSize: choices, stride: choices };
}

function clampCount(value: number, label: string): { value: number; announcement: string | null } {
  const rounded = Number.isFinite(value)
    ? Math.max(MINIMUM_COUNT, Math.floor(value))
    : MINIMUM_COUNT;
  if (rounded === value) return { value, announcement: null };
  return {
    value: rounded,
    announcement: `${label} changed from ${value} to ${rounded}. A layer must produce at least one number.`
  };
}

function clampDimension(
  value: number,
  label: string,
  limit: number,
  inShape: number[]
): { value: number; announcement: string | null } {
  const rounded = Number.isFinite(value)
    ? Math.min(Math.max(MINIMUM_COUNT, Math.floor(value)), limit)
    : MINIMUM_COUNT;
  if (rounded === value) return { value, announcement: null };
  const [height, width] = inShape;
  return {
    value: rounded,
    announcement: `${label} changed from ${value} to ${rounded} because the incoming data is ${height}×${width}.`
  };
}

export function clampBlockPatch(net: Network, id: string, patch: Partial<Block>): ClampResult {
  const index = net.blocks.findIndex((block) => block.id === id);
  if (index === -1) return { patch, announcement: null };

  const inShape = inferShapes(net).perBlock[index].inShape;
  const bounds = parameterBounds(inShape);
  const result: Partial<Block> = { ...patch };
  let announcement: string | null = null;

  if ('units' in patch && typeof patch.units === 'number') {
    const next = clampCount(patch.units, 'Units');
    Object.assign(result, { units: next.value });
    if (announcement === null) announcement = next.announcement;
  }

  if ('filters' in patch && typeof patch.filters === 'number') {
    const next = clampCount(patch.filters, 'Filters');
    Object.assign(result, { filters: next.value });
    if (announcement === null) announcement = next.announcement;
  }

  if (bounds && inShape) {
    const limit = Math.max(...bounds.kernelSize);
    if ('kernelSize' in patch && typeof patch.kernelSize === 'number') {
      const next = clampDimension(patch.kernelSize, 'Kernel size', limit, inShape);
      Object.assign(result, { kernelSize: next.value });
      if (announcement === null) announcement = next.announcement;
    }
    if ('stride' in patch && typeof patch.stride === 'number') {
      const next = clampDimension(patch.stride, 'Stride', limit, inShape);
      Object.assign(result, { stride: next.value });
      if (announcement === null) announcement = next.announcement;
    }
  }

  return { patch: result, announcement };
}

export function clampNetwork(net: Network): NetworkClampResult {
  let current = net;
  const announcements: string[] = [];

  for (let pass = 0; pass < net.blocks.length; pass++) {
    let changed = false;
    const { perBlock } = inferShapes(current);

    current.blocks.forEach((block, index) => {
      if (block.kind !== 'conv2d') return;
      const bounds = parameterBounds(perBlock[index].inShape);
      if (!bounds) return;

      const corrected = clampBlockPatch(current, block.id, {
        kernelSize: block.kernelSize,
        stride: block.stride
      });
      if (!corrected.announcement) return;

      current = replaceBlock(current, block.id, corrected.patch);
      announcements.push(corrected.announcement);
      changed = true;
    });

    if (!changed) break;
  }

  return { network: current, announcements };
}
