import type { BlockColour } from '../network/types';

export interface BlockColourOption {
  id: BlockColour;
  label: string;
  hex: string;
}

export const BLOCK_COLOUR_OPTIONS: BlockColourOption[] = [
  { id: 'blue', label: 'Blue', hex: '#dbeafe' },
  { id: 'green', label: 'Green', hex: '#dcfce7' },
  { id: 'amber', label: 'Amber', hex: '#fef3c7' },
  { id: 'red', label: 'Red', hex: '#fee2e2' },
  { id: 'violet', label: 'Violet', hex: '#ede9fe' },
  { id: 'teal', label: 'Teal', hex: '#ccfbf1' },
  { id: 'pink', label: 'Pink', hex: '#fce7f3' },
  { id: 'slate', label: 'Slate', hex: '#e2e8f0' }
];

export const DEFAULT_BLOCK_COLOUR = '#ffffff';

export function blockColourHex(colour: BlockColour | undefined): string {
  return BLOCK_COLOUR_OPTIONS.find((option) => option.id === colour)?.hex ?? DEFAULT_BLOCK_COLOUR;
}
