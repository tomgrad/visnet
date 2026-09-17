export interface ClassColour {
  hex: string;
  rgb: [number, number, number];
}

export const CLASS_COLOURS: [ClassColour, ClassColour] = [
  { hex: '#38bdf8', rgb: [56, 189, 248] },
  { hex: '#fb7185', rgb: [251, 113, 133] }
];

export function classColour(index: number): ClassColour {
  return CLASS_COLOURS[index] ?? { hex: '#94a3b8', rgb: [148, 163, 184] };
}

export const BACKGROUND_RGB: [number, number, number] = [248, 250, 252];
