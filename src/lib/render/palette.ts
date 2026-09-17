export interface ClassColour {
  hex: string;
  rgb: [number, number, number];
}

export const CLASS_COLOURS: ClassColour[] = [
  { hex: '#38bdf8', rgb: [56, 189, 248] },
  { hex: '#fb7185', rgb: [251, 113, 133] },
  { hex: '#f59e0b', rgb: [245, 158, 11] },
  { hex: '#10b981', rgb: [16, 185, 129] },
  { hex: '#8b5cf6', rgb: [139, 92, 246] },
  { hex: '#f97316', rgb: [249, 115, 22] },
  { hex: '#14b8a6', rgb: [20, 184, 166] },
  { hex: '#d946ef', rgb: [217, 70, 239] },
  { hex: '#84cc16', rgb: [132, 204, 22] },
  { hex: '#6366f1', rgb: [99, 102, 241] }
];

export function classColour(index: number): ClassColour {
  return CLASS_COLOURS[index] ?? { hex: '#94a3b8', rgb: [148, 163, 184] };
}

export const BACKGROUND_RGB: [number, number, number] = [248, 250, 252];

export const MESH_STROKE = 'rgba(15, 23, 42, 0.4)';
