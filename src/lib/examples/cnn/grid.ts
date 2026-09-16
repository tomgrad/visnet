export interface GridPrediction {
  predicted: number;
  correct: boolean;
}

export function predictionsFromLogits(
  logits: number[][],
  labels: Uint8Array,
  indices: number[]
): GridPrediction[] {
  return logits.map((row, position) => {
    let predicted = 0;
    for (let candidate = 1; candidate < row.length; candidate++) {
      if (row[candidate] > row[predicted]) predicted = candidate;
    }
    return { predicted, correct: predicted === labels[indices[position]] };
  });
}

export function gridCellRect(
  cell: number,
  columns: number,
  cellWidth: number,
  cellHeight: number,
  gap: number
): { x: number; y: number; width: number; height: number } {
  return {
    x: (cell % columns) * (cellWidth + gap),
    y: Math.floor(cell / columns) * (cellHeight + gap),
    width: cellWidth,
    height: cellHeight
  };
}
