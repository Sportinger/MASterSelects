// Output Slice system types for mapping input regions to warped output quads
// Used by the Output Manager for Resolume-style slice/warp functionality

export interface Point2D {
  x: number;
  y: number;
}

export interface SliceInputRect {
  x: number;       // top-left X (0-1 normalized)
  y: number;       // top-left Y (0-1 normalized)
  width: number;   // size (0-1)
  height: number;  // size (0-1)
}

export type WarpMode = 'cornerPin' | 'meshGrid';

export interface CornerPinWarp {
  mode: 'cornerPin';
  corners: [Point2D, Point2D, Point2D, Point2D]; // TL, TR, BR, BL (0-1 normalized)
}

export interface MeshGridWarp {
  mode: 'meshGrid';
  cols: number;
  rows: number;
  points: Point2D[]; // (cols+1)*(rows+1) points, row-major order
}

export type SliceWarp = CornerPinWarp | MeshGridWarp;

export interface OutputSlice {
  id: string;
  name: string;
  enabled: boolean;
  inputRect: SliceInputRect;
  warp: SliceWarp;
}

export interface TargetSliceConfig {
  targetId: string;
  slices: OutputSlice[];
  selectedSliceId: string | null;
}

// === Factory Functions ===

let sliceCounter = 0;

export function createDefaultSlice(name?: string): OutputSlice {
  const id = `slice_${Date.now()}_${++sliceCounter}`;
  return {
    id,
    name: name ?? `Slice ${sliceCounter}`,
    enabled: true,
    inputRect: { x: 0, y: 0, width: 1, height: 1 },
    warp: {
      mode: 'cornerPin',
      corners: [
        { x: 0, y: 0 }, // TL
        { x: 1, y: 0 }, // TR
        { x: 1, y: 1 }, // BR
        { x: 0, y: 1 }, // BL
      ],
    },
  };
}

export function createMeshGrid(cols: number, rows: number): MeshGridWarp {
  const points: Point2D[] = [];
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c <= cols; c++) {
      points.push({ x: c / cols, y: r / rows });
    }
  }
  return { mode: 'meshGrid', cols, rows, points };
}

export function cornerPinToMeshGrid(
  corners: [Point2D, Point2D, Point2D, Point2D],
  cols: number,
  rows: number
): MeshGridWarp {
  const [tl, tr, br, bl] = corners;
  const points: Point2D[] = [];
  for (let r = 0; r <= rows; r++) {
    const t = r / rows;
    for (let c = 0; c <= cols; c++) {
      const s = c / cols;
      // Bilinear interpolation of the 4 corners
      const x = (1 - s) * (1 - t) * tl.x + s * (1 - t) * tr.x + s * t * br.x + (1 - s) * t * bl.x;
      const y = (1 - s) * (1 - t) * tl.y + s * (1 - t) * tr.y + s * t * br.y + (1 - s) * t * bl.y;
      points.push({ x, y });
    }
  }
  return { mode: 'meshGrid', cols, rows, points };
}
