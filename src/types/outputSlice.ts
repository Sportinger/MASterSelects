// Output Slice system types for mapping input regions to warped output quads
// Used by the Output Manager for Resolume-style slice/warp functionality

export interface Point2D {
  x: number;
  y: number;
}

/** @deprecated Use inputCorners instead */
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

export type SliceItemType = 'slice' | 'mask';

export interface OutputSlice {
  id: string;
  name: string;
  type: SliceItemType;
  inverted: boolean;
  enabled: boolean;
  inputCorners: [Point2D, Point2D, Point2D, Point2D]; // TL, TR, BR, BL (0-1 normalized)
  warp: SliceWarp;
}

export interface TargetSliceConfig {
  targetId: string;
  slices: OutputSlice[];
  selectedSliceId: string | null;
}

// === Factory Functions ===

let sliceCounter = 0;

export const DEFAULT_CORNERS: [Point2D, Point2D, Point2D, Point2D] = [
  { x: 0, y: 0 }, // TL
  { x: 1, y: 0 }, // TR
  { x: 1, y: 1 }, // BR
  { x: 0, y: 1 }, // BL
];

export function createDefaultSlice(name?: string): OutputSlice {
  const id = `slice_${Date.now()}_${++sliceCounter}`;
  return {
    id,
    name: name ?? `Slice ${sliceCounter}`,
    type: 'slice',
    inverted: false,
    enabled: true,
    inputCorners: [
      { x: 0, y: 0 }, // TL
      { x: 1, y: 0 }, // TR
      { x: 1, y: 1 }, // BR
      { x: 0, y: 1 }, // BL
    ],
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

let maskCounter = 0;

export function createDefaultMask(name?: string): OutputSlice {
  const id = `mask_${Date.now()}_${++maskCounter}`;
  return {
    id,
    name: name ?? `Mask ${maskCounter}`,
    type: 'mask',
    inverted: true,
    enabled: true,
    inputCorners: [
      { x: 0.25, y: 0.25 },
      { x: 0.75, y: 0.25 },
      { x: 0.75, y: 0.75 },
      { x: 0.25, y: 0.75 },
    ],
    warp: {
      mode: 'cornerPin',
      corners: [
        { x: 0.25, y: 0.25 },
        { x: 0.75, y: 0.25 },
        { x: 0.75, y: 0.75 },
        { x: 0.25, y: 0.75 },
      ],
    },
  };
}

/** Migrate legacy slices that have inputRect instead of inputCorners, or missing type/inverted */
export function migrateSlice(slice: OutputSlice & { inputRect?: SliceInputRect }): OutputSlice {
  let result = slice;
  if (!result.inputCorners) {
    const r = slice.inputRect ?? { x: 0, y: 0, width: 1, height: 1 };
    result = {
      ...result,
      inputCorners: [
        { x: r.x, y: r.y },
        { x: r.x + r.width, y: r.y },
        { x: r.x + r.width, y: r.y + r.height },
        { x: r.x, y: r.y + r.height },
      ],
    };
  }
  // Add type/inverted for legacy data
  if (!result.type) {
    result = { ...result, type: 'slice' };
  }
  if (result.inverted === undefined) {
    result = { ...result, inverted: false };
  }
  return result;
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
