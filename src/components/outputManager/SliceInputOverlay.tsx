// SliceInputOverlay - SVG overlay showing input rects for each slice
// Phase 1: Display only (no dragging). Phase 2 will add resize/reposition.

import { useSliceStore } from '../../stores/sliceStore';

interface SliceInputOverlayProps {
  targetId: string;
  width: number;
  height: number;
}

const SLICE_COLORS = ['#2D8CEB', '#EB8C2D', '#2DEB8C', '#EB2D8C', '#8C2DEB', '#8CEB2D'];

export function SliceInputOverlay({ targetId, width, height }: SliceInputOverlayProps) {
  const config = useSliceStore((s) => s.configs.get(targetId));
  const selectSlice = useSliceStore((s) => s.selectSlice);

  if (!config || config.slices.length === 0) return null;

  const selectedSliceId = config.selectedSliceId;

  return (
    <svg
      className="om-slice-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {config.slices.map((slice, idx) => {
        if (!slice.enabled) return null;
        const color = SLICE_COLORS[idx % SLICE_COLORS.length];
        const isSelected = slice.id === selectedSliceId;
        const r = slice.inputRect;

        return (
          <rect
            key={slice.id}
            x={r.x * width}
            y={r.y * height}
            width={r.width * width}
            height={r.height * height}
            fill={isSelected ? `${color}22` : `${color}11`}
            stroke={color}
            strokeWidth={isSelected ? 2 : 1}
            strokeDasharray={isSelected ? 'none' : '4 2'}
            style={{ cursor: 'pointer' }}
            onClick={() => selectSlice(targetId, slice.id)}
          />
        );
      })}
    </svg>
  );
}
