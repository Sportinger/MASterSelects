// SliceOutputOverlay - SVG overlay for dragging corner pin warp points
// Renders 4 draggable corner points per slice + quad outlines

import { useCallback, useRef } from 'react';
import { useSliceStore } from '../../stores/sliceStore';
import type { Point2D } from '../../types/outputSlice';

interface SliceOutputOverlayProps {
  targetId: string;
  width: number;
  height: number;
}

const SLICE_COLORS = ['#2D8CEB', '#EB8C2D', '#2DEB8C', '#EB2D8C', '#8C2DEB', '#8CEB2D'];
const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];
const POINT_RADIUS = 6;
const THROTTLE_MS = 16;

export function SliceOutputOverlay({ targetId, width, height }: SliceOutputOverlayProps) {
  const config = useSliceStore((s) => s.configs.get(targetId));
  const selectSlice = useSliceStore((s) => s.selectSlice);
  const setCornerPinCorner = useSliceStore((s) => s.setCornerPinCorner);
  const svgRef = useRef<SVGSVGElement>(null);
  const lastUpdateRef = useRef(0);

  const handlePointerDown = useCallback((
    e: React.PointerEvent,
    sliceId: string,
    cornerIndex: number
  ) => {
    e.preventDefault();
    e.stopPropagation();

    // Select the slice on interaction
    selectSlice(targetId, sliceId);

    const svg = svgRef.current;
    if (!svg) return;

    const handleMove = (moveEvent: PointerEvent) => {
      const now = performance.now();
      if (now - lastUpdateRef.current < THROTTLE_MS) return;
      lastUpdateRef.current = now;

      const rect = svg.getBoundingClientRect();
      // Calculate position relative to SVG viewBox
      const x = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (moveEvent.clientY - rect.top) / rect.height));

      setCornerPinCorner(targetId, sliceId, cornerIndex, { x, y });
    };

    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  }, [targetId, selectSlice, setCornerPinCorner]);

  if (!config || config.slices.length === 0) return null;

  const selectedSliceId = config.selectedSliceId;

  return (
    <svg
      ref={svgRef}
      className="om-slice-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {config.slices.map((slice, idx) => {
        if (!slice.enabled || slice.warp.mode !== 'cornerPin') return null;

        const color = SLICE_COLORS[idx % SLICE_COLORS.length];
        const isSelected = slice.id === selectedSliceId;
        const corners = slice.warp.corners;

        // Convert normalized coords to SVG coords
        const pts = corners.map((c: Point2D) => ({
          x: c.x * width,
          y: c.y * height,
        }));

        const pathData = `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y} L ${pts[2].x} ${pts[2].y} L ${pts[3].x} ${pts[3].y} Z`;

        return (
          <g key={slice.id} onClick={() => selectSlice(targetId, slice.id)}>
            {/* Quad fill */}
            <path
              d={pathData}
              fill={isSelected ? `${color}15` : 'transparent'}
              stroke={color}
              strokeWidth={isSelected ? 2 : 1}
              strokeOpacity={isSelected ? 1 : 0.5}
            />

            {/* Corner points */}
            {pts.map((pt, ci) => (
              <g key={ci}>
                {/* Larger invisible hit area */}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={POINT_RADIUS * 2}
                  fill="transparent"
                  style={{ cursor: 'grab' }}
                  onPointerDown={(e) => handlePointerDown(e, slice.id, ci)}
                />
                {/* Visible point */}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isSelected ? POINT_RADIUS : POINT_RADIUS - 1}
                  fill={isSelected ? color : `${color}88`}
                  stroke={isSelected ? '#fff' : color}
                  strokeWidth={isSelected ? 2 : 1}
                  style={{ cursor: 'grab', pointerEvents: 'none' }}
                />
                {/* Corner label (only for selected slice) */}
                {isSelected && (
                  <text
                    x={pt.x}
                    y={pt.y - POINT_RADIUS - 4}
                    textAnchor="middle"
                    fill={color}
                    fontSize={10}
                    fontWeight="bold"
                    style={{ pointerEvents: 'none' }}
                  >
                    {CORNER_LABELS[ci]}
                  </text>
                )}
              </g>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
