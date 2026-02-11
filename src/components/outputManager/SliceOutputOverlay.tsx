// SliceOutputOverlay - SVG overlay for dragging corner pin warp points
// Uses setPointerCapture + SVG getScreenCTM for correct coordinate mapping
// Right-click context menu: "Match Input Shape"

import { useCallback, useRef, useState } from 'react';
import { useSliceStore } from '../../stores/sliceStore';
import type { Point2D } from '../../types/outputSlice';

interface SliceOutputOverlayProps {
  targetId: string;
  width: number;
  height: number;
}

const SLICE_COLORS = ['#2D8CEB', '#EB8C2D', '#2DEB8C', '#EB2D8C', '#8C2DEB', '#8CEB2D'];
const MASK_COLOR = '#FF4444';
const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];
const POINT_RADIUS = 6;

/** Convert client (screen) coords to normalized 0-1 coords via SVG's CTM.
 *  Correctly handles preserveAspectRatio letterboxing. */
function clientToNormalized(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
  vbWidth: number,
  vbHeight: number
): Point2D | null {
  const ctm = svg.getScreenCTM();
  if (!ctm) return null;
  const inv = ctm.inverse();
  // Transform client → SVG viewBox coords
  const svgX = inv.a * clientX + inv.c * clientY + inv.e;
  const svgY = inv.b * clientX + inv.d * clientY + inv.f;
  return {
    x: svgX / vbWidth,
    y: svgY / vbHeight,
  };
}

interface DragState {
  sliceId: string;
  cornerIndex: number;
  pointerId: number;
  // Offset between pointer position and corner center (in normalized coords)
  offsetX: number;
  offsetY: number;
}

interface ContextMenu {
  x: number;
  y: number;
  sliceId: string;
}

export function SliceOutputOverlay({ targetId, width, height }: SliceOutputOverlayProps) {
  const config = useSliceStore((s) => s.configs.get(targetId));
  const selectSlice = useSliceStore((s) => s.selectSlice);
  const setCornerPinCorner = useSliceStore((s) => s.setCornerPinCorner);
  const matchOutputToInput = useSliceStore((s) => s.matchOutputToInput);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenu | null>(null);

  const handlePointerDown = useCallback((
    e: React.PointerEvent,
    sliceId: string,
    cornerIndex: number
  ) => {
    e.preventDefault();
    e.stopPropagation();

    selectSlice(targetId, sliceId);

    const svg = svgRef.current;
    if (!svg) return;

    // Get the current corner position to calculate click offset
    const currentConfig = useSliceStore.getState().configs.get(targetId);
    const slice = currentConfig?.slices.find((s) => s.id === sliceId);
    if (!slice || slice.warp.mode !== 'cornerPin') return;

    const cornerPos = slice.warp.corners[cornerIndex];
    const clickPos = clientToNormalized(svg, e.clientX, e.clientY, width, height);
    if (!clickPos) return;

    // Store offset so the corner doesn't jump to the pointer
    dragRef.current = {
      sliceId,
      cornerIndex,
      pointerId: e.pointerId,
      offsetX: cornerPos.x - clickPos.x,
      offsetY: cornerPos.y - clickPos.y,
    };
    svg.setPointerCapture(e.pointerId);
  }, [targetId, width, height, selectSlice]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || e.pointerId !== drag.pointerId) return;

    const svg = svgRef.current;
    if (!svg) return;

    const pos = clientToNormalized(svg, e.clientX, e.clientY, width, height);
    if (!pos) return;

    // Apply offset so corner stays under the pointer where it was grabbed
    const x = pos.x + drag.offsetX;
    const y = pos.y + drag.offsetY;

    setCornerPinCorner(targetId, drag.sliceId, drag.cornerIndex, { x, y });
  }, [targetId, width, height, setCornerPinCorner]);

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || e.pointerId !== drag.pointerId) return;

    const svg = svgRef.current;
    if (svg) svg.releasePointerCapture(drag.pointerId);
    dragRef.current = null;
  }, []);

  const handleContextMenu = useCallback((e: React.MouseEvent, sliceId: string) => {
    e.preventDefault();
    e.stopPropagation();
    selectSlice(targetId, sliceId);
    setContextMenu({ x: e.clientX, y: e.clientY, sliceId });
  }, [targetId, selectSlice]);

  const handleMatchInputShape = useCallback(() => {
    if (contextMenu) {
      matchOutputToInput(targetId, contextMenu.sliceId);
    }
    setContextMenu(null);
  }, [targetId, contextMenu, matchOutputToInput]);

  const handleCloseMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  if (!config || config.slices.length === 0) return null;

  const selectedSliceId = config.selectedSliceId;

  return (
    <>
      <svg
        ref={svgRef}
        className="om-slice-overlay"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        overflow="visible"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onLostPointerCapture={handlePointerUp}
        onClick={handleCloseMenu}
      >
        {config.slices.map((slice, idx) => {
          if (!slice.enabled || slice.warp.mode !== 'cornerPin') return null;

          const isMask = slice.type === 'mask';
          const color = isMask ? MASK_COLOR : SLICE_COLORS[idx % SLICE_COLORS.length];
          const isSelected = slice.id === selectedSliceId;
          const corners = slice.warp.corners;

          // Convert normalized coords to SVG viewBox coords
          const pts = corners.map((c: Point2D) => ({
            x: c.x * width,
            y: c.y * height,
          }));

          const pathData = `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y} L ${pts[2].x} ${pts[2].y} L ${pts[3].x} ${pts[3].y} Z`;

          return (
            <g
              key={slice.id}
              onClick={() => selectSlice(targetId, slice.id)}
              onContextMenu={(e) => handleContextMenu(e, slice.id)}
            >
              {/* Quad fill */}
              <path
                d={pathData}
                fill={isSelected ? `${color}15` : 'transparent'}
                stroke={color}
                strokeWidth={isSelected ? 2 : 1}
                strokeOpacity={isSelected ? 1 : 0.5}
                strokeDasharray={isMask ? '6 3' : 'none'}
              />

              {/* Corner points */}
              {pts.map((pt, ci) => (
                <g key={ci}>
                  {/* Larger invisible hit area */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={POINT_RADIUS * 2.5}
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
                  {/* Corner label (only for selected item) */}
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
                      {isMask ? 'mask' : CORNER_LABELS[ci]}
                    </text>
                  )}
                </g>
              ))}
            </g>
          );
        })}
      </svg>

      {/* Context menu */}
      {contextMenu && (
        <div
          className="om-context-menu-backdrop"
          onClick={handleCloseMenu}
          onContextMenu={(e) => { e.preventDefault(); handleCloseMenu(); }}
        >
          <div
            className="om-context-menu"
            style={{ left: contextMenu.x, top: contextMenu.y }}
            onClick={(e) => e.stopPropagation()}
          >
            <button className="om-context-menu-item" onClick={handleMatchInputShape}>
              Match Input Shape
            </button>
          </div>
        </div>
      )}
    </>
  );
}
