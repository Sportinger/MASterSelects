// SliceInputOverlay - SVG overlay for dragging input corner points
// Uses setPointerCapture + SVG getScreenCTM for correct coordinate mapping
// Right-click context menu: "Match Output Shape"

import { useCallback, useRef, useState } from 'react';
import { useSliceStore } from '../../stores/sliceStore';
import type { Point2D } from '../../types/outputSlice';

interface SliceInputOverlayProps {
  targetId: string;
  width: number;
  height: number;
}

const SLICE_COLORS = ['#2D8CEB', '#EB8C2D', '#2DEB8C', '#EB2D8C', '#8C2DEB', '#8CEB2D'];
const MASK_COLOR = '#FF4444';
const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];
const POINT_RADIUS = 6;

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
  offsetX: number;
  offsetY: number;
}

interface ContextMenu {
  x: number;
  y: number;
  sliceId: string;
}

export function SliceInputOverlay({ targetId, width, height }: SliceInputOverlayProps) {
  const config = useSliceStore((s) => s.configs.get(targetId));
  const selectSlice = useSliceStore((s) => s.selectSlice);
  const setInputCorner = useSliceStore((s) => s.setInputCorner);
  const matchInputToOutput = useSliceStore((s) => s.matchInputToOutput);
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

    const currentConfig = useSliceStore.getState().configs.get(targetId);
    const slice = currentConfig?.slices.find((s) => s.id === sliceId);
    if (!slice) return;

    const cornerPos = slice.inputCorners[cornerIndex];
    const clickPos = clientToNormalized(svg, e.clientX, e.clientY, width, height);
    if (!clickPos) return;

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

    // Clamp input corners to 0-1 range (must stay within source area)
    const x = Math.max(0, Math.min(1, pos.x + drag.offsetX));
    const y = Math.max(0, Math.min(1, pos.y + drag.offsetY));

    setInputCorner(targetId, drag.sliceId, drag.cornerIndex, { x, y });
  }, [targetId, width, height, setInputCorner]);

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

  const handleMatchOutputShape = useCallback(() => {
    if (contextMenu) {
      matchInputToOutput(targetId, contextMenu.sliceId);
    }
    setContextMenu(null);
  }, [targetId, contextMenu, matchInputToOutput]);

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
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onLostPointerCapture={handlePointerUp}
        onClick={handleCloseMenu}
      >
        {config.slices.map((slice, idx) => {
          if (!slice.enabled) return null;

          const isMask = slice.type === 'mask';
          const color = isMask ? MASK_COLOR : SLICE_COLORS[idx % SLICE_COLORS.length];
          const isSelected = slice.id === selectedSliceId;
          const corners = slice.inputCorners;

          const pts = corners.map((c: Point2D) => ({
            x: c.x * width,
            y: c.y * height,
          }));

          const pathData = `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y} L ${pts[2].x} ${pts[2].y} L ${pts[3].x} ${pts[3].y} Z`;

          // Masks are non-interactive reference shapes in the input view
          if (isMask) {
            return (
              <g key={slice.id} style={{ pointerEvents: 'none' }}>
                <path
                  d={pathData}
                  fill="transparent"
                  stroke={color}
                  strokeWidth={1}
                  strokeOpacity={0.3}
                  strokeDasharray="6 3"
                />
                {/* Label at center */}
                {isSelected && (
                  <text
                    x={(pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4}
                    y={(pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4}
                    textAnchor="middle"
                    fill={color}
                    fontSize={10}
                    fontWeight="bold"
                    opacity={0.5}
                  >
                    mask
                  </text>
                )}
              </g>
            );
          }

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
                strokeDasharray={isSelected ? 'none' : '6 3'}
              />

              {/* Corner points */}
              {pts.map((pt, ci) => (
                <g key={ci}>
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={POINT_RADIUS * 2.5}
                    fill="transparent"
                    style={{ cursor: 'grab' }}
                    onPointerDown={(e) => handlePointerDown(e, slice.id, ci)}
                  />
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isSelected ? POINT_RADIUS : POINT_RADIUS - 1}
                    fill={isSelected ? color : `${color}88`}
                    stroke={isSelected ? '#fff' : color}
                    strokeWidth={isSelected ? 2 : 1}
                    style={{ cursor: 'grab', pointerEvents: 'none' }}
                  />
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

      {/* Context menu (rendered as HTML overlay, uses client coords) */}
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
            <button className="om-context-menu-item" onClick={handleMatchOutputShape}>
              Match Output Shape
            </button>
          </div>
        </div>
      )}
    </>
  );
}
