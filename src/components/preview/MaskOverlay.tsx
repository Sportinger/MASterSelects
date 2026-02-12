// MaskOverlay - SVG overlay for mask drawing and editing on preview canvas

import { useRef, useCallback, useEffect, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import { generatePathData, normalizedToCanvas } from './maskUtils';
import { useMaskVertexDrag } from './useMaskVertexDrag';
import { useMaskDrag } from './useMaskDrag';
import { useMaskEdgeDrag } from './useMaskEdgeDrag';
import { useMaskShapeDraw } from './useMaskShapeDraw';

interface MaskOverlayProps {
  canvasWidth: number;
  canvasHeight: number;
}

export function MaskOverlay({ canvasWidth, canvasHeight }: MaskOverlayProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const {
    clips,
    selectedClipIds,
    maskEditMode,
    activeMaskId,
    selectedVertexIds,
    setMaskEditMode,
    deselectAllVertices,
    addVertex,
    closeMask,
    addMask,
    setActiveMask,
  } = useTimelineStore();

  // Get first selected clip for mask editing
  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find(c => c.id === selectedClipId);
  const activeMask = selectedClip?.masks?.find(m => m.id === activeMaskId);

  // Extracted hooks
  const { handleVertexMouseDown } = useMaskVertexDrag(svgRef, canvasWidth, canvasHeight, selectedClip, activeMask);
  const { handleMaskDragStart } = useMaskDrag(svgRef, canvasWidth, canvasHeight, selectedClip, activeMask);
  const { handleEdgeMouseDown } = useMaskEdgeDrag(svgRef, canvasWidth, canvasHeight, selectedClip, activeMask);
  const { shapeDrawState, justFinishedDrawing, handleShapeMouseDown, handleShapeMouseMove, handleShapeMouseUp } =
    useMaskShapeDraw(svgRef, selectedClip, maskEditMode);

  // Convert mask vertices to canvas coordinates for rendering
  const canvasVertices = useMemo(() => {
    if (!activeMask) return [];
    const posX = activeMask.position?.x || 0;
    const posY = activeMask.position?.y || 0;

    return activeMask.vertices.map(v => ({
      ...v,
      ...normalizedToCanvas(v.x + posX, v.y + posY, canvasWidth, canvasHeight),
      handleIn: normalizedToCanvas(v.handleIn.x, v.handleIn.y, canvasWidth, canvasHeight),
      handleOut: normalizedToCanvas(v.handleOut.x, v.handleOut.y, canvasWidth, canvasHeight),
    }));
  }, [activeMask, canvasWidth, canvasHeight]);

  // Generate path data for the active mask
  const pathData = useMemo(() => {
    if (!activeMask) return '';
    return generatePathData(
      activeMask.vertices,
      activeMask.closed,
      activeMask.position?.x || 0,
      activeMask.position?.y || 0,
      canvasWidth,
      canvasHeight
    );
  }, [activeMask, canvasWidth, canvasHeight]);

  // Generate individual edge path segments for hit testing
  const edgeSegments = useMemo(() => {
    if (!activeMask || !activeMask.visible || activeMask.vertices.length < 2) return [];
    const verts = activeMask.vertices;
    const posX = activeMask.position?.x || 0;
    const posY = activeMask.position?.y || 0;
    const segments: Array<{ d: string; idA: string; idB: string }> = [];

    for (let i = 1; i < verts.length; i++) {
      const prev = verts[i - 1];
      const curr = verts[i];
      const prevX = (prev.x + posX) * canvasWidth;
      const prevY = (prev.y + posY) * canvasHeight;
      const currX = (curr.x + posX) * canvasWidth;
      const currY = (curr.y + posY) * canvasHeight;
      const cp1x = prevX + prev.handleOut.x * canvasWidth;
      const cp1y = prevY + prev.handleOut.y * canvasHeight;
      const cp2x = currX + curr.handleIn.x * canvasWidth;
      const cp2y = currY + curr.handleIn.y * canvasHeight;
      segments.push({
        d: `M ${prevX} ${prevY} C ${cp1x},${cp1y} ${cp2x},${cp2y} ${currX},${currY}`,
        idA: prev.id,
        idB: curr.id,
      });
    }

    if (activeMask.closed && verts.length > 2) {
      const last = verts[verts.length - 1];
      const first = verts[0];
      const lastX = (last.x + posX) * canvasWidth;
      const lastY = (last.y + posY) * canvasHeight;
      const firstX = (first.x + posX) * canvasWidth;
      const firstY = (first.y + posY) * canvasHeight;
      const cp1x = lastX + last.handleOut.x * canvasWidth;
      const cp1y = lastY + last.handleOut.y * canvasHeight;
      const cp2x = firstX + first.handleIn.x * canvasWidth;
      const cp2y = firstY + first.handleIn.y * canvasHeight;
      segments.push({
        d: `M ${lastX} ${lastY} C ${cp1x},${cp1y} ${cp2x},${cp2y} ${firstX},${firstY}`,
        idA: last.id,
        idB: first.id,
      });
    }

    return segments;
  }, [activeMask, canvasWidth, canvasHeight]);

  // Handle clicking on SVG background
  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!selectedClip) return;

    if (justFinishedDrawing.current) {
      justFinishedDrawing.current = false;
      return;
    }

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width);
    const y = ((e.clientY - rect.top) / rect.height);

    if (maskEditMode === 'drawing' && activeMask) {
      addVertex(selectedClip.id, activeMask.id, {
        x,
        y,
        handleIn: { x: 0, y: 0 },
        handleOut: { x: 0, y: 0 },
      });
    } else if (maskEditMode === 'drawingPen') {
      if (activeMask) {
        addVertex(selectedClip.id, activeMask.id, {
          x,
          y,
          handleIn: { x: 0, y: 0 },
          handleOut: { x: 0, y: 0 },
        });
      } else {
        const maskId = addMask(selectedClip.id, { name: 'Pen Mask' });
        setActiveMask(selectedClip.id, maskId);
        addVertex(selectedClip.id, maskId, {
          x,
          y,
          handleIn: { x: 0, y: 0 },
          handleOut: { x: 0, y: 0 },
        });
        setMaskEditMode('drawing');
      }
    } else if (maskEditMode === 'editing' && activeMask) {
      deselectAllVertices();
    }
  }, [selectedClip, activeMask, maskEditMode, addVertex, addMask, deselectAllVertices, setMaskEditMode]);

  // Handle clicking on first vertex to close path
  const handleFirstVertexClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!selectedClip || !activeMask) return;

    if (maskEditMode === 'drawing' && activeMask.vertices.length >= 3) {
      closeMask(selectedClip.id, activeMask.id);
      setMaskEditMode('editing');
    }
  }, [selectedClip, activeMask, maskEditMode, closeMask, setMaskEditMode]);

  // Handle escape key to exit drawing mode + delete selected vertices
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (shapeDrawState.isDrawing) {
          handleShapeMouseUp();
        } else if (maskEditMode === 'drawing' || maskEditMode === 'drawingRect' ||
                   maskEditMode === 'drawingEllipse' || maskEditMode === 'drawingPen') {
          setMaskEditMode('none');
        } else if (maskEditMode === 'editing') {
          setMaskEditMode('none');
        }
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && maskEditMode === 'editing') {
        if (selectedVertexIds.size > 0 && selectedClip && activeMask) {
          const { removeVertex } = useTimelineStore.getState();
          selectedVertexIds.forEach(vertexId => {
            removeVertex(selectedClip.id, activeMask.id, vertexId);
          });
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [maskEditMode, setMaskEditMode, selectedVertexIds, selectedClip, activeMask, shapeDrawState.isDrawing, handleShapeMouseUp]);

  // Don't render if not in mask editing mode
  const isShapeDrawingMode = maskEditMode === 'drawingRect' || maskEditMode === 'drawingEllipse' || maskEditMode === 'drawingPen';
  if (maskEditMode === 'none' || !selectedClip) {
    return null;
  }
  if (!isShapeDrawingMode && !activeMask) {
    return null;
  }

  const vertexSize = 8;
  const handleSize = 6;

  const getCursor = () => {
    if (maskEditMode === 'drawingRect' || maskEditMode === 'drawingEllipse' || maskEditMode === 'drawingPen') {
      return 'crosshair';
    }
    if (maskEditMode === 'drawing') return 'crosshair';
    return 'default';
  };

  return (
    <svg
      ref={svgRef}
      className="mask-overlay-svg"
      viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
      preserveAspectRatio="xMidYMid meet"
      onClick={handleSvgClick}
      onMouseDown={handleShapeMouseDown}
      onMouseMove={handleShapeMouseMove}
      onMouseUp={handleShapeMouseUp}
      onMouseLeave={handleShapeMouseUp}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'auto',
        cursor: getCursor(),
      }}
    >
      {/* Shape preview while drawing */}
      {shapeDrawState.isDrawing && (
        <>
          {maskEditMode === 'drawingRect' && (
            <rect
              x={Math.min(shapeDrawState.startX, shapeDrawState.currentX) * canvasWidth}
              y={Math.min(shapeDrawState.startY, shapeDrawState.currentY) * canvasHeight}
              width={Math.abs(shapeDrawState.currentX - shapeDrawState.startX) * canvasWidth}
              height={Math.abs(shapeDrawState.currentY - shapeDrawState.startY) * canvasHeight}
              fill="rgba(45, 140, 235, 0.15)"
              stroke="#2997E5"
              strokeWidth="2"
              strokeDasharray="5,5"
              pointerEvents="none"
            />
          )}
          {maskEditMode === 'drawingEllipse' && (
            <ellipse
              cx={(shapeDrawState.startX + shapeDrawState.currentX) / 2 * canvasWidth}
              cy={(shapeDrawState.startY + shapeDrawState.currentY) / 2 * canvasHeight}
              rx={Math.abs(shapeDrawState.currentX - shapeDrawState.startX) / 2 * canvasWidth}
              ry={Math.abs(shapeDrawState.currentY - shapeDrawState.startY) / 2 * canvasHeight}
              fill="rgba(45, 140, 235, 0.15)"
              stroke="#2997E5"
              strokeWidth="2"
              strokeDasharray="5,5"
              pointerEvents="none"
            />
          )}
        </>
      )}

      {/* Mask path fill - clickable for dragging when visible */}
      {activeMask?.closed && activeMask.visible && pathData && (
        <path
          d={pathData}
          fill={activeMask.inverted ? 'rgba(45, 140, 235, 0.1)' : 'rgba(45, 140, 235, 0.15)'}
          stroke="none"
          pointerEvents="all"
          cursor="move"
          onMouseDown={handleMaskDragStart}
        />
      )}

      {/* Mask path stroke - only when visible */}
      {activeMask && activeMask.visible && pathData && (
        <path
          d={pathData}
          fill="none"
          stroke="#2997E5"
          strokeWidth="2"
          strokeDasharray={activeMask.closed ? 'none' : '5,5'}
          pointerEvents="none"
        />
      )}

      {/* Edge hit areas */}
      {maskEditMode === 'editing' && edgeSegments.map((seg) => (
        <path
          key={`edge-${seg.idA}-${seg.idB}`}
          d={seg.d}
          fill="none"
          stroke="transparent"
          strokeWidth="12"
          cursor="move"
          pointerEvents="stroke"
          onMouseDown={(e) => handleEdgeMouseDown(e, seg.idA, seg.idB)}
        />
      ))}

      {/* Bezier control handles */}
      {activeMask?.visible && canvasVertices.map((vertex) => {
        const isSelected = selectedVertexIds.has(vertex.id);
        if (!isSelected) return null;

        return (
          <g key={`handles-${vertex.id}`}>
            <line
              x1={vertex.x}
              y1={vertex.y}
              x2={vertex.x + vertex.handleIn.x}
              y2={vertex.y + vertex.handleIn.y}
              stroke="#ff9900"
              strokeWidth="1"
              pointerEvents="none"
            />
            <circle
              cx={vertex.x + vertex.handleIn.x}
              cy={vertex.y + vertex.handleIn.y}
              r={handleSize / 2}
              fill="#ff9900"
              stroke="#fff"
              strokeWidth="1"
              cursor="move"
              onMouseDown={(e) => handleVertexMouseDown(e, vertex.id, 'handleIn')}
            />

            <line
              x1={vertex.x}
              y1={vertex.y}
              x2={vertex.x + vertex.handleOut.x}
              y2={vertex.y + vertex.handleOut.y}
              stroke="#ff9900"
              strokeWidth="1"
              pointerEvents="none"
            />
            <circle
              cx={vertex.x + vertex.handleOut.x}
              cy={vertex.y + vertex.handleOut.y}
              r={handleSize / 2}
              fill="#ff9900"
              stroke="#fff"
              strokeWidth="1"
              cursor="move"
              onMouseDown={(e) => handleVertexMouseDown(e, vertex.id, 'handleOut')}
            />
          </g>
        );
      })}

      {/* Vertex points */}
      {activeMask?.visible && canvasVertices.map((vertex, index) => {
        const isSelected = selectedVertexIds.has(vertex.id);
        const isFirst = index === 0;

        return (
          <rect
            key={vertex.id}
            x={vertex.x - vertexSize / 2}
            y={vertex.y - vertexSize / 2}
            width={vertexSize}
            height={vertexSize}
            fill={isSelected ? '#2997E5' : '#fff'}
            stroke={isFirst && maskEditMode === 'drawing' ? '#ff0000' : '#2997E5'}
            strokeWidth={isFirst && maskEditMode === 'drawing' ? '2' : '1'}
            cursor="move"
            onClick={isFirst && maskEditMode === 'drawing' ? handleFirstVertexClick : undefined}
            onMouseDown={(e) => handleVertexMouseDown(e, vertex.id, 'vertex')}
          />
        );
      })}

      {/* Instructions */}
      <text
        x="10"
        y="20"
        fill="#fff"
        fontSize="12"
        fontFamily="sans-serif"
        pointerEvents="none"
      >
        {maskEditMode === 'drawingRect' && 'Click and drag to draw rectangle. ESC to cancel.'}
        {maskEditMode === 'drawingEllipse' && 'Click and drag to draw ellipse. ESC to cancel.'}
        {maskEditMode === 'drawingPen' && 'Click to add points. Click first point to close. ESC to cancel.'}
        {maskEditMode === 'drawing' && 'Click to add points. Click first point to close. ESC to cancel.'}
        {maskEditMode === 'editing' && 'Drag vertices to move. Del to delete. ESC to exit.'}
      </text>

      {/* Debug info */}
      <text
        x="10"
        y="40"
        fill="#ff0"
        fontSize="10"
        fontFamily="monospace"
        pointerEvents="none"
      >
        Canvas: {canvasWidth}x{canvasHeight} (AR: {(canvasWidth/canvasHeight).toFixed(2)})
      </text>
    </svg>
  );
}
