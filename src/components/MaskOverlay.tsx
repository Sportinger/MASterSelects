// MaskOverlay - SVG overlay for mask drawing and editing on preview canvas

import { useRef, useCallback, useEffect, useMemo, useState } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import type { MaskVertex } from '../types';

// Shape drawing state
interface ShapeDrawState {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
  isDrawing: boolean;
}

interface MaskOverlayProps {
  canvasWidth: number;
  canvasHeight: number;
}

// Generate SVG path data from mask vertices using cubic bezier curves
function generatePathData(vertices: MaskVertex[], closed: boolean): string {
  if (vertices.length < 2) return '';

  let d = '';

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];

    if (i === 0) {
      d += `M ${v.x} ${v.y}`;
    } else {
      const prev = vertices[i - 1];
      // Cubic bezier: C cp1x,cp1y cp2x,cp2y x,y
      const cp1x = prev.x + prev.handleOut.x;
      const cp1y = prev.y + prev.handleOut.y;
      const cp2x = v.x + v.handleIn.x;
      const cp2y = v.y + v.handleIn.y;
      d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${v.x},${v.y}`;
    }
  }

  if (closed && vertices.length > 2) {
    const last = vertices[vertices.length - 1];
    const first = vertices[0];
    const cp1x = last.x + last.handleOut.x;
    const cp1y = last.y + last.handleOut.y;
    const cp2x = first.x + first.handleIn.x;
    const cp2y = first.y + first.handleIn.y;
    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${first.x},${first.y} Z`;
  }

  return d;
}

// Convert normalized (0-1) coordinates to canvas coordinates
function normalizedToCanvas(
  x: number,
  y: number,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number } {
  return {
    x: x * canvasWidth,
    y: y * canvasHeight,
  };
}

export function MaskOverlay({ canvasWidth, canvasHeight }: MaskOverlayProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const {
    clips,
    selectedClipId,
    maskEditMode,
    activeMaskId,
    selectedVertexIds,
    setMaskEditMode,
    selectVertex,
    deselectAllVertices,
    updateVertex,
    addVertex,
    closeMask,
    addMask,
  } = useTimelineStore();

  const selectedClip = clips.find(c => c.id === selectedClipId);
  const activeMask = selectedClip?.masks?.find(m => m.id === activeMaskId);

  // Shape drawing state for drag-to-draw modes
  const [shapeDrawState, setShapeDrawState] = useState<ShapeDrawState>({
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
    isDrawing: false,
  });

  // Drag state
  const dragState = useRef<{
    vertexId: string | null;
    handleType: 'vertex' | 'handleIn' | 'handleOut' | null;
    startX: number;
    startY: number;
    startVertexX: number;
    startVertexY: number;
    startHandleX: number;
    startHandleY: number;
  }>({
    vertexId: null,
    handleType: null,
    startX: 0,
    startY: 0,
    startVertexX: 0,
    startVertexY: 0,
    startHandleX: 0,
    startHandleY: 0,
  });

  // Convert mask vertices to canvas coordinates for rendering
  const canvasVertices = useMemo(() => {
    if (!activeMask) return [];
    return activeMask.vertices.map(v => ({
      ...v,
      ...normalizedToCanvas(v.x, v.y, canvasWidth, canvasHeight),
      handleIn: normalizedToCanvas(v.handleIn.x, v.handleIn.y, canvasWidth, canvasHeight),
      handleOut: normalizedToCanvas(v.handleOut.x, v.handleOut.y, canvasWidth, canvasHeight),
    }));
  }, [activeMask, canvasWidth, canvasHeight]);

  // Generate path data for the active mask
  const pathData = useMemo(() => {
    if (!activeMask || canvasVertices.length === 0) return '';
    return generatePathData(canvasVertices, activeMask.closed);
  }, [canvasVertices, activeMask]);

  // Handle vertex drag start
  const handleVertexMouseDown = useCallback((
    e: React.MouseEvent,
    vertexId: string,
    handleType: 'vertex' | 'handleIn' | 'handleOut'
  ) => {
    e.stopPropagation();
    e.preventDefault();

    if (!activeMask || !selectedClip) return;

    const vertex = activeMask.vertices.find(v => v.id === vertexId);
    if (!vertex) return;

    // Select vertex
    selectVertex(vertexId, e.shiftKey);

    // Start drag
    dragState.current = {
      vertexId,
      handleType,
      startX: e.clientX,
      startY: e.clientY,
      startVertexX: vertex.x,
      startVertexY: vertex.y,
      startHandleX: handleType === 'handleIn' ? vertex.handleIn.x : vertex.handleOut.x,
      startHandleY: handleType === 'handleIn' ? vertex.handleIn.y : vertex.handleOut.y,
    };

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragState.current.vertexId || !dragState.current.handleType) return;
      if (!selectedClip || !activeMask) return;

      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const scaleX = canvasWidth / rect.width;
      const scaleY = canvasHeight / rect.height;

      const dx = (moveEvent.clientX - dragState.current.startX) * scaleX;
      const dy = (moveEvent.clientY - dragState.current.startY) * scaleY;

      // Convert delta to normalized coordinates
      const normalizedDx = dx / canvasWidth;
      const normalizedDy = dy / canvasHeight;

      if (dragState.current.handleType === 'vertex') {
        // Move vertex position
        const newX = Math.max(0, Math.min(1, dragState.current.startVertexX + normalizedDx));
        const newY = Math.max(0, Math.min(1, dragState.current.startVertexY + normalizedDy));
        updateVertex(selectedClip.id, activeMask.id, dragState.current.vertexId, {
          x: newX,
          y: newY,
        });
      } else {
        // Move bezier handle
        const handleKey = dragState.current.handleType;
        updateVertex(selectedClip.id, activeMask.id, dragState.current.vertexId, {
          [handleKey]: {
            x: dragState.current.startHandleX + normalizedDx,
            y: dragState.current.startHandleY + normalizedDy,
          },
        });
      }
    };

    const handleMouseUp = () => {
      dragState.current = {
        vertexId: null,
        handleType: null,
        startX: 0,
        startY: 0,
        startVertexX: 0,
        startVertexY: 0,
        startHandleX: 0,
        startHandleY: 0,
      };
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [activeMask, selectedClip, selectVertex, updateVertex, canvasWidth, canvasHeight]);

  // Handle clicking on SVG background (add vertex in drawing mode, deselect in editing mode)
  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!selectedClip) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width);
    const y = ((e.clientY - rect.top) / rect.height);

    if (maskEditMode === 'drawing' && activeMask) {
      // Add new vertex at click position
      addVertex(selectedClip.id, activeMask.id, {
        x,
        y,
        handleIn: { x: 0, y: 0 },
        handleOut: { x: 0, y: 0 },
      });
    } else if (maskEditMode === 'drawingPen') {
      // Pen tool - create new mask and add first vertex, or add vertex to active mask
      if (activeMask) {
        addVertex(selectedClip.id, activeMask.id, {
          x,
          y,
          handleIn: { x: 0, y: 0 },
          handleOut: { x: 0, y: 0 },
        });
      } else {
        // Create a new mask and switch to drawing mode
        const maskId = addMask(selectedClip.id, { name: 'Pen Mask' });
        useTimelineStore.getState().setActiveMask(selectedClip.id, maskId);
        addVertex(selectedClip.id, maskId, {
          x,
          y,
          handleIn: { x: 0, y: 0 },
          handleOut: { x: 0, y: 0 },
        });
        setMaskEditMode('drawing');
      }
    } else if (maskEditMode === 'editing' && activeMask) {
      // Deselect all vertices when clicking on empty space
      deselectAllVertices();
    }
  }, [selectedClip, activeMask, maskEditMode, addVertex, addMask, deselectAllVertices, setMaskEditMode]);

  // Handle mouse down for shape drawing (rectangle/ellipse)
  const handleShapeMouseDown = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!selectedClip) return;
    if (maskEditMode !== 'drawingRect' && maskEditMode !== 'drawingEllipse') return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    setShapeDrawState({
      startX: x,
      startY: y,
      currentX: x,
      currentY: y,
      isDrawing: true,
    });

    e.preventDefault();
  }, [selectedClip, maskEditMode]);

  // Handle mouse move for shape drawing
  const handleShapeMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!shapeDrawState.isDrawing) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    setShapeDrawState(prev => ({
      ...prev,
      currentX: x,
      currentY: y,
    }));
  }, [shapeDrawState.isDrawing]);

  // Handle mouse up for shape drawing - create the mask
  const handleShapeMouseUp = useCallback(() => {
    if (!shapeDrawState.isDrawing || !selectedClip) {
      setShapeDrawState(prev => ({ ...prev, isDrawing: false }));
      return;
    }

    const { startX, startY, currentX, currentY } = shapeDrawState;
    const minX = Math.min(startX, currentX);
    const maxX = Math.max(startX, currentX);
    const minY = Math.min(startY, currentY);
    const maxY = Math.max(startY, currentY);

    // Minimum size check
    if (maxX - minX < 0.01 || maxY - minY < 0.01) {
      setShapeDrawState(prev => ({ ...prev, isDrawing: false }));
      return;
    }

    let maskId: string;
    let vertices: MaskVertex[];

    if (maskEditMode === 'drawingRect') {
      // Create rectangle vertices
      maskId = addMask(selectedClip.id, { name: 'Rectangle Mask' });
      vertices = [
        { id: `v-${Date.now()}-1`, x: minX, y: minY, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
        { id: `v-${Date.now()}-2`, x: maxX, y: minY, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
        { id: `v-${Date.now()}-3`, x: maxX, y: maxY, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
        { id: `v-${Date.now()}-4`, x: minX, y: maxY, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
      ];
    } else {
      // Create ellipse vertices with bezier handles
      maskId = addMask(selectedClip.id, { name: 'Ellipse Mask' });
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const rx = (maxX - minX) / 2;
      const ry = (maxY - minY) / 2;
      const k = 0.5523; // Bezier approximation constant

      vertices = [
        { id: `v-${Date.now()}-1`, x: cx, y: minY, handleIn: { x: -rx * k, y: 0 }, handleOut: { x: rx * k, y: 0 } },
        { id: `v-${Date.now()}-2`, x: maxX, y: cy, handleIn: { x: 0, y: -ry * k }, handleOut: { x: 0, y: ry * k } },
        { id: `v-${Date.now()}-3`, x: cx, y: maxY, handleIn: { x: rx * k, y: 0 }, handleOut: { x: -rx * k, y: 0 } },
        { id: `v-${Date.now()}-4`, x: minX, y: cy, handleIn: { x: 0, y: ry * k }, handleOut: { x: 0, y: -ry * k } },
      ];
    }

    // Update the mask with vertices and close it
    const { clips } = useTimelineStore.getState();
    const updatedClips = clips.map(c => {
      if (c.id !== selectedClip.id) return c;
      return {
        ...c,
        masks: (c.masks || []).map(m =>
          m.id === maskId ? { ...m, vertices, closed: true } : m
        ),
      };
    });
    useTimelineStore.setState({ clips: updatedClips });
    useTimelineStore.getState().setActiveMask(selectedClip.id, maskId);
    setMaskEditMode('editing');

    setShapeDrawState({
      startX: 0,
      startY: 0,
      currentX: 0,
      currentY: 0,
      isDrawing: false,
    });
  }, [shapeDrawState, selectedClip, maskEditMode, addMask, setMaskEditMode]);

  // Handle clicking on first vertex to close path
  const handleFirstVertexClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!selectedClip || !activeMask) return;

    if (maskEditMode === 'drawing' && activeMask.vertices.length >= 3) {
      closeMask(selectedClip.id, activeMask.id);
      setMaskEditMode('editing');
    }
  }, [selectedClip, activeMask, maskEditMode, closeMask, setMaskEditMode]);

  // Handle escape key to exit drawing mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (shapeDrawState.isDrawing) {
          setShapeDrawState(prev => ({ ...prev, isDrawing: false }));
        } else if (maskEditMode === 'drawing' || maskEditMode === 'drawingRect' ||
                   maskEditMode === 'drawingEllipse' || maskEditMode === 'drawingPen') {
          setMaskEditMode('none');
        } else if (maskEditMode === 'editing') {
          setMaskEditMode('none');
        }
      }
      // Delete selected vertices
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
  }, [maskEditMode, setMaskEditMode, selectedVertexIds, selectedClip, activeMask, shapeDrawState.isDrawing]);

  // Don't render if not in mask editing mode
  // For shape drawing modes, we show even without an active mask
  const isShapeDrawingMode = maskEditMode === 'drawingRect' || maskEditMode === 'drawingEllipse' || maskEditMode === 'drawingPen';
  if (maskEditMode === 'none' || !selectedClip) {
    return null;
  }
  if (!isShapeDrawingMode && !activeMask) {
    return null;
  }

  const vertexSize = 8;
  const handleSize = 6;

  // Calculate cursor based on mode
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
      preserveAspectRatio="none"
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
              fill="rgba(0, 212, 255, 0.15)"
              stroke="#00d4ff"
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
              fill="rgba(0, 212, 255, 0.15)"
              stroke="#00d4ff"
              strokeWidth="2"
              strokeDasharray="5,5"
              pointerEvents="none"
            />
          )}
        </>
      )}

      {/* Mask path fill (semi-transparent) */}
      {activeMask?.closed && activeMask.visible && pathData && (
        <path
          d={pathData}
          fill={activeMask.inverted ? 'rgba(0, 212, 255, 0.1)' : 'rgba(0, 212, 255, 0.15)'}
          stroke="none"
          pointerEvents="none"
        />
      )}

      {/* Mask path stroke */}
      {activeMask && activeMask.visible && pathData && (
        <path
          d={pathData}
          fill="none"
          stroke="#00d4ff"
          strokeWidth="2"
          strokeDasharray={activeMask.closed ? 'none' : '5,5'}
          pointerEvents="none"
        />
      )}

      {/* Bezier control handles (only show for selected vertices) */}
      {canvasVertices.map((vertex) => {
        const isSelected = selectedVertexIds.has(vertex.id);
        if (!isSelected) return null;

        return (
          <g key={`handles-${vertex.id}`}>
            {/* Handle In line and point */}
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

            {/* Handle Out line and point */}
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
      {canvasVertices.map((vertex, index) => {
        const isSelected = selectedVertexIds.has(vertex.id);
        const isFirst = index === 0;

        return (
          <rect
            key={vertex.id}
            x={vertex.x - vertexSize / 2}
            y={vertex.y - vertexSize / 2}
            width={vertexSize}
            height={vertexSize}
            fill={isSelected ? '#00d4ff' : '#fff'}
            stroke={isFirst && maskEditMode === 'drawing' ? '#ff0000' : '#00d4ff'}
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
    </svg>
  );
}
