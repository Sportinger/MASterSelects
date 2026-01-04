// Preview canvas component with After Effects-style editing overlay

import { useEffect, useRef, useState, useCallback } from 'react';
import { useEngine } from '../hooks/useEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timelineStore';

export function Preview() {
  const { canvasRef, isEngineReady } = useEngine();
  const { engineStats, outputResolution, layers, selectedLayerId, selectLayer } = useMixerStore();
  const { clips, selectedClipId, selectClip, updateClipTransform } = useTimelineStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1920, height: 1080 });

  // Overlay padding to show handles/outlines beyond canvas bounds
  const OVERLAY_PADDING = 100;

  // Edit mode state
  const [editMode, setEditMode] = useState(false);
  const [viewZoom, setViewZoom] = useState(1);
  const [viewPan, setViewPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Calculate canvas size to fit container while maintaining aspect ratio
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      if (containerWidth === 0 || containerHeight === 0) return;

      const videoAspect = outputResolution.width / outputResolution.height;
      const containerAspect = containerWidth / containerHeight;

      let width: number;
      let height: number;

      if (containerAspect > videoAspect) {
        height = containerHeight;
        width = height * videoAspect;
      } else {
        width = containerWidth;
        height = width / videoAspect;
      }

      setCanvasSize({
        width: Math.floor(width),
        height: Math.floor(height),
      });
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [outputResolution.width, outputResolution.height]);

  // Handle zoom with Shift+Scroll
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!editMode) return;

    if (e.shiftKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setViewZoom(prev => Math.max(0.1, Math.min(5, prev * delta)));
    } else if (e.altKey) {
      // Alt+scroll for horizontal pan
      e.preventDefault();
      setViewPan(prev => ({
        x: prev.x - e.deltaY,
        y: prev.y
      }));
    }
  }, [editMode]);

  // Handle panning with middle mouse or Alt+drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!editMode) return;

    // Middle mouse button or Alt+left click
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      setIsPanning(true);
      panStart.current = {
        x: e.clientX,
        y: e.clientY,
        panX: viewPan.x,
        panY: viewPan.y
      };
    }
  }, [editMode, viewPan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      const dx = e.clientX - panStart.current.x;
      const dy = e.clientY - panStart.current.y;
      setViewPan({
        x: panStart.current.panX + dx,
        y: panStart.current.panY + dy
      });
    }
  }, [isPanning]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  // Reset view
  const resetView = useCallback(() => {
    setViewZoom(1);
    setViewPan({ x: 0, y: 0 });
  }, []);

  // Draw overlay with bounding boxes
  useEffect(() => {
    if (!editMode || !overlayRef.current) return;

    const ctx = overlayRef.current.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      ctx.clearRect(0, 0, overlayRef.current!.width, overlayRef.current!.height);

      // The overlay canvas is larger than the video canvas by OVERLAY_PADDING on each side
      // So the video canvas area starts at (OVERLAY_PADDING, OVERLAY_PADDING)
      const videoCanvasWidth = overlayRef.current!.width - OVERLAY_PADDING * 2;
      const videoCanvasHeight = overlayRef.current!.height - OVERLAY_PADDING * 2;

      // Get visible layers (from timeline clips)
      const visibleLayers = layers.filter(l => l?.visible && l?.source);

      visibleLayers.forEach((layer, index) => {
        if (!layer) return;

        const isSelected = layer.id === selectedLayerId ||
          clips.find(c => c.id === selectedClipId)?.name === layer.name;

        // Calculate bounding box based on layer transform
        // Center is now offset by OVERLAY_PADDING
        const centerX = OVERLAY_PADDING + videoCanvasWidth / 2;
        const centerY = OVERLAY_PADDING + videoCanvasHeight / 2;

        // Get source dimensions
        let sourceWidth = outputResolution.width;
        let sourceHeight = outputResolution.height;

        if (layer.source?.videoElement) {
          sourceWidth = layer.source.videoElement.videoWidth || sourceWidth;
          sourceHeight = layer.source.videoElement.videoHeight || sourceHeight;
        } else if (layer.source?.imageElement) {
          sourceWidth = layer.source.imageElement.naturalWidth || sourceWidth;
          sourceHeight = layer.source.imageElement.naturalHeight || sourceHeight;
        }

        // Calculate aspect ratio correction (same as shader)
        const sourceAspect = sourceWidth / sourceHeight;
        const outputAspect = outputResolution.width / outputResolution.height;

        let displayWidth = videoCanvasWidth;
        let displayHeight = videoCanvasHeight;

        if (sourceAspect > outputAspect) {
          // Source is wider - fit to width
          displayHeight = displayWidth / sourceAspect;
        } else {
          // Source is taller - fit to height
          displayWidth = displayHeight * sourceAspect;
        }

        // Apply layer scale
        displayWidth *= layer.scale.x;
        displayHeight *= layer.scale.y;

        // Apply layer position (in normalized coordinates)
        const posX = centerX + layer.position.x * (videoCanvasWidth / 2);
        const posY = centerY - layer.position.y * (videoCanvasHeight / 2);

        // Save context for rotation
        ctx.save();
        ctx.translate(posX, posY);
        ctx.rotate(layer.rotation);

        // Draw bounding box
        const halfW = displayWidth / 2;
        const halfH = displayHeight / 2;

        ctx.strokeStyle = isSelected ? '#00d4ff' : 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash(isSelected ? [] : [5, 5]);
        ctx.strokeRect(-halfW, -halfH, displayWidth, displayHeight);

        // Draw corner handles for selected layer
        if (isSelected) {
          const handleSize = 8;
          ctx.fillStyle = '#00d4ff';

          // Corners
          ctx.fillRect(-halfW - handleSize/2, -halfH - handleSize/2, handleSize, handleSize);
          ctx.fillRect(halfW - handleSize/2, -halfH - handleSize/2, handleSize, handleSize);
          ctx.fillRect(-halfW - handleSize/2, halfH - handleSize/2, handleSize, handleSize);
          ctx.fillRect(halfW - handleSize/2, halfH - handleSize/2, handleSize, handleSize);

          // Edge midpoints
          ctx.fillRect(-handleSize/2, -halfH - handleSize/2, handleSize, handleSize);
          ctx.fillRect(-handleSize/2, halfH - handleSize/2, handleSize, handleSize);
          ctx.fillRect(-halfW - handleSize/2, -handleSize/2, handleSize, handleSize);
          ctx.fillRect(halfW - handleSize/2, -handleSize/2, handleSize, handleSize);

          // Center crosshair
          ctx.strokeStyle = '#00d4ff';
          ctx.lineWidth = 1;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(-10, 0);
          ctx.lineTo(10, 0);
          ctx.moveTo(0, -10);
          ctx.lineTo(0, 10);
          ctx.stroke();
        }

        // Draw layer name label
        ctx.fillStyle = isSelected ? '#00d4ff' : 'rgba(255, 255, 255, 0.7)';
        ctx.font = '11px sans-serif';
        ctx.fillText(layer.name, -halfW + 4, -halfH - 6);

        ctx.restore();
      });

      // Draw canvas bounds indicator (the dashed line showing video bounds)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 10]);
      ctx.strokeRect(OVERLAY_PADDING, OVERLAY_PADDING, videoCanvasWidth, videoCanvasHeight);
    };

    draw();

    // Redraw on animation frame for smooth updates
    const animId = requestAnimationFrame(function loop() {
      draw();
      requestAnimationFrame(loop);
    });

    return () => cancelAnimationFrame(animId);
  }, [editMode, layers, selectedLayerId, selectedClipId, clips, outputResolution, viewZoom]);

  // Handle click to select layer
  const handleOverlayClick = useCallback((e: React.MouseEvent) => {
    if (!editMode || !overlayRef.current || e.altKey) return;

    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // The overlay canvas is larger than the video canvas by OVERLAY_PADDING
    const videoCanvasWidth = overlayRef.current.width - OVERLAY_PADDING * 2;
    const videoCanvasHeight = overlayRef.current.height - OVERLAY_PADDING * 2;

    // Check which layer was clicked (in reverse order - top layers first)
    const visibleLayers = layers.filter(l => l?.visible && l?.source).reverse();

    for (const layer of visibleLayers) {
      if (!layer) continue;

      const centerX = OVERLAY_PADDING + videoCanvasWidth / 2;
      const centerY = OVERLAY_PADDING + videoCanvasHeight / 2;

      let sourceWidth = outputResolution.width;
      let sourceHeight = outputResolution.height;

      if (layer.source?.videoElement) {
        sourceWidth = layer.source.videoElement.videoWidth || sourceWidth;
        sourceHeight = layer.source.videoElement.videoHeight || sourceHeight;
      } else if (layer.source?.imageElement) {
        sourceWidth = layer.source.imageElement.naturalWidth || sourceWidth;
        sourceHeight = layer.source.imageElement.naturalHeight || sourceHeight;
      }

      const sourceAspect = sourceWidth / sourceHeight;
      const outputAspect = outputResolution.width / outputResolution.height;

      let displayWidth = videoCanvasWidth;
      let displayHeight = videoCanvasHeight;

      if (sourceAspect > outputAspect) {
        displayHeight = displayWidth / sourceAspect;
      } else {
        displayWidth = displayHeight * sourceAspect;
      }

      displayWidth *= layer.scale.x;
      displayHeight *= layer.scale.y;

      const posX = centerX + layer.position.x * (videoCanvasWidth / 2);
      const posY = centerY - layer.position.y * (videoCanvasHeight / 2);

      // Simple bounding box hit test (ignoring rotation for simplicity)
      const halfW = displayWidth / 2;
      const halfH = displayHeight / 2;

      if (x >= posX - halfW && x <= posX + halfW &&
          y >= posY - halfH && y <= posY + halfH) {

        // Find corresponding clip
        const clip = clips.find(c => c.name === layer.name);
        if (clip) {
          selectClip(clip.id);
        }
        selectLayer(layer.id);
        return;
      }
    }

    // Click on empty space - deselect
    selectClip(null);
    selectLayer(null);
  }, [editMode, layers, clips, outputResolution, selectClip, selectLayer]);

  // Calculate transform for zoomed/panned view
  const viewTransform = editMode ? {
    transform: `scale(${viewZoom}) translate(${viewPan.x / viewZoom}px, ${viewPan.y / viewZoom}px)`,
  } : {};

  return (
    <div
      className="preview-container"
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      style={{ cursor: isPanning ? 'grabbing' : (editMode ? 'crosshair' : 'default') }}
    >
      {/* Edit mode toggle button */}
      <div className="preview-controls">
        <button
          className={`preview-edit-btn ${editMode ? 'active' : ''}`}
          onClick={() => setEditMode(!editMode)}
          title="Toggle Edit Mode (show layer bounds)"
        >
          {editMode ? '✓ Edit' : 'Edit'}
        </button>
        {editMode && (
          <>
            <span className="preview-zoom-label">{Math.round(viewZoom * 100)}%</span>
            <button
              className="preview-reset-btn"
              onClick={resetView}
              title="Reset View"
            >
              Reset
            </button>
          </>
        )}
      </div>

      <div className="preview-stats">
        {engineStats.fps} FPS | {outputResolution.width}x{outputResolution.height}
      </div>

      <div className="preview-canvas-wrapper" style={viewTransform}>
        {!isEngineReady ? (
          <div className="loading">
            <div className="loading-spinner" />
            <p>Initializing WebGPU...</p>
          </div>
        ) : (
          <>
            <canvas
              ref={canvasRef}
              width={outputResolution.width}
              height={outputResolution.height}
              className="preview-canvas"
              style={{
                width: canvasSize.width,
                height: canvasSize.height,
              }}
            />
            {editMode && (
              <canvas
                ref={overlayRef}
                width={canvasSize.width + OVERLAY_PADDING * 2}
                height={canvasSize.height + OVERLAY_PADDING * 2}
                className="preview-overlay"
                onClick={handleOverlayClick}
                style={{
                  width: canvasSize.width + OVERLAY_PADDING * 2,
                  height: canvasSize.height + OVERLAY_PADDING * 2,
                  left: -OVERLAY_PADDING,
                  top: -OVERLAY_PADDING,
                }}
              />
            )}
          </>
        )}
      </div>

      {editMode && (
        <div className="preview-edit-hint">
          Shift+Scroll: Zoom | Alt+Drag: Pan | Click: Select
        </div>
      )}
    </div>
  );
}
