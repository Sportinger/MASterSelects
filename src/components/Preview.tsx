// Preview canvas component with After Effects-style editing overlay

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useEngine } from '../hooks/useEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timeline';
import { MaskOverlay } from './MaskOverlay';
import type { Layer, EngineStats } from '../types';

// Detailed stats overlay component
function StatsOverlay({ stats, resolution, expanded, onToggle }: {
  stats: EngineStats;
  resolution: { width: number; height: number };
  expanded: boolean;
  onToggle: () => void;
}) {
  const fpsColor = stats.fps >= 55 ? '#4f4' : stats.fps >= 30 ? '#ff4' : '#f44';
  const dropColor = stats.drops.lastSecond > 0 ? '#f44' : '#4f4';
  const decoderColor = stats.decoder === 'WebCodecs' ? '#4f4' : stats.decoder === 'HTMLVideo' ? '#fa4' : '#888';

  // Determine bottleneck
  const bottleneck = useMemo(() => {
    const { timing } = stats;
    if (timing.total < 10) return null;
    if (timing.importTexture > timing.renderPass && timing.importTexture > timing.submit) {
      return 'Video Import';
    }
    if (timing.renderPass > timing.submit) {
      return 'GPU Render';
    }
    return 'GPU Submit';
  }, [stats.timing]);

  if (!expanded) {
    return (
      <div
        className="preview-stats preview-stats-compact"
        onClick={onToggle}
        title="Click for detailed stats"
      >
        <span style={{ color: fpsColor, fontWeight: 'bold' }}>{stats.fps}</span>
        <span style={{ opacity: 0.7 }}> FPS</span>
        {stats.decoder !== 'none' && (
          <span style={{ color: decoderColor, marginLeft: 6, fontSize: 9 }}>[{stats.decoder === 'WebCodecs' ? 'WC' : 'HTML'}]</span>
        )}
        {stats.drops.lastSecond > 0 && (
          <span style={{ color: '#f44', marginLeft: 6 }}>▼{stats.drops.lastSecond}</span>
        )}
        <span style={{ opacity: 0.5, marginLeft: 8 }}>
          {resolution.width}×{resolution.height}
        </span>
      </div>
    );
  }

  return (
    <div className="preview-stats preview-stats-expanded" onClick={onToggle}>
      <div className="stats-header">
        <span style={{ color: fpsColor, fontWeight: 'bold', fontSize: 18 }}>{stats.fps}</span>
        <span style={{ opacity: 0.7 }}> / {stats.targetFps} FPS</span>
        <span style={{ opacity: 0.5, marginLeft: 8, fontSize: 11 }}>
          {resolution.width}×{resolution.height}
        </span>
      </div>

      <div className="stats-section">
        <div className="stats-row">
          <span>Frame Gap</span>
          <span style={{ color: stats.timing.rafGap > 20 ? '#ff4' : '#aaa' }}>
            {stats.timing.rafGap.toFixed(1)}ms
          </span>
        </div>
        <div className="stats-row">
          <span>Render Total</span>
          <span style={{ color: stats.timing.total > 12 ? '#ff4' : '#aaa' }}>
            {stats.timing.total.toFixed(2)}ms
          </span>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-label">Pipeline Breakdown</div>
        <div className="stats-bar-container">
          <div
            className="stats-bar stats-bar-import"
            style={{ width: `${Math.min(100, (stats.timing.importTexture / 16.67) * 100)}%` }}
            title={`Import: ${stats.timing.importTexture.toFixed(2)}ms`}
          />
          <div
            className="stats-bar stats-bar-render"
            style={{ width: `${Math.min(100, (stats.timing.renderPass / 16.67) * 100)}%` }}
            title={`Render: ${stats.timing.renderPass.toFixed(2)}ms`}
          />
          <div
            className="stats-bar stats-bar-submit"
            style={{ width: `${Math.min(100, (stats.timing.submit / 16.67) * 100)}%` }}
            title={`Submit: ${stats.timing.submit.toFixed(2)}ms`}
          />
        </div>
        <div className="stats-row" style={{ fontSize: 10, opacity: 0.6 }}>
          <span>Import: {stats.timing.importTexture.toFixed(2)}ms</span>
          <span>Render: {stats.timing.renderPass.toFixed(2)}ms</span>
          <span>Submit: {stats.timing.submit.toFixed(2)}ms</span>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-row">
          <span>Layers</span>
          <span>{stats.layerCount}</span>
        </div>
        <div className="stats-row">
          <span>Decoder</span>
          <span style={{ color: decoderColor }}>{stats.decoder}</span>
        </div>
        <div className="stats-row">
          <span style={{ color: dropColor }}>Drops (last sec)</span>
          <span style={{ color: dropColor }}>{stats.drops.lastSecond}</span>
        </div>
        <div className="stats-row">
          <span>Drops (total)</span>
          <span>{stats.drops.count}</span>
        </div>
        {stats.drops.reason !== 'none' && (
          <div className="stats-row">
            <span>Last Drop Reason</span>
            <span style={{ color: '#f44' }}>{stats.drops.reason.replace('_', ' ')}</span>
          </div>
        )}
        {bottleneck && (
          <div className="stats-row">
            <span>Bottleneck</span>
            <span style={{ color: '#ff4' }}>{bottleneck}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function Preview() {
  const { canvasRef, isEngineReady } = useEngine();
  const { engineStats, outputResolution, layers, selectedLayerId, selectLayer } = useMixerStore();
  const { clips, selectedClipId, selectClip, updateClipTransform, maskEditMode } = useTimelineStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1920, height: 1080 });

  // Stats overlay state
  const [statsExpanded, setStatsExpanded] = useState(false);

  // Edit mode state
  const [editMode, setEditMode] = useState(false);
  const [viewZoom, setViewZoom] = useState(1);
  const [viewPan, setViewPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Drag state for moving layers
  const [isDragging, setIsDragging] = useState(false);
  const [dragLayerId, setDragLayerId] = useState<string | null>(null);
  const dragStart = useRef({ x: 0, y: 0, layerPosX: 0, layerPosY: 0 });
  const currentDragPos = useRef({ x: 0, y: 0 }); // Current drag position for immediate visual feedback

  // Helper function to calculate layer bounding box in canvas coordinates
  // This matches the shader's transform calculation exactly
  const calculateLayerBounds = useCallback((layer: Layer, canvasW: number, canvasH: number, forcePos?: { x: number; y: number }) => {
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

    // Calculate aspect ratios (same as shader)
    const sourceAspect = sourceWidth / sourceHeight;
    const outputAspect = outputResolution.width / outputResolution.height;
    const aspectRatio = sourceAspect / outputAspect;

    // Calculate display size in canvas coordinates
    // The shader fits the source into the output while maintaining aspect ratio
    let displayWidth: number;
    let displayHeight: number;

    if (aspectRatio > 1) {
      // Source is wider than output - fit to width, letterbox top/bottom
      displayWidth = canvasW;
      displayHeight = canvasH / aspectRatio;
    } else {
      // Source is taller than output - fit to height, pillarbox left/right
      displayWidth = canvasW * aspectRatio;
      displayHeight = canvasH;
    }

    // Apply user scale
    displayWidth *= layer.scale.x;
    displayHeight *= layer.scale.y;

    // Calculate center position
    const centerX = canvasW / 2;
    const centerY = canvasH / 2;

    // Use forced position if dragging (for immediate visual feedback), otherwise use layer position
    const layerPos = forcePos || layer.position;

    // Position mapping: match the shader's visual output
    // Shader does: uv = uv + 0.5 - pos
    // When pos.x > 0: uv = ... - pos.x → samples from LEFT → image moves LEFT → box.x decreases
    // When pos.y > 0: uv = ... - pos.y → samples from TOP → image moves UP → box.y decreases
    // Both use SUBTRACT
    const posX = centerX - (layerPos.x * canvasW / 2);
    const posY = centerY - (layerPos.y * canvasH / 2);

    return {
      x: posX,
      y: posY,
      width: displayWidth,
      height: displayHeight,
      rotation: layer.rotation,
    };
  }, [outputResolution]);

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
      const overlayWidth = overlayRef.current!.width;
      const overlayHeight = overlayRef.current!.height;
      ctx.clearRect(0, 0, overlayWidth, overlayHeight);

      // Get visible layers (from timeline clips)
      const visibleLayers = layers.filter(l => l?.visible && l?.source);

      visibleLayers.forEach((layer) => {
        if (!layer) return;

        const isSelected = layer.id === selectedLayerId ||
          clips.find(c => c.id === selectedClipId)?.name === layer.name;

        // Use current drag position if this layer is being dragged (for immediate feedback)
        const forcePos = (isDragging && layer.id === dragLayerId) ? currentDragPos.current : undefined;

        // Calculate bounding box using the helper function
        const bounds = calculateLayerBounds(layer, canvasSize.width, canvasSize.height, forcePos);

        // Save context for rotation
        ctx.save();
        ctx.translate(bounds.x, bounds.y);
        ctx.rotate(bounds.rotation);

        // Draw bounding box
        const halfW = bounds.width / 2;
        const halfH = bounds.height / 2;

        ctx.strokeStyle = isSelected ? '#00d4ff' : 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash(isSelected ? [] : [5, 5]);
        ctx.strokeRect(-halfW, -halfH, bounds.width, bounds.height);

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
      ctx.strokeRect(0, 0, canvasSize.width, canvasSize.height);
    };

    draw();

    // Redraw on animation frame for smooth updates
    const animId = requestAnimationFrame(function loop() {
      draw();
      requestAnimationFrame(loop);
    });

    return () => cancelAnimationFrame(animId);
  }, [editMode, layers, selectedLayerId, selectedClipId, clips, canvasSize, calculateLayerBounds, isDragging, dragLayerId]);

  // Find layer at mouse position
  const findLayerAtPosition = useCallback((x: number, y: number): Layer | null => {
    const visibleLayers = layers.filter(l => l?.visible && l?.source).reverse();

    for (const layer of visibleLayers) {
      if (!layer) continue;

      const bounds = calculateLayerBounds(layer, canvasSize.width, canvasSize.height);

      // Simple bounding box hit test (ignoring rotation for simplicity)
      const halfW = bounds.width / 2;
      const halfH = bounds.height / 2;

      if (x >= bounds.x - halfW && x <= bounds.x + halfW &&
          y >= bounds.y - halfH && y <= bounds.y + halfH) {
        return layer;
      }
    }
    return null;
  }, [layers, canvasSize, calculateLayerBounds]);

  // Handle mouse down on overlay - select or start dragging
  const handleOverlayMouseDown = useCallback((e: React.MouseEvent) => {
    if (!editMode || !overlayRef.current || e.altKey) return;
    if (e.button !== 0) return; // Only left click

    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const layer = findLayerAtPosition(x, y);

    if (layer) {
      // Select the layer
      const clip = clips.find(c => c.name === layer.name);
      if (clip) {
        selectClip(clip.id);
      }
      selectLayer(layer.id);

      // Start dragging
      setIsDragging(true);
      setDragLayerId(layer.id);
      dragStart.current = {
        x: e.clientX,
        y: e.clientY,
        layerPosX: layer.position.x,
        layerPosY: layer.position.y,
      };
    } else {
      // Click on empty space - deselect
      selectClip(null);
      selectLayer(null);
    }
  }, [editMode, findLayerAtPosition, clips, selectClip, selectLayer]);

  // Handle mouse move - drag layer
  const handleOverlayMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragLayerId) return;

    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;

    // Convert pixel movement to normalized position change
    // Use canvas dimensions directly for proper aspect ratio handling
    const normalizedDx = (dx / viewZoom) / canvasSize.width;
    const normalizedDy = (dy / viewZoom) / canvasSize.height;

    // Box uses negated values (box formula: posX = center - layer.position.x * canvasW/2)
    // Box formula has /2 built in, so we multiply by 2
    const boxPosX = dragStart.current.layerPosX - (normalizedDx * 2);
    const boxPosY = dragStart.current.layerPosY - (normalizedDy * 2);

    // Image uses direct values without scaling (shader works in 0-1 UV space)
    const imagePosX = dragStart.current.layerPosX + normalizedDx;
    const imagePosY = dragStart.current.layerPosY + normalizedDy;

    // Update current drag position for box visual feedback
    currentDragPos.current = { x: boxPosX, y: boxPosY };

    console.log(`[Drag] mouse dx=${dx.toFixed(0)}, box=${boxPosX.toFixed(4)}, image=${imagePosX.toFixed(4)}`);

    // Find the corresponding clip and update its transform
    const layer = layers.find(l => l?.id === dragLayerId);
    if (layer) {
      const clip = clips.find(c => c.name === layer.name);
      if (clip) {
        updateClipTransform(clip.id, {
          position: { x: imagePosX, y: imagePosY, z: 0 },
        });
      }
    }
  }, [isDragging, dragLayerId, viewZoom, canvasSize, layers, clips, updateClipTransform]);

  // Handle mouse up - stop dragging
  const handleOverlayMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragLayerId(null);
    currentDragPos.current = { x: 0, y: 0 };
  }, []);

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

      <StatsOverlay
        stats={engineStats}
        resolution={outputResolution}
        expanded={statsExpanded}
        onToggle={() => setStatsExpanded(!statsExpanded)}
      />

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
                width={canvasSize.width}
                height={canvasSize.height}
                className="preview-overlay"
                onMouseDown={handleOverlayMouseDown}
                onMouseMove={handleOverlayMouseMove}
                onMouseUp={handleOverlayMouseUp}
                onMouseLeave={handleOverlayMouseUp}
                style={{
                  width: canvasSize.width,
                  height: canvasSize.height,
                  cursor: isDragging ? 'grabbing' : 'crosshair',
                }}
              />
            )}
            {maskEditMode !== 'none' && (
              <MaskOverlay
                canvasWidth={outputResolution.width}
                canvasHeight={outputResolution.height}
              />
            )}
          </>
        )}
      </div>

      {editMode && (
        <div className="preview-edit-hint">
          Drag: Move Layer | Shift+Scroll: Zoom | Alt+Drag: Pan
        </div>
      )}
    </div>
  );
}
