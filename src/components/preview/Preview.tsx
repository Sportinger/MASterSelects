// Preview canvas component with After Effects-style editing overlay

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { Logger } from '../../services/logger';

const log = Logger.create('Preview');
import { useEngine } from '../../hooks/useEngine';
import { useEngineStore } from '../../stores/engineStore';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { useDockStore } from '../../stores/dockStore';
import { useSettingsStore, type PreviewQuality } from '../../stores/settingsStore';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { MaskOverlay } from './MaskOverlay';
import { SAM2Overlay } from './SAM2Overlay';
import { SourceMonitor } from './SourceMonitor';
import { useSAM2Store } from '../../stores/sam2Store';
import { renderScheduler } from '../../services/renderScheduler';
import { engine } from '../../engine/WebGPUEngine';
import type { RenderSource } from '../../types/renderTarget';
import type { EngineStats, Layer } from '../../types';

interface PreviewProps {
  panelId: string;
  compositionId: string | null; // null = active composition
}

// Detailed stats overlay component
function StatsOverlay({ stats, resolution, expanded, onToggle }: {
  stats: EngineStats;
  resolution: { width: number; height: number };
  expanded: boolean;
  onToggle: () => void;
}) {
  const fpsColor = stats.fps >= 55 ? '#4f4' : stats.fps >= 30 ? '#ff4' : '#f44';
  const dropColor = stats.drops.lastSecond > 0 ? '#f44' : '#4f4';
  const decoderColor = stats.decoder === 'NativeHelper' ? '#4af' : stats.decoder === 'WebCodecs' ? '#4f4' : stats.decoder === 'ParallelDecode' ? '#a4f' : stats.decoder.startsWith('HTMLVideo') ? '#fa4' : '#888';
  // Render time color: green < 10ms, yellow < 16.67ms (60fps target), red >= 16.67ms
  const renderTime = stats.timing.total;
  const renderTimeColor = renderTime < 10 ? '#4f4' : renderTime < 16.67 ? '#ff4' : '#f44';

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
        {!stats.isIdle && renderTime > 0 && (
          <span style={{ color: renderTimeColor, marginLeft: 6, fontSize: 10 }}>
            {renderTime.toFixed(1)}ms
          </span>
        )}
        {stats.isIdle && (
          <span style={{ color: '#888', marginLeft: 6, fontSize: 9 }}>[IDLE]</span>
        )}
        {stats.decoder !== 'none' && !stats.isIdle && (
          <span style={{ color: decoderColor, marginLeft: 6, fontSize: 9 }}>[{stats.decoder === 'WebCodecs' ? 'WC' : stats.decoder === 'NativeHelper' ? 'NH' : stats.decoder === 'ParallelDecode' ? 'PD' : 'HTML'}]</span>
        )}
        {stats.drops.lastSecond > 0 && (
          <span style={{ color: '#f44', marginLeft: 6 }}>▼{stats.drops.lastSecond}</span>
        )}
        {stats.audio?.status && stats.audio.status !== 'silent' && (
          <span style={{
            marginLeft: 6,
            color: stats.audio.status === 'sync' ? '#4f4'
              : stats.audio.status === 'drift' ? '#ff4'
              : '#f44'
          }}>
            🔊{stats.audio.status === 'drift' ? `(${stats.audio.drift}ms)` : ''}
          </span>
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
        {stats.isIdle && (
          <span style={{ color: '#888', marginLeft: 8, fontSize: 11 }}>[IDLE]</span>
        )}
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
          <span>Engine</span>
          <span style={{ color: stats.isIdle ? '#888' : '#4f4' }}>
            {stats.isIdle ? '● Idle (saving power)' : '● Active'}
          </span>
        </div>
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

      {/* Audio Status Section */}
      {stats.audio && (
        <div className="stats-section">
          <div className="stats-label">Audio</div>
          <div className="stats-row">
            <span>Status</span>
            <span style={{
              color: stats.audio.status === 'sync' ? '#4f4'
                : stats.audio.status === 'drift' ? '#ff4'
                : stats.audio.status === 'error' ? '#f44'
                : '#888'
            }}>
              {stats.audio.status === 'sync' ? '● Sync'
                : stats.audio.status === 'drift' ? '◐ Drift'
                : stats.audio.status === 'error' ? '✕ Error'
                : '○ Silent'}
            </span>
          </div>
          {stats.audio.playing > 0 && (
            <div className="stats-row">
              <span>Playing</span>
              <span>{stats.audio.playing} track{stats.audio.playing !== 1 ? 's' : ''}</span>
            </div>
          )}
          {stats.audio.drift > 0 && (
            <div className="stats-row">
              <span>Drift</span>
              <span style={{ color: stats.audio.drift > 100 ? '#f44' : stats.audio.drift > 50 ? '#ff4' : '#aaa' }}>
                {stats.audio.drift}ms
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function Preview({ panelId, compositionId }: PreviewProps) {
  const { isEngineReady } = useEngine();
  const { engineStats } = useEngineStore();
  const { clips, selectedClipIds, selectClip, updateClipTransform, maskEditMode, layers, selectedLayerId, selectLayer, updateLayer } = useTimelineStore();
  const { compositions, activeCompositionId } = useMediaStore();
  const { addPreviewPanel, updatePanelData, closePanelById } = useDockStore();
  const { previewQuality, setPreviewQuality, showTransparencyGrid, setShowTransparencyGrid } = useSettingsStore();
  const sam2Active = useSAM2Store((s) => s.isActive);

  // Get first selected clip for preview
  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1920, height: 1080 });
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [, setCompReady] = useState(false);

  const previewCompositionId = useMediaStore(state => state.previewCompositionId);
  const sourceMonitorFileId = useMediaStore(state => state.sourceMonitorFileId);
  const sourceMonitorFile = useMediaStore(state =>
    state.sourceMonitorFileId ? state.files.find(f => f.id === state.sourceMonitorFileId) ?? null : null
  );

  // Source monitor: show raw media file instead of composition
  const sourceMonitorActive = compositionId === null && sourceMonitorFile !== null;

  const closeSourceMonitor = useCallback(() => {
    useMediaStore.getState().setSourceMonitorFile(null);
  }, []);

  // Clear source monitor when active composition changes
  useEffect(() => {
    if (activeCompositionId && sourceMonitorFileId) {
      useMediaStore.getState().setSourceMonitorFile(null);
    }
  }, [activeCompositionId]);

  // Determine which composition this preview is showing
  // When in "Active" mode (compositionId === null) but a slot preview is active,
  // override to show the previewed composition independently
  const slotPreviewActive = compositionId === null && previewCompositionId !== null;
  const displayedCompId = slotPreviewActive
    ? previewCompositionId
    : (compositionId ?? activeCompositionId);
  const displayedComp = compositions.find(c => c.id === displayedCompId);

  // Engine resolution = active composition dimensions (fallback to settingsStore default)
  const effectiveResolution = displayedComp
    ? { width: displayedComp.width, height: displayedComp.height }
    : useSettingsStore.getState().outputResolution;

  // Unified RenderTarget registration
  // Determines source, registers canvas with engine, registers target in store,
  // and sets up the render scheduler for independent (non-activeComp) sources
  useEffect(() => {
    if (!isEngineReady || !canvasRef.current) return;

    // Determine source based on composition selector and slot preview state
    const source: RenderSource = compositionId
      ? { type: 'composition', compositionId }
      : slotPreviewActive && previewCompositionId
        ? { type: 'composition', compositionId: previewCompositionId }
        : { type: 'activeComp' };

    const isIndependent = source.type !== 'activeComp';

    log.debug(`[${panelId}] Registering render target`, { source, isIndependent });

    // Register canvas with engine (creates WebGPU context)
    const gpuContext = engine.registerTargetCanvas(panelId, canvasRef.current);
    if (!gpuContext) return;

    // Register as render target in store
    useRenderTargetStore.getState().registerTarget({
      id: panelId,
      name: 'Preview',
      source,
      destinationType: 'canvas',
      enabled: true,
      canvas: canvasRef.current,
      context: gpuContext,
      window: null,
      isFullscreen: false,
    });

    // For independent sources, register with the render scheduler
    // (it handles: composition preparation, RAF loop, nested comp sync)
    if (isIndependent) {
      renderScheduler.register(panelId);
      setCompReady(true);
    }

    return () => {
      log.debug(`[${panelId}] Unregistering render target`);
      if (isIndependent) {
        renderScheduler.unregister(panelId);
      }
      useRenderTargetStore.getState().unregisterTarget(panelId);
      engine.unregisterTargetCanvas(panelId);
    };
  }, [isEngineReady, panelId, compositionId, previewCompositionId, slotPreviewActive]);

  // Composition selector state
  const [selectorOpen, setSelectorOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>({});

  // Quality selector state
  const [qualityOpen, setQualityOpen] = useState(false);
  const qualityDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdowns when clicking outside
  useEffect(() => {
    if (!selectorOpen && !qualityOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (selectorOpen && dropdownRef.current && !dropdownRef.current.contains(target)) {
        setSelectorOpen(false);
      }
      if (qualityOpen && qualityDropdownRef.current && !qualityDropdownRef.current.contains(target)) {
        setQualityOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [selectorOpen, qualityOpen]);

  // Adjust dropdown position when opened to stay within viewport
  useEffect(() => {
    if (selectorOpen && dropdownRef.current) {
      const rect = dropdownRef.current.getBoundingClientRect();
      const style: React.CSSProperties = {};

      // Check if dropdown goes off left edge
      if (rect.left < 8) {
        style.left = '0';
        style.right = 'auto';
      }
      // Check if dropdown goes off right edge
      if (rect.right > window.innerWidth - 8) {
        style.right = '0';
        style.left = 'auto';
      }
      // Check if dropdown goes off bottom edge
      if (rect.bottom > window.innerHeight - 8) {
        style.bottom = '100%';
        style.top = 'auto';
        style.marginTop = '0';
        style.marginBottom = '4px';
      }

      setDropdownStyle(style);
    } else {
      setDropdownStyle({});
    }
  }, [selectorOpen]);

  // Stats overlay state
  const [statsExpanded, setStatsExpanded] = useState(false);

  // Edit mode state
  const [editMode, setEditMode] = useState(false);
  const [viewZoom, setViewZoom] = useState(1);
  const [viewPan, setViewPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Drag state for moving/scaling layers
  const [isDragging, setIsDragging] = useState(false);
  const [dragLayerId, setDragLayerId] = useState<string | null>(null);
  const [dragMode, setDragMode] = useState<'move' | 'scale'>('move');
  const [dragHandle, setDragHandle] = useState<string | null>(null); // 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'
  const [hoverHandle, setHoverHandle] = useState<string | null>(null); // For cursor feedback
  const dragStart = useRef({ x: 0, y: 0, layerPosX: 0, layerPosY: 0, layerScaleX: 1, layerScaleY: 1 });
  const currentDragPos = useRef({ x: 0, y: 0 }); // Current drag position for immediate visual feedback

  // Sync layer selection when clip is selected in timeline (for edit mode)
  useEffect(() => {
    if (!selectedClipId || !editMode) return;

    const clip = clips.find(c => c.id === selectedClipId);
    if (clip) {
      // Find matching layer by name
      const layer = layers.find(l => l?.name === clip.name);
      if (layer && layer.id !== selectedLayerId) {
        selectLayer(layer.id);
      }
    }
  }, [selectedClipId, editMode, clips, layers, selectedLayerId, selectLayer]);

  // Helper function to calculate layer bounding box in canvas coordinates
  // This matches the shader's transform calculation exactly
  const calculateLayerBounds = useCallback((layer: Layer, canvasW: number, canvasH: number, forcePos?: { x: number; y: number }) => {
    // Get source dimensions
    let sourceWidth = effectiveResolution.width;
    let sourceHeight = effectiveResolution.height;

    if (layer.source?.videoElement) {
      sourceWidth = layer.source.videoElement.videoWidth || sourceWidth;
      sourceHeight = layer.source.videoElement.videoHeight || sourceHeight;
    } else if (layer.source?.imageElement) {
      sourceWidth = layer.source.imageElement.naturalWidth || sourceWidth;
      sourceHeight = layer.source.imageElement.naturalHeight || sourceHeight;
    }

    // Calculate aspect ratios (same as shader)
    const sourceAspect = sourceWidth / sourceHeight;
    const outputAspect = effectiveResolution.width / effectiveResolution.height;
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
    // Shader does: uv = uv + 0.5 - pos (samples from offset UV)
    // pos is in normalized space where 1.0 = full canvas width/height
    // Box position uses the same scale as shader (no / 2)
    const posX = centerX + (layerPos.x * canvasW);
    const posY = centerY + (layerPos.y * canvasH);

    // Extract rotation value - if it's an object, use z rotation
    const rotationValue = typeof layer.rotation === 'number' ? layer.rotation : layer.rotation.z;

    return {
      x: posX,
      y: posY,
      width: displayWidth,
      height: displayHeight,
      rotation: rotationValue,
    };
  }, [effectiveResolution]);

  // Calculate canvas size to fit container while maintaining aspect ratio
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      if (containerWidth === 0 || containerHeight === 0) return;

      // Track container size for overlay
      setContainerSize({ width: containerWidth, height: containerHeight });

      const videoAspect = effectiveResolution.width / effectiveResolution.height;
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
  }, [effectiveResolution.width, effectiveResolution.height]);

  // Handle zoom with scroll wheel in edit mode
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!editMode || !containerRef.current) return;

    e.preventDefault();

    if (e.altKey) {
      // Alt+scroll for horizontal pan
      setViewPan(prev => ({
        x: prev.x - e.deltaY,
        y: prev.y
      }));
    } else {
      // Zoom towards mouse position (like After Effects)
      const rect = containerRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.1, Math.min(150, viewZoom * zoomFactor));

      // Calculate the point under the mouse in "world" coordinates (before zoom)
      // The canvas center is at (containerWidth/2 + panX, containerHeight/2 + panY)
      // World point = (mousePos - canvasCenter) / zoom
      const containerCenterX = containerSize.width / 2;
      const containerCenterY = containerSize.height / 2;

      const worldX = (mouseX - containerCenterX - viewPan.x) / viewZoom;
      const worldY = (mouseY - containerCenterY - viewPan.y) / viewZoom;

      // After zoom, adjust pan so the same world point stays under the mouse
      // mousePos = worldPoint * newZoom + canvasCenter + newPan
      // newPan = mousePos - worldPoint * newZoom - canvasCenter
      const newPanX = mouseX - worldX * newZoom - containerCenterX;
      const newPanY = mouseY - worldY * newZoom - containerCenterY;

      setViewZoom(newZoom);
      setViewPan({ x: newPanX, y: newPanY });
    }
  }, [editMode, viewZoom, viewPan, containerSize]);

  // Tab key to toggle edit mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle Tab when preview panel is focused or no input is focused
      const activeElement = document.activeElement;
      const isInputFocused = activeElement instanceof HTMLInputElement ||
                            activeElement instanceof HTMLTextAreaElement ||
                            activeElement?.getAttribute('contenteditable') === 'true';

      if (e.key === 'Tab' && !isInputFocused) {
        e.preventDefault();
        setEditMode(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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

  // Calculate canvas position within container (for full-container overlay)
  // This accounts for zoom and pan to determine where the composition canvas appears
  const canvasInContainer = useMemo(() => {
    // Canvas is centered in container, then zoom and pan are applied
    const scaledWidth = canvasSize.width * viewZoom;
    const scaledHeight = canvasSize.height * viewZoom;

    // Center position before pan
    const centerX = (containerSize.width - scaledWidth) / 2;
    const centerY = (containerSize.height - scaledHeight) / 2;

    // Apply pan (pan is in screen pixels)
    return {
      x: centerX + viewPan.x,
      y: centerY + viewPan.y,
      width: scaledWidth,
      height: scaledHeight,
    };
  }, [containerSize, canvasSize, viewZoom, viewPan]);

  // Draw overlay with bounding boxes (full-container overlay for pasteboard support)
  useEffect(() => {
    if (!editMode || !overlayRef.current) return;

    const ctx = overlayRef.current.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const overlayWidth = overlayRef.current!.width;
      const overlayHeight = overlayRef.current!.height;
      ctx.clearRect(0, 0, overlayWidth, overlayHeight);

      // Fill grey pasteboard area (outside composition bounds)
      ctx.fillStyle = '#2a2a2a';
      ctx.fillRect(0, 0, overlayWidth, overlayHeight);

      // Clear the composition area (make it transparent so video shows through)
      ctx.clearRect(
        canvasInContainer.x,
        canvasInContainer.y,
        canvasInContainer.width,
        canvasInContainer.height
      );

      // Get visible layers (from timeline clips)
      const visibleLayers = layers.filter(l => l?.visible && l?.source);

      visibleLayers.forEach((layer) => {
        if (!layer) return;

        const isSelected = layer.id === selectedLayerId ||
          clips.find(c => c.id === selectedClipId)?.name === layer.name;

        // Use current drag position if this layer is being dragged (for immediate feedback)
        const forcePos = (isDragging && layer.id === dragLayerId) ? currentDragPos.current : undefined;

        // Calculate bounding box in canvas coordinates
        const bounds = calculateLayerBounds(layer, canvasSize.width, canvasSize.height, forcePos);

        // Convert to container coordinates (apply zoom and canvas offset)
        const containerX = canvasInContainer.x + bounds.x * viewZoom;
        const containerY = canvasInContainer.y + bounds.y * viewZoom;
        const containerWidth = bounds.width * viewZoom;
        const containerHeight = bounds.height * viewZoom;

        // Save context for rotation
        ctx.save();
        ctx.translate(containerX, containerY);
        ctx.rotate(bounds.rotation);

        // Draw bounding box
        const halfW = containerWidth / 2;
        const halfH = containerHeight / 2;

        ctx.strokeStyle = isSelected ? '#2997E5' : 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash(isSelected ? [] : [5, 5]);
        ctx.strokeRect(-halfW, -halfH, containerWidth, containerHeight);

        // Draw corner handles for selected layer
        if (isSelected) {
          const handleSize = 8;
          ctx.fillStyle = '#2997E5';

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
          ctx.strokeStyle = '#2997E5';
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
        ctx.fillStyle = isSelected ? '#2997E5' : 'rgba(255, 255, 255, 0.7)';
        ctx.font = '11px sans-serif';
        ctx.fillText(layer.name, -halfW + 4, -halfH - 6);

        ctx.restore();
      });
    };

    draw();

    // Only use RAF loop during active dragging for smooth updates
    // Otherwise, a single draw() is enough since we re-render on dependency changes
    let animId: number | null = null;
    if (isDragging) {
      const loop = () => {
        draw();
        animId = requestAnimationFrame(loop);
      };
      animId = requestAnimationFrame(loop);
    }

    return () => {
      if (animId !== null) cancelAnimationFrame(animId);
    };
  }, [editMode, layers, selectedLayerId, selectedClipIds, clips, canvasSize, containerSize, canvasInContainer, viewZoom, calculateLayerBounds, isDragging, dragLayerId]);

  // Find layer at mouse position (input is in container coordinates)
  const findLayerAtPosition = useCallback((containerX: number, containerY: number): Layer | null => {
    const visibleLayers = layers.filter(l => l?.visible && l?.source).reverse();

    for (const layer of visibleLayers) {
      if (!layer) continue;

      // Get bounds in canvas coordinates
      const bounds = calculateLayerBounds(layer, canvasSize.width, canvasSize.height);

      // Convert bounds to container coordinates
      const layerContainerX = canvasInContainer.x + bounds.x * viewZoom;
      const layerContainerY = canvasInContainer.y + bounds.y * viewZoom;
      const halfW = (bounds.width * viewZoom) / 2;
      const halfH = (bounds.height * viewZoom) / 2;

      // Simple bounding box hit test (ignoring rotation for simplicity)
      if (containerX >= layerContainerX - halfW && containerX <= layerContainerX + halfW &&
          containerY >= layerContainerY - halfH && containerY <= layerContainerY + halfH) {
        return layer;
      }
    }
    return null;
  }, [layers, canvasSize, canvasInContainer, viewZoom, calculateLayerBounds]);

  // Find which handle (if any) was clicked on the selected layer
  const findHandleAtPosition = useCallback((containerX: number, containerY: number, layer: Layer): string | null => {
    const bounds = calculateLayerBounds(layer, canvasSize.width, canvasSize.height);

    // Convert bounds to container coordinates
    const cx = canvasInContainer.x + bounds.x * viewZoom;
    const cy = canvasInContainer.y + bounds.y * viewZoom;
    const halfW = (bounds.width * viewZoom) / 2;
    const halfH = (bounds.height * viewZoom) / 2;

    const handleSize = 12; // Slightly larger hit area than visual size

    // Check corners first (they take priority)
    const corners = [
      { id: 'tl', x: cx - halfW, y: cy - halfH },
      { id: 'tr', x: cx + halfW, y: cy - halfH },
      { id: 'bl', x: cx - halfW, y: cy + halfH },
      { id: 'br', x: cx + halfW, y: cy + halfH },
    ];

    for (const corner of corners) {
      if (Math.abs(containerX - corner.x) <= handleSize && Math.abs(containerY - corner.y) <= handleSize) {
        return corner.id;
      }
    }

    // Check edge midpoints
    const edges = [
      { id: 't', x: cx, y: cy - halfH },
      { id: 'b', x: cx, y: cy + halfH },
      { id: 'l', x: cx - halfW, y: cy },
      { id: 'r', x: cx + halfW, y: cy },
    ];

    for (const edge of edges) {
      if (Math.abs(containerX - edge.x) <= handleSize && Math.abs(containerY - edge.y) <= handleSize) {
        return edge.id;
      }
    }

    return null;
  }, [canvasSize, canvasInContainer, viewZoom, calculateLayerBounds]);

  // Get cursor style for handle
  const getCursorForHandle = useCallback((handle: string | null): string => {
    if (!handle) return 'crosshair';
    switch (handle) {
      case 'tl':
      case 'br':
        return 'nwse-resize';
      case 'tr':
      case 'bl':
        return 'nesw-resize';
      case 't':
      case 'b':
        return 'ns-resize';
      case 'l':
      case 'r':
        return 'ew-resize';
      default:
        return 'crosshair';
    }
  }, []);

  // Handle mouse down on overlay - select or start dragging
  const handleOverlayMouseDown = useCallback((e: React.MouseEvent) => {
    if (!editMode || !overlayRef.current || e.altKey) return;
    if (e.button !== 0) return; // Only left click

    const rect = overlayRef.current.getBoundingClientRect();
    // Use container coordinates directly (findLayerAtPosition handles zoom)
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // First check if clicking on a handle of the currently selected layer
    const selectedLayer = selectedLayerId ? layers.find(l => l?.id === selectedLayerId) : null;
    if (selectedLayer) {
      const handle = findHandleAtPosition(x, y, selectedLayer);
      if (handle) {
        // Start scaling
        setIsDragging(true);
        setDragLayerId(selectedLayer.id);
        setDragMode('scale');
        setDragHandle(handle);
        dragStart.current = {
          x: e.clientX,
          y: e.clientY,
          layerPosX: selectedLayer.position.x,
          layerPosY: selectedLayer.position.y,
          layerScaleX: selectedLayer.scale.x,
          layerScaleY: selectedLayer.scale.y,
        };
        return;
      }
    }

    const layer = findLayerAtPosition(x, y);

    if (layer) {
      // Select the layer
      const clip = clips.find(c => c.name === layer.name);
      if (clip) {
        selectClip(clip.id);
      }
      selectLayer(layer.id);

      // Start moving
      setIsDragging(true);
      setDragLayerId(layer.id);
      setDragMode('move');
      setDragHandle(null);
      dragStart.current = {
        x: e.clientX,
        y: e.clientY,
        layerPosX: layer.position.x,
        layerPosY: layer.position.y,
        layerScaleX: layer.scale.x,
        layerScaleY: layer.scale.y,
      };
    } else {
      // Click on empty space - deselect
      selectClip(null);
      selectLayer(null);
    }
  }, [editMode, findLayerAtPosition, findHandleAtPosition, clips, layers, selectedLayerId, selectClip, selectLayer]);

  // Handle mouse move on overlay - detect handle hover for cursor feedback
  // Actual dragging is handled by document-level listeners for reliability
  const handleOverlayMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging || !overlayRef.current) return;

    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if hovering over a handle of the selected layer
    const selectedLayer = selectedLayerId ? layers.find(l => l?.id === selectedLayerId) : null;
    if (selectedLayer) {
      const handle = findHandleAtPosition(x, y, selectedLayer);
      setHoverHandle(handle);
    } else {
      setHoverHandle(null);
    }
  }, [isDragging, selectedLayerId, layers, findHandleAtPosition]);

  // Handle mouse up - stop dragging
  const handleOverlayMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragLayerId(null);
    currentDragPos.current = { x: 0, y: 0 };
  }, []);

  // Use document-level listeners during drag to allow dragging beyond canvas bounds
  useEffect(() => {
    if (!isDragging) return;

    const handleDocumentMouseMove = (e: MouseEvent) => {
      if (!dragLayerId) return;

      const layer = layers.find(l => l?.id === dragLayerId);
      if (!layer) return;

      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;

      if (dragMode === 'scale' && dragHandle) {
        // Scaling mode
        // Calculate scale change based on handle being dragged
        // The scale is relative to the center of the layer

        // Get the original bounds to calculate scale factor
        const originalScaleX = dragStart.current.layerScaleX;
        const originalScaleY = dragStart.current.layerScaleY;

        // Convert pixel movement to scale factor
        // Positive dx/dy when dragging outward should increase scale
        const scaleSensitivity = 0.005 / viewZoom; // Adjust sensitivity based on zoom

        let newScaleX = originalScaleX;
        let newScaleY = originalScaleY;

        // Determine scale change based on which handle is being dragged
        switch (dragHandle) {
          case 'tl': // Top-left: decrease both when moving toward center
            newScaleX = originalScaleX - dx * scaleSensitivity;
            newScaleY = originalScaleY - dy * scaleSensitivity;
            break;
          case 'tr': // Top-right: increase X, decrease Y when moving outward
            newScaleX = originalScaleX + dx * scaleSensitivity;
            newScaleY = originalScaleY - dy * scaleSensitivity;
            break;
          case 'bl': // Bottom-left: decrease X, increase Y when moving outward
            newScaleX = originalScaleX - dx * scaleSensitivity;
            newScaleY = originalScaleY + dy * scaleSensitivity;
            break;
          case 'br': // Bottom-right: increase both when moving outward
            newScaleX = originalScaleX + dx * scaleSensitivity;
            newScaleY = originalScaleY + dy * scaleSensitivity;
            break;
          case 't': // Top edge: only Y scale
            newScaleY = originalScaleY - dy * scaleSensitivity;
            break;
          case 'b': // Bottom edge: only Y scale
            newScaleY = originalScaleY + dy * scaleSensitivity;
            break;
          case 'l': // Left edge: only X scale
            newScaleX = originalScaleX - dx * scaleSensitivity;
            break;
          case 'r': // Right edge: only X scale
            newScaleX = originalScaleX + dx * scaleSensitivity;
            break;
        }

        // Shift key for aspect ratio lock (uniform scaling)
        if (e.shiftKey && (dragHandle === 'tl' || dragHandle === 'tr' || dragHandle === 'bl' || dragHandle === 'br')) {
          // For corner handles, use the larger scale change for both axes
          const avgScale = (newScaleX / originalScaleX + newScaleY / originalScaleY) / 2;
          newScaleX = originalScaleX * avgScale;
          newScaleY = originalScaleY * avgScale;
        }

        // Clamp scale to reasonable values
        newScaleX = Math.max(0.01, Math.min(10, newScaleX));
        newScaleY = Math.max(0.01, Math.min(10, newScaleY));

        // Update layer directly for immediate visual feedback
        updateLayer(dragLayerId, {
          scale: { x: newScaleX, y: newScaleY },
        });

        // Also update clip transform for persistence
        const clip = clips.find(c => c.name === layer.name);
        if (clip) {
          updateClipTransform(clip.id, {
            scale: { x: newScaleX, y: newScaleY },
          });
        }
      } else {
        // Move mode
        // Convert pixel movement to normalized position change
        // Position is in normalized space where 1.0 = full canvas width/height
        const normalizedDx = (dx / viewZoom) / canvasSize.width;
        const normalizedDy = (dy / viewZoom) / canvasSize.height;

        const newPosX = dragStart.current.layerPosX + normalizedDx;
        const newPosY = dragStart.current.layerPosY + normalizedDy;

        currentDragPos.current = { x: newPosX, y: newPosY };

        // Update layer directly for immediate visual feedback (both box and video)
        updateLayer(dragLayerId, {
          position: { x: newPosX, y: newPosY, z: layer.position.z },
        });

        // Also update clip transform for persistence
        const clip = clips.find(c => c.name === layer.name);
        if (clip) {
          updateClipTransform(clip.id, {
            position: { x: newPosX, y: newPosY, z: 0 },
          });
        }
      }
    };

    const handleDocumentMouseUp = () => {
      setIsDragging(false);
      setDragLayerId(null);
      setDragMode('move');
      setDragHandle(null);
      currentDragPos.current = { x: 0, y: 0 };
    };

    document.addEventListener('mousemove', handleDocumentMouseMove);
    document.addEventListener('mouseup', handleDocumentMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleDocumentMouseMove);
      document.removeEventListener('mouseup', handleDocumentMouseUp);
    };
  }, [isDragging, dragLayerId, dragMode, dragHandle, viewZoom, canvasSize, layers, clips, updateClipTransform, updateLayer]);

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
      {/* Controls bar */}
      <div className="preview-controls">
        {sourceMonitorActive ? (
          <>
            <span className="preview-source-label" title={sourceMonitorFile!.name}>
              {sourceMonitorFile!.name}
            </span>
            <button
              className="preview-close-source-btn"
              onClick={closeSourceMonitor}
              title="Close source monitor [Esc]"
            >
              ✕
            </button>
          </>
        ) : (
          <>
            <button
              className={`preview-edit-btn ${editMode ? 'active' : ''}`}
              onClick={() => setEditMode(!editMode)}
              title="Toggle Edit Mode [Tab]"
            >
              {editMode ? '✓ Edit' : 'Edit'} <span className="menu-wip-badge">🐛</span>
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
            <div className="preview-comp-dropdown-wrapper">
              <button
                className="preview-comp-dropdown-btn"
                onClick={() => setSelectorOpen(!selectorOpen)}
                title="Select composition to display"
              >
                <span className="preview-comp-name">
                  {compositionId === null ? 'Active' : displayedComp?.name || 'Unknown'}
                </span>
                <span className="preview-comp-arrow">▼</span>
              </button>
              {selectorOpen && (
                <div className="preview-comp-dropdown" ref={dropdownRef} style={dropdownStyle}>
                  <button
                    className={`preview-comp-option ${compositionId === null ? 'active' : ''}`}
                    onClick={() => {
                      updatePanelData(panelId, { compositionId: null });
                      setSelectorOpen(false);
                    }}
                  >
                    Active Composition
                  </button>
                  <div className="preview-comp-separator" />
                  {compositions.map((comp) => (
                    <button
                      key={comp.id}
                      className={`preview-comp-option ${compositionId === comp.id ? 'active' : ''}`}
                      onClick={() => {
                        updatePanelData(panelId, { compositionId: comp.id });
                        setSelectorOpen(false);
                      }}
                    >
                      {comp.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
        <button
          className="preview-add-btn"
          onClick={() => addPreviewPanel(null)}
          title="Add another preview panel"
        >
          +
        </button>
        <button
          className="preview-close-btn"
          onClick={() => closePanelById(panelId)}
          title="Close this preview panel"
        >
          -
        </button>
      </div>

      {sourceMonitorActive ? (
        <SourceMonitor file={sourceMonitorFile!} onClose={closeSourceMonitor} />
      ) : (
        <>
          <StatsOverlay
            stats={engineStats}
            resolution={effectiveResolution}
            expanded={statsExpanded}
            onToggle={() => setStatsExpanded(!statsExpanded)}
          />

          <div className={`preview-canvas-wrapper ${showTransparencyGrid ? 'show-transparency-grid' : ''}`} style={viewTransform}>
            {!isEngineReady ? (
              <div className="loading">
                <div className="loading-spinner" />
                <p>Initializing WebGPU...</p>
              </div>
            ) : (
              <>
                <canvas
                  ref={canvasRef}
                  width={effectiveResolution.width}
                  height={effectiveResolution.height}
                  className="preview-canvas"
                  style={{
                    width: canvasSize.width,
                    height: canvasSize.height,
                  }}
                />
                {maskEditMode !== 'none' && (
                  <MaskOverlay
                    canvasWidth={effectiveResolution.width}
                    canvasHeight={effectiveResolution.height}
                  />
                )}
                {sam2Active && (
                  <SAM2Overlay
                    canvasWidth={effectiveResolution.width}
                    canvasHeight={effectiveResolution.height}
                    displayWidth={canvasSize.width}
                    displayHeight={canvasSize.height}
                  />
                )}
              </>
            )}
          </div>

          {/* Edit mode overlay - covers full container for pasteboard support */}
          {editMode && isEngineReady && (
            <canvas
              ref={overlayRef}
              width={containerSize.width || 100}
              height={containerSize.height || 100}
              className="preview-overlay-fullscreen"
              onMouseDown={handleOverlayMouseDown}
              onMouseMove={handleOverlayMouseMove}
              onMouseUp={handleOverlayMouseUp}
              onMouseLeave={handleOverlayMouseUp}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: containerSize.width || '100%',
                height: containerSize.height || '100%',
                cursor: isDragging
                  ? (dragMode === 'scale' ? getCursorForHandle(dragHandle) : 'grabbing')
                  : getCursorForHandle(hoverHandle),
                pointerEvents: 'auto',
              }}
            />
          )}

          {editMode && (
            <div className="preview-edit-hint">
              Drag: Move | Handles: Scale (Shift: Lock Ratio) | Scroll: Zoom | Alt+Drag: Pan
            </div>
          )}

          {/* Bottom-left controls */}
          <div className="preview-controls-bottom">
            {/* Transparency grid toggle */}
            <button
              className={`preview-transparency-toggle ${showTransparencyGrid ? 'active' : ''}`}
              onClick={() => setShowTransparencyGrid(!showTransparencyGrid)}
              title="Toggle transparency grid (checkerboard)"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <rect x="0" y="0" width="4" height="4" opacity="0.6" />
                <rect x="8" y="0" width="4" height="4" opacity="0.6" />
                <rect x="4" y="4" width="4" height="4" opacity="0.6" />
                <rect x="12" y="4" width="4" height="4" opacity="0.6" />
                <rect x="0" y="8" width="4" height="4" opacity="0.6" />
                <rect x="8" y="8" width="4" height="4" opacity="0.6" />
                <rect x="4" y="12" width="4" height="4" opacity="0.6" />
                <rect x="12" y="12" width="4" height="4" opacity="0.6" />
              </svg>
            </button>

            <div className="preview-quality-dropdown-wrapper" ref={qualityDropdownRef}>
              <button
                className="preview-quality-dropdown-btn"
                onClick={() => setQualityOpen(!qualityOpen)}
                title="Preview quality (affects performance)"
              >
                <span className="preview-quality-label">
                  {previewQuality === 1 ? 'Full' : previewQuality === 0.5 ? 'Half' : 'Quarter'}
                </span>
                <span className="preview-comp-arrow">▼</span>
              </button>
              {qualityOpen && (
                <div className="preview-quality-dropdown">
                  {([
                    { value: 1 as PreviewQuality, label: 'Full', desc: '100%' },
                    { value: 0.5 as PreviewQuality, label: 'Half', desc: '50%' },
                    { value: 0.25 as PreviewQuality, label: 'Quarter', desc: '25%' },
                  ]).map(({ value, label, desc }) => (
                    <button
                      key={value}
                      className={`preview-quality-option ${previewQuality === value ? 'active' : ''}`}
                      onClick={() => {
                        setPreviewQuality(value);
                        setQualityOpen(false);
                      }}
                    >
                      {label} <span className="preview-quality-desc">{desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
