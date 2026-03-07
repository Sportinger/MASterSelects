// Preview canvas component with After Effects-style editing overlay

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Logger } from '../../services/logger';

const log = Logger.create('Preview');
import { useEngine } from '../../hooks/useEngine';
import { useEngineStore } from '../../stores/engineStore';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { useDockStore } from '../../stores/dockStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { MaskOverlay } from './MaskOverlay';
import { SAM2Overlay } from './SAM2Overlay';
import { SourceMonitor } from './SourceMonitor';
import { StatsOverlay } from './StatsOverlay';
import { PreviewControls } from './PreviewControls';
import { PreviewBottomControls } from './PreviewBottomControls';
import { useEditModeOverlay } from './useEditModeOverlay';
import { useLayerDrag } from './useLayerDrag';
import { useSAM2Store } from '../../stores/sam2Store';
import { renderScheduler } from '../../services/renderScheduler';
import { engine } from '../../engine/WebGPUEngine';
import type { RenderSource } from '../../types/renderTarget';

interface PreviewProps {
  panelId: string;
  compositionId: string | null; // null = active composition
  showTransparencyGrid: boolean; // per-tab transparency toggle
}

export function Preview({ panelId, compositionId, showTransparencyGrid }: PreviewProps) {
  const { isEngineReady } = useEngine();
  const engineStats = useEngineStore(s => s.engineStats);
  const { clips, selectedClipIds, selectClip, updateClipTransform, maskEditMode, layers, selectedLayerId, selectLayer, updateLayer } = useTimelineStore(useShallow(s => ({
    clips: s.clips,
    selectedClipIds: s.selectedClipIds,
    selectClip: s.selectClip,
    updateClipTransform: s.updateClipTransform,
    maskEditMode: s.maskEditMode,
    layers: s.layers,
    selectedLayerId: s.selectedLayerId,
    selectLayer: s.selectLayer,
    updateLayer: s.updateLayer,
  })));
  const { compositions, activeCompositionId } = useMediaStore(useShallow(s => ({
    compositions: s.compositions,
    activeCompositionId: s.activeCompositionId,
  })));
  const { addPreviewPanel, updatePanelData, closePanelById } = useDockStore(useShallow(s => ({
    addPreviewPanel: s.addPreviewPanel,
    updatePanelData: s.updatePanelData,
    closePanelById: s.closePanelById,
  })));
  const { previewQuality, setPreviewQuality } = useSettingsStore(useShallow(s => ({
    previewQuality: s.previewQuality,
    setPreviewQuality: s.setPreviewQuality,
  })));
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
  useEffect(() => {
    if (!isEngineReady || !canvasRef.current) return;

    const source: RenderSource = compositionId
      ? { type: 'composition', compositionId }
      : slotPreviewActive && previewCompositionId
        ? { type: 'composition', compositionId: previewCompositionId }
        : { type: 'activeComp' };

    const isIndependent = source.type !== 'activeComp';

    log.debug(`[${panelId}] Registering render target`, { source, isIndependent });

    const gpuContext = engine.registerTargetCanvas(panelId, canvasRef.current);
    if (!gpuContext) return;

    useRenderTargetStore.getState().registerTarget({
      id: panelId,
      name: 'Preview',
      source,
      destinationType: 'canvas',
      enabled: true,
      showTransparencyGrid,
      canvas: canvasRef.current,
      context: gpuContext,
      window: null,
      isFullscreen: false,
    });

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

  // Sync per-tab transparency grid flag
  useEffect(() => {
    if (!isEngineReady) return;
    useRenderTargetStore.getState().setTargetTransparencyGrid(panelId, showTransparencyGrid);
    engine.requestRender();
  }, [isEngineReady, panelId, showTransparencyGrid]);

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

  // Adjust dropdown position when opened
  useEffect(() => {
    if (selectorOpen && dropdownRef.current) {
      const rect = dropdownRef.current.getBoundingClientRect();
      const style: React.CSSProperties = {};

      if (rect.left < 8) {
        style.left = '0';
        style.right = 'auto';
      }
      if (rect.right > window.innerWidth - 8) {
        style.right = '0';
        style.left = 'auto';
      }
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

  // Sync layer selection when clip is selected in timeline (for edit mode)
  useEffect(() => {
    if (!selectedClipId || !editMode) return;

    const clip = clips.find(c => c.id === selectedClipId);
    if (clip) {
      const layer = layers.find(l => l?.name === clip.name);
      if (layer && layer.id !== selectedLayerId) {
        selectLayer(layer.id);
      }
    }
  }, [selectedClipId, editMode, clips, layers, selectedLayerId, selectLayer]);

  // Calculate canvas size to fit container while maintaining aspect ratio
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      if (containerWidth === 0 || containerHeight === 0) return;

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
      setViewPan(prev => ({
        x: prev.x - e.deltaY,
        y: prev.y
      }));
    } else {
      const rect = containerRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.1, Math.min(150, viewZoom * zoomFactor));

      const containerCenterX = containerSize.width / 2;
      const containerCenterY = containerSize.height / 2;

      const worldX = (mouseX - containerCenterX - viewPan.x) / viewZoom;
      const worldY = (mouseY - containerCenterY - viewPan.y) / viewZoom;

      const newPanX = mouseX - worldX * newZoom - containerCenterX;
      const newPanY = mouseY - worldY * newZoom - containerCenterY;

      setViewZoom(newZoom);
      setViewPan({ x: newPanX, y: newPanY });
    }
  }, [editMode, viewZoom, viewPan, containerSize]);

  // Tab key to toggle edit mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
  const canvasInContainer = useMemo(() => {
    const scaledWidth = canvasSize.width * viewZoom;
    const scaledHeight = canvasSize.height * viewZoom;

    const centerX = (containerSize.width - scaledWidth) / 2;
    const centerY = (containerSize.height - scaledHeight) / 2;

    return {
      x: centerX + viewPan.x,
      y: centerY + viewPan.y,
      width: scaledWidth,
      height: scaledHeight,
    };
  }, [containerSize, canvasSize, viewZoom, viewPan]);

  // Edit mode helpers (bounding box calculation, hit testing, cursor mapping)
  const { calculateLayerBounds, findLayerAtPosition, findHandleAtPosition, getCursorForHandle } =
    useEditModeOverlay({ effectiveResolution, canvasSize, canvasInContainer, viewZoom, layers });

  // Layer drag logic (move/scale, overlay drawing, document-level listeners)
  const { isDragging, dragMode, dragHandle, hoverHandle, handleOverlayMouseDown, handleOverlayMouseMove, handleOverlayMouseUp } =
    useLayerDrag({
      editMode, overlayRef, canvasSize, canvasInContainer, viewZoom,
      layers, clips, selectedLayerId, selectedClipId,
      selectClip, selectLayer, updateClipTransform, updateLayer,
      calculateLayerBounds, findLayerAtPosition, findHandleAtPosition,
    });

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
      <PreviewControls
        sourceMonitorActive={sourceMonitorActive}
        sourceMonitorFileName={sourceMonitorFile?.name ?? null}
        closeSourceMonitor={closeSourceMonitor}
        editMode={editMode}
        setEditMode={setEditMode}
        viewZoom={viewZoom}
        resetView={resetView}
        compositionId={compositionId}
        displayedComp={displayedComp}
        selectorOpen={selectorOpen}
        setSelectorOpen={setSelectorOpen}
        dropdownRef={dropdownRef}
        dropdownStyle={dropdownStyle}
        compositions={compositions}
        updatePanelData={updatePanelData}
        panelId={panelId}
        addPreviewPanel={addPreviewPanel}
        closePanelById={closePanelById}
      />

      {/* Source monitor overlay - shown on top when active */}
      {sourceMonitorActive && (
        <SourceMonitor file={sourceMonitorFile!} onClose={closeSourceMonitor} />
      )}

      {/* Engine canvas + overlays - always in DOM to keep WebGPU registration alive */}
      <div style={{ display: sourceMonitorActive ? 'none' : 'contents' }}>
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
        <PreviewBottomControls
          showTransparencyGrid={showTransparencyGrid}
          compositionId={compositionId}
          panelId={panelId}
          updatePanelData={updatePanelData}
          previewQuality={previewQuality}
          setPreviewQuality={setPreviewQuality}
          qualityOpen={qualityOpen}
          setQualityOpen={setQualityOpen}
          qualityDropdownRef={qualityDropdownRef}
        />
      </div>
    </div>
  );
}
