// Individual preview slot for the Multi Preview panel
// Stripped-down canvas that renders a single composition independently

import { useEffect, useRef, useState, useCallback } from 'react';
import { useEngine } from '../../hooks/useEngine';
import { useMediaStore } from '../../stores/mediaStore';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { renderScheduler } from '../../services/renderScheduler';
import { engine } from '../../engine/WebGPUEngine';

interface MultiPreviewSlotProps {
  panelId: string;
  slotIndex: number;
  compositionId: string | null;
  showTransparencyGrid: boolean;
  onCompositionChange: (compositionId: string | null) => void;
}

export function MultiPreviewSlot({
  panelId,
  slotIndex,
  compositionId,
  showTransparencyGrid,
  onCompositionChange,
}: MultiPreviewSlotProps) {
  const { isEngineReady } = useEngine();
  const compositions = useMediaStore((s) => s.compositions);
  const activeCompositionId = useMediaStore((s) => s.activeCompositionId);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1920, height: 1080 });
  const [selectorOpen, setSelectorOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Unique target ID for this slot
  const targetId = `mp-${panelId}-slot-${slotIndex}`;

  // Determine which composition to display
  const displayedCompId = compositionId ?? activeCompositionId;
  const displayedComp = compositions.find((c) => c.id === displayedCompId);

  const effectiveResolution = displayedComp
    ? { width: displayedComp.width, height: displayedComp.height }
    : useSettingsStore.getState().outputResolution;

  // Register canvas with engine + render target store + render scheduler
  useEffect(() => {
    if (!isEngineReady || !canvasRef.current) return;

    const source = compositionId
      ? { type: 'composition' as const, compositionId }
      : { type: 'activeComp' as const };

    const isIndependent = source.type !== 'activeComp';

    const gpuContext = engine.registerTargetCanvas(targetId, canvasRef.current);
    if (!gpuContext) return;

    useRenderTargetStore.getState().registerTarget({
      id: targetId,
      name: `Multi Preview Slot ${slotIndex + 1}`,
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
      renderScheduler.register(targetId);
    }

    return () => {
      if (isIndependent) {
        renderScheduler.unregister(targetId);
      }
      useRenderTargetStore.getState().unregisterTarget(targetId);
      engine.unregisterTargetCanvas(targetId);
    };
  }, [isEngineReady, targetId, compositionId, slotIndex, showTransparencyGrid]);

  // Sync transparency grid flag without full re-registration
  useEffect(() => {
    if (!isEngineReady) return;
    useRenderTargetStore.getState().setTargetTransparencyGrid(targetId, showTransparencyGrid);
    engine.requestRender();
  }, [isEngineReady, targetId, showTransparencyGrid]);

  // ResizeObserver for aspect-ratio-correct sizing
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;
      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;
      if (containerWidth === 0 || containerHeight === 0) return;

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

      setCanvasSize({ width: Math.floor(width), height: Math.floor(height) });
    };

    updateSize();
    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => resizeObserver.disconnect();
  }, [effectiveResolution.width, effectiveResolution.height]);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!selectorOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setSelectorOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [selectorOpen]);

  const handleSelect = useCallback(
    (id: string | null) => {
      onCompositionChange(id);
      setSelectorOpen(false);
    },
    [onCompositionChange]
  );

  return (
    <div className="multi-preview-slot" ref={containerRef}>
      {/* Hover-visible composition dropdown */}
      <div className="multi-preview-slot-dropdown-wrapper" ref={dropdownRef}>
        <button
          className="multi-preview-slot-dropdown-btn"
          onClick={() => setSelectorOpen(!selectorOpen)}
          title="Select composition"
        >
          <span className="multi-preview-slot-comp-name">
            {compositionId === null ? 'Active' : displayedComp?.name || 'None'}
          </span>
          <span className="preview-comp-arrow">▼</span>
        </button>
        {selectorOpen && (
          <div className="multi-preview-slot-dropdown">
            <button
              className={`preview-comp-option ${compositionId === null ? 'active' : ''}`}
              onClick={() => handleSelect(null)}
            >
              Active Composition
            </button>
            <div className="preview-comp-separator" />
            {compositions.map((comp) => (
              <button
                key={comp.id}
                className={`preview-comp-option ${compositionId === comp.id ? 'active' : ''}`}
                onClick={() => handleSelect(comp.id)}
              >
                {comp.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Canvas */}
      <div className={`preview-canvas-wrapper ${showTransparencyGrid ? 'show-transparency-grid' : ''}`}>
        {!isEngineReady ? (
          <div className="multi-preview-slot-placeholder">
            <div className="loading-spinner" />
          </div>
        ) : !displayedComp && compositionId !== null ? (
          <div className="multi-preview-slot-placeholder">
            <span>Select Composition</span>
          </div>
        ) : (
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
        )}
      </div>
    </div>
  );
}
