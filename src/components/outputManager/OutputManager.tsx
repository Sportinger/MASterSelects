// OutputManager - main component for managing output targets
// Renders in a popup window, manages source routing for all output windows

import { useEffect, useState, useCallback, useRef } from 'react';
import { TargetList } from './TargetList';
import { TargetPreview } from './TargetPreview';
import { TabBar } from './TabBar';
import { SliceInputOverlay } from './SliceInputOverlay';
import { SliceOutputOverlay } from './SliceOutputOverlay';
import { useSliceStore, getSavedTargetMeta } from '../../stores/sliceStore';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { closeOutputManager } from './OutputManagerBoot';

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 5;

export function OutputManager() {
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);
  const activeTab = useSliceStore((s) => s.activeTab);
  const saveToLocalStorage = useSliceStore((s) => s.saveToLocalStorage);
  const loadFromLocalStorage = useSliceStore((s) => s.loadFromLocalStorage);

  // Zoom/pan state
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // On mount: load saved configs and restore saved targets as deactivated entries
  useEffect(() => {
    loadFromLocalStorage();

    // Restore saved targets that don't exist yet in renderTargetStore
    // They'll appear as "closed" (grayed out) with a Restore button
    const savedTargets = getSavedTargetMeta();
    const store = useRenderTargetStore.getState();
    for (const saved of savedTargets) {
      if (!store.targets.has(saved.id)) {
        store.registerTarget({
          id: saved.id,
          name: saved.name,
          source: saved.source,
          destinationType: 'window',
          enabled: false,
          showTransparencyGrid: false,
          canvas: null,
          context: null,
          window: null,
          isFullscreen: saved.isFullscreen ?? false,
        });
      }
    }
  }, [loadFromLocalStorage]);

  const handleSaveAndExit = useCallback(() => {
    saveToLocalStorage();
    closeOutputManager();
  }, [saveToLocalStorage]);

  // Mouse wheel zoom (centered on cursor)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const rect = wrapper.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    setZoom((prevZoom) => {
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prevZoom * factor));
      const scale = newZoom / prevZoom;

      // Adjust pan to keep the point under cursor fixed
      setPanX((prev) => mouseX - scale * (mouseX - prev));
      setPanY((prev) => mouseY - scale * (mouseY - prev));

      return newZoom;
    });
  }, []);

  // Middle mouse button pan
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button === 1) { // middle mouse
      e.preventDefault();
      panStartRef.current = { x: e.clientX, y: e.clientY, panX, panY };
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    }
  }, [panX, panY]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const start = panStartRef.current;
    if (!start) return;

    setPanX(start.panX + (e.clientX - start.x));
    setPanY(start.panY + (e.clientY - start.y));
  }, []);

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    if (panStartRef.current) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      panStartRef.current = null;
    }
  }, []);

  // Double-click to reset zoom/pan
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setZoom(1);
      setPanX(0);
      setPanY(0);
    }
  }, []);

  return (
    <div className="om-container">
      <div className="om-header">
        <h2 className="om-title">Output Manager</h2>
      </div>
      <div className="om-body">
        <div className="om-main">
          <TabBar />
          <div
            ref={wrapperRef}
            className="om-preview-wrapper"
            onWheel={handleWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onDoubleClick={handleDoubleClick}
          >
            <div
              className="om-preview-viewport"
              style={{
                transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
                transformOrigin: '0 0',
              }}
            >
              <TargetPreview targetId={selectedTargetId} />
              {selectedTargetId && activeTab === 'input' && (
                <SliceInputOverlay targetId={selectedTargetId} width={1920} height={1080} />
              )}
              {selectedTargetId && activeTab === 'output' && (
                <SliceOutputOverlay targetId={selectedTargetId} width={1920} height={1080} />
              )}
            </div>
          </div>
          <div className="om-footer">
            <span className="om-zoom-label">{Math.round(zoom * 100)}%</span>
            <button className="om-save-exit-btn" onClick={handleSaveAndExit}>
              Save &amp; Exit
            </button>
          </div>
        </div>
        <div className="om-sidebar">
          <TargetList
            selectedTargetId={selectedTargetId}
            onSelect={setSelectedTargetId}
          />
        </div>
      </div>
    </div>
  );
}
