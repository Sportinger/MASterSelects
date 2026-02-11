// SlotGrid - Resolume-style grid with row labels (layers) on left, column numbers on top
// Multi-layer playback: each row (A-D) can have an active composition playing simultaneously
// Click = activate on layer + play from start, Drag = reorder/move to any slot
// Column header click = activate all compositions in that column

import { Fragment, useCallback, useEffect, useMemo, useRef, useState, memo } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { playheadState } from '../../services/layerBuilder';
import { layerPlaybackManager } from '../../services/layerPlaybackManager';
import { animateSlotGrid } from './slotGridAnimation';
import { MiniTimeline } from './MiniTimeline';
import type { Composition } from '../../stores/mediaStore';

interface SlotGridProps {
  opacity: number;
}

const SLOT_SIZE = 100; // fixed slot size in px
const GRID_COLS = 12;
const GRID_ROWS = 4;
const TOTAL_SLOTS = GRID_COLS * GRID_ROWS;
const LABEL_WIDTH = 32;

export function SlotGrid({ opacity }: SlotGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const activeCompositionId = useMediaStore(state => state.activeCompositionId);
  const slotAssignments = useMediaStore(state => state.slotAssignments);
  const activeLayerSlots = useMediaStore(state => state.activeLayerSlots);
  const openCompositionTab = useMediaStore(state => state.openCompositionTab);
  const deactivateLayer = useMediaStore(state => state.deactivateLayer);
  const activateColumn = useMediaStore(state => state.activateColumn);
  const moveSlot = useMediaStore(state => state.moveSlot);
  const unassignSlot = useMediaStore(state => state.unassignSlot);
  const assignMediaFileToSlot = useMediaStore(state => state.assignMediaFileToSlot);
  const getSlotMap = useMediaStore(state => state.getSlotMap);
  const compositions = useMediaStore(state => state.compositions);
  const files = useMediaStore(state => state.files);

  // Build mediaFileId → thumbnailUrl lookup
  const thumbnailMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const f of files) {
      if (f.thumbnailUrl) map.set(f.id, f.thumbnailUrl);
    }
    return map;
  }, [files]);

  // Build a set of active composition IDs from activeLayerSlots
  const activeLayerCompIds = useMemo(() => {
    const ids = new Set<string>();
    for (const compId of Object.values(activeLayerSlots)) {
      if (compId) ids.add(compId);
    }
    return ids;
  }, [activeLayerSlots]);

  // Drag state
  const [dragCompId, setDragCompId] = useState<string | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [isExternalDrag, setIsExternalDrag] = useState(false);

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; compId: string } | null>(null);

  // Sync LayerPlaybackManager when activeLayerSlots changes
  useEffect(() => {
    const { activeCompositionId } = useMediaStore.getState();

    // Determine which layers need activation/deactivation in the playback manager
    const currentManagerLayers = new Set(layerPlaybackManager.getActiveLayerIndices());
    const desiredLayers = new Map<number, string>();

    for (const [key, compId] of Object.entries(activeLayerSlots)) {
      if (compId && compId !== activeCompositionId) {
        desiredLayers.set(Number(key), compId);
      }
    }

    // Deactivate layers no longer needed
    for (const layerIndex of currentManagerLayers) {
      if (!desiredLayers.has(layerIndex)) {
        layerPlaybackManager.deactivateLayer(layerIndex);
      }
    }

    // Activate new layers
    const { compositions } = useMediaStore.getState();
    for (const [layerIndex, compId] of desiredLayers) {
      const existing = layerPlaybackManager.getLayerState(layerIndex);
      if (!existing || existing.compositionId !== compId) {
        // Resume from saved playhead position so background layers don't restart
        const comp = compositions.find(c => c.id === compId);
        const savedPosition = comp?.timelineData?.playheadPosition ?? 0;
        layerPlaybackManager.activateLayer(layerIndex, compId, savedPosition);
      }
    }
  }, [activeLayerSlots]);

  // Dismiss context menu on click-outside
  useEffect(() => {
    if (!contextMenu) return;
    const dismiss = () => setContextMenu(null);
    document.addEventListener('mousedown', dismiss);
    return () => document.removeEventListener('mousedown', dismiss);
  }, [contextMenu]);

  // Handle Ctrl+Shift+Scroll on the SlotGrid itself
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        if (e.deltaY > 0) {
          // Zoom out → show grid (always allowed)
          animateSlotGrid(1);
        } else {
          // Zoom in → back to timeline, only if hovering a filled slot
          const target = e.target as HTMLElement;
          const slotEl = target.closest('.slot-grid-item:not(.empty)');
          if (!slotEl) return; // Not over a filled slot — block transition
          const compId = slotEl.getAttribute('data-comp-id');
          if (compId) {
            useMediaStore.getState().openCompositionTab(compId, { skipAnimation: true });
          }
          animateSlotGrid(0);
        }
      } else {
        // Stop propagation so timeline's wheel handler doesn't preventDefault
        // This lets the container scroll natively
        e.stopPropagation();
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  // Click = activate on layer + open in editor + play from start
  const handleSlotClick = useCallback((comp: Composition, slotIndex: number) => {
    const layerIndex = Math.floor(slotIndex / GRID_COLS);
    useMediaStore.getState().activateOnLayer(comp.id, layerIndex);

    // Open in timeline editor and play from start
    openCompositionTab(comp.id, { skipAnimation: true, playFromStart: true });
  }, [openCompositionTab]);

  // Click empty slot = fully deactivate that layer
  const handleEmptySlotClick = useCallback((slotIndex: number) => {
    const layerIndex = Math.floor(slotIndex / GRID_COLS);
    const { activeLayerSlots, activeCompositionId } = useMediaStore.getState();
    const compOnLayer = activeLayerSlots[layerIndex];

    deactivateLayer(layerIndex);

    // Check which layers are still active after removing this one
    const remaining = { ...activeLayerSlots };
    delete remaining[layerIndex];
    const stillActive = Object.entries(remaining)
      .filter(([, id]) => id != null)
      .sort(([a], [b]) => Number(a) - Number(b)); // prefer top layer (A first)

    if (compOnLayer && compOnLayer === activeCompositionId) {
      // Deactivated comp was the editor-active one
      if (stillActive.length > 0) {
        // Remember if playback was running before the switch
        const wasPlaying = useTimelineStore.getState().isPlaying;
        // Switch editor to another still-active layer's comp
        // (openCompositionTab → loadState → pause() internally, so we re-play after)
        useMediaStore.getState().openCompositionTab(stillActive[0][1]!, { skipAnimation: true });
        if (wasPlaying) {
          // Resume playback after comp switch so other layers keep running
          setTimeout(() => useTimelineStore.getState().play(), 50);
        }
      } else {
        // No layers left — stop everything
        useTimelineStore.getState().stop();
        useMediaStore.getState().setActiveComposition(null);
      }
    }
    // If deactivated comp was NOT the editor-active one, other layers keep playing
  }, [deactivateLayer]);

  // Click column header = activate all compositions in that column
  const handleColumnClick = useCallback((colIndex: number) => {
    const slotMap = getSlotMap(TOTAL_SLOTS);
    activateColumn(colIndex);
    // Open topmost (row A first) filled slot in that column in editor
    for (let row = 0; row < GRID_ROWS; row++) {
      const comp = slotMap[row * GRID_COLS + colIndex];
      if (comp) {
        openCompositionTab(comp.id, { skipAnimation: true, playFromStart: true });
        break;
      }
    }
  }, [activateColumn, getSlotMap, openCompositionTab]);

  // Right-click context menu on filled slots
  const handleContextMenu = useCallback((e: React.MouseEvent, comp: Composition) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, compId: comp.id });
  }, []);

  const handleRemoveFromSlot = useCallback(() => {
    if (contextMenu) {
      unassignSlot(contextMenu.compId);
      setContextMenu(null);
    }
  }, [contextMenu, unassignSlot]);

  // Drag handlers — track comp ID, not slot index
  const handleDragStart = useCallback((e: React.DragEvent, comp: Composition) => {
    setDragCompId(comp.id);
    setIsExternalDrag(false);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', comp.id);
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, slotIndex: number) => {
    e.preventDefault();
    const types = e.dataTransfer.types;
    const isExternal = types.includes('application/x-media-file-id') || types.includes('application/x-composition-id');
    if (isExternal) {
      e.dataTransfer.dropEffect = 'copy';
      setIsExternalDrag(true);
    } else {
      e.dataTransfer.dropEffect = 'move';
    }
    setDragOverIndex(slotIndex);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOverIndex(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, toSlotIndex: number) => {
    e.preventDefault();
    setDragCompId(null);
    setDragOverIndex(null);
    setIsExternalDrag(false);

    // Check for external drops from MediaPanel
    const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
    if (mediaFileId) {
      assignMediaFileToSlot(mediaFileId, toSlotIndex);
      return;
    }

    const compositionId = e.dataTransfer.getData('application/x-composition-id');
    if (compositionId) {
      moveSlot(compositionId, toSlotIndex);
      return;
    }

    // Internal slot drag (text/plain = comp ID)
    const compId = e.dataTransfer.getData('text/plain') || dragCompId;
    if (compId) {
      moveSlot(compId, toSlotIndex);
    }
  }, [dragCompId, moveSlot, assignMediaFileToSlot]);

  const handleDragEnd = useCallback(() => {
    setDragCompId(null);
    setDragOverIndex(null);
    setIsExternalDrag(false);
  }, []);

  // Build slot map from assignments (reacts to slotAssignments + compositions changes)
  const slotMap = useMemo(() => {
    return getSlotMap(TOTAL_SLOTS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getSlotMap, slotAssignments, compositions]);

  return (
    <div
      ref={containerRef}
      className="slot-grid-container"
      style={{ opacity }}
    >
      <div
        className="slot-grid-resolume"
        style={{
          gridTemplateColumns: `${LABEL_WIDTH}px repeat(${GRID_COLS}, ${SLOT_SIZE}px)`,
          gridTemplateRows: `24px repeat(${GRID_ROWS}, ${SLOT_SIZE}px)`,
        }}
      >
        {/* Empty corner */}
        <div className="slot-grid-corner" />

        {/* Column headers — clickable to activate column */}
        {Array.from({ length: GRID_COLS }, (_, i) => (
          <div
            key={`col-${i}`}
            className="slot-grid-col-header slot-grid-col-header-clickable"
            onClick={() => handleColumnClick(i)}
            title={`Activate column ${i + 1}`}
          >
            {i + 1}
          </div>
        ))}

        {/* Rows: label + slots */}
        {Array.from({ length: GRID_ROWS }, (_, rowIndex) => (
          <Fragment key={`row-${rowIndex}`}>
            <div className="slot-grid-row-label">
              {String.fromCharCode(65 + rowIndex)}
            </div>
            {Array.from({ length: GRID_COLS }, (_, colIndex) => {
              const slotIndex = rowIndex * GRID_COLS + colIndex;
              const comp = slotMap[slotIndex];
              const isDragOver = slotIndex === dragOverIndex && (dragCompId !== null || isExternalDrag);

              if (comp) {
                const isEditorActive = comp.id === activeCompositionId;
                const isLayerActive = activeLayerCompIds.has(comp.id);
                const isSelf = comp.id === dragCompId;
                // Find thumbnail from first video clip's media file
                const firstVideoClip = comp.timelineData?.clips?.find(
                  (c: { sourceType: string; mediaFileId?: string }) => c.sourceType === 'video' && c.mediaFileId
                );
                const thumbUrl = firstVideoClip?.mediaFileId ? thumbnailMap.get(firstVideoClip.mediaFileId) : undefined;
                return (
                  <div
                    key={slotIndex}
                    className={
                      `slot-grid-item` +
                      `${isEditorActive ? ' active' : ''}` +
                      `${isLayerActive && !isEditorActive ? ' layer-active' : ''}` +
                      `${isDragOver && !isSelf ? ' drag-over' : ''}`
                    }
                    data-comp-id={comp.id}
                    style={thumbUrl ? { backgroundImage: `url(${thumbUrl})`, backgroundSize: 'cover', backgroundPosition: 'center' } : undefined}
                    onClick={() => handleSlotClick(comp, slotIndex)}
                    onContextMenu={(e) => handleContextMenu(e, comp)}
                    title={comp.name}
                    draggable
                    onDragStart={(e) => handleDragStart(e, comp)}
                    onDragEnter={handleDragEnter}
                    onDragOver={(e) => handleDragOver(e, slotIndex)}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, slotIndex)}
                    onDragEnd={handleDragEnd}
                  >
                    <MiniTimeline
                      timelineData={comp.timelineData}
                      compositionName={comp.name}
                      compositionDuration={comp.duration}
                      isActive={isEditorActive}
                      width={SLOT_SIZE - 4}
                      height={SLOT_SIZE - 4}
                    />
                    <div className="slot-grid-name">{comp.name}</div>
                    <SlotTimeOverlay
                      compId={comp.id}
                      duration={comp.duration}
                      isActive={isEditorActive || isLayerActive}
                      layerIndex={rowIndex}
                      slotSize={SLOT_SIZE - 4}
                    />
                  </div>
                );
              }

              return (
                <div
                  key={slotIndex}
                  className={`slot-grid-item empty${isDragOver ? ' drag-over' : ''}`}
                  onClick={() => handleEmptySlotClick(slotIndex)}
                  onDragEnter={handleDragEnter}
                  onDragOver={(e) => handleDragOver(e, slotIndex)}
                  onDragLeave={handleDragLeave}
                  onDrop={(e) => handleDrop(e, slotIndex)}
                />
              );
            })}
          </Fragment>
        ))}
      </div>

      {/* Context menu */}
      {contextMenu && (
        <div
          className="slot-grid-context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <button onClick={handleRemoveFromSlot}>Remove from Slot</button>
        </div>
      )}
    </div>
  );
}

function fmtTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

/** Slot overlay — playhead line + live time / duration display via rAF */
const SlotTimeOverlay = memo(function SlotTimeOverlay({
  compId,
  duration,
  isActive,
  layerIndex,
  slotSize,
}: {
  compId: string;
  duration: number;
  isActive: boolean;
  layerIndex: number;
  slotSize: number;
}) {
  const lineRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef<HTMLDivElement>(null);
  const lastPosRef = useRef(0);

  useEffect(() => {
    const line = lineRef.current;
    const timeEl = timeRef.current;
    if (!line || !timeEl || duration <= 0) return;

    // Always show duration
    const durationStr = fmtTime(duration);

    if (!isActive) {
      line.style.display = 'none';
      timeEl.textContent = `00:00.00 / ${durationStr}`;
      lastPosRef.current = 0;
      return;
    }

    line.style.display = '';
    let rafId: number;
    const padding = 3;
    const trackWidth = slotSize - padding * 2;

    const update = () => {
      // Dynamically check if this comp is the editor comp (no prop dependency)
      const isEditor = useMediaStore.getState().activeCompositionId === compId;
      let pos: number;
      if (isEditor) {
        pos = playheadState.isUsingInternalPosition
          ? playheadState.position
          : useTimelineStore.getState().playheadPosition;
      } else {
        const layerState = layerPlaybackManager.getLayerState(layerIndex);
        // Fall back to last known position if playback manager isn't ready yet
        pos = layerState ? layerPlaybackManager.getLayerTime(layerState) : lastPosRef.current;
      }
      lastPosRef.current = pos;
      const pct = Math.max(0, Math.min(1, pos / duration));
      line.style.left = `${padding + pct * trackWidth}px`;
      timeEl.textContent = `${fmtTime(pos)} / ${durationStr}`;
      rafId = requestAnimationFrame(update);
    };

    rafId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(rafId);
  }, [compId, duration, isActive, layerIndex, slotSize]);

  return (
    <>
      <div
        ref={lineRef}
        className="slot-grid-playhead"
      />
      <div
        ref={timeRef}
        className="slot-grid-time"
      />
    </>
  );
});
