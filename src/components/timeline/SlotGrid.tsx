// SlotGrid - Resolume-style grid with row labels (layers) on left, column numbers on top
// Compositions fill slots left-to-right, top-to-bottom; remaining slots are empty
// Click = play from start (stay in grid), Drag = reorder/move to any slot, Bottom strip = preview

import { Fragment, useCallback, useEffect, useMemo, useRef, useState, memo } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { playheadState } from '../../services/layerBuilder';
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
  const previewCompositionId = useMediaStore(state => state.previewCompositionId);
  const slotAssignments = useMediaStore(state => state.slotAssignments);
  const openCompositionTab = useMediaStore(state => state.openCompositionTab);
  const moveSlot = useMediaStore(state => state.moveSlot);
  const setPreviewComposition = useMediaStore(state => state.setPreviewComposition);
  const getSlotMap = useMediaStore(state => state.getSlotMap);

  // Drag state
  const [dragCompId, setDragCompId] = useState<string | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

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
            useMediaStore.getState().openCompositionTab(compId, { skipAnimation: true, playFromStart: true });
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

  // Click = play from start (re-click restarts)
  const handleSlotClick = useCallback((comp: Composition) => {
    openCompositionTab(comp.id, { skipAnimation: true, playFromStart: true });
  }, [openCompositionTab]);

  // Click empty slot = stop playback and deselect
  const handleEmptySlotClick = useCallback(() => {
    useTimelineStore.getState().stop();
    useMediaStore.getState().setActiveComposition(null);
  }, []);

  // Preview strip click
  const handlePreviewClick = useCallback((e: React.MouseEvent, comp: Composition) => {
    e.stopPropagation();
    if (previewCompositionId === comp.id) {
      setPreviewComposition(null);
    } else {
      setPreviewComposition(comp.id);
    }
  }, [previewCompositionId, setPreviewComposition]);

  // Drag handlers — track comp ID, not slot index
  const handleDragStart = useCallback((e: React.DragEvent, comp: Composition) => {
    setDragCompId(comp.id);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', comp.id);
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, slotIndex: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverIndex(slotIndex);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOverIndex(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, toSlotIndex: number) => {
    e.preventDefault();
    // Read comp ID from dataTransfer (reliable across browsers, not affected by state timing)
    const compId = e.dataTransfer.getData('text/plain') || dragCompId;
    setDragCompId(null);
    setDragOverIndex(null);
    if (compId) {
      moveSlot(compId, toSlotIndex);
    }
  }, [dragCompId, moveSlot]);

  const handleDragEnd = useCallback(() => {
    setDragCompId(null);
    setDragOverIndex(null);
  }, []);

  // Build slot map from assignments (reacts to slotAssignments changes)
  const slotMap = useMemo(() => {
    return getSlotMap(TOTAL_SLOTS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getSlotMap, slotAssignments]);

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

        {/* Column headers */}
        {Array.from({ length: GRID_COLS }, (_, i) => (
          <div key={`col-${i}`} className="slot-grid-col-header">
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
              const isDragOver = slotIndex === dragOverIndex && dragCompId !== null;

              if (comp) {
                const isActive = comp.id === activeCompositionId;
                const isPreviewed = comp.id === previewCompositionId;
                const isSelf = comp.id === dragCompId;
                return (
                  <div
                    key={slotIndex}
                    className={`slot-grid-item${isActive ? ' active' : ''}${isPreviewed ? ' previewed' : ''}${isDragOver && !isSelf ? ' drag-over' : ''}`}
                    data-comp-id={comp.id}
                    onClick={() => handleSlotClick(comp)}
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
                      isActive={isActive}
                      width={SLOT_SIZE - 4}
                      height={SLOT_SIZE - 4}
                    />
                    {isActive && <LivePlayhead duration={comp.duration} slotSize={SLOT_SIZE - 4} />}
                    <div
                      className={`slot-grid-preview-strip${isPreviewed ? ' active' : ''}`}
                      onClick={(e) => handlePreviewClick(e, comp)}
                      title="Preview this composition"
                    >
                      PRV
                    </div>
                  </div>
                );
              }

              return (
                <div
                  key={slotIndex}
                  className={`slot-grid-item empty${isDragOver ? ' drag-over' : ''}`}
                  onClick={handleEmptySlotClick}
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
    </div>
  );
}

/** Live playhead indicator — reads from high-frequency playheadState via rAF */
const LivePlayhead = memo(function LivePlayhead({ duration, slotSize }: { duration: number; slotSize: number }) {
  const lineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const line = lineRef.current;
    if (!line || duration <= 0) return;

    let rafId: number;
    const padding = 3;
    const trackWidth = slotSize - padding * 2;

    const update = () => {
      const pos = playheadState.isUsingInternalPosition
        ? playheadState.position
        : useTimelineStore.getState().playheadPosition;
      const pct = Math.max(0, Math.min(1, pos / duration));
      line.style.left = `${padding + pct * trackWidth}px`;
      rafId = requestAnimationFrame(update);
    };

    rafId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(rafId);
  }, [duration, slotSize]);

  return (
    <div
      ref={lineRef}
      style={{
        position: 'absolute',
        top: 19,
        bottom: 3,
        width: 1.5,
        background: '#ff3b3b',
        pointerEvents: 'none',
        zIndex: 1,
      }}
    />
  );
});
