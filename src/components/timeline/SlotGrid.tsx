// SlotGrid - CSS Grid container showing composition slots with MiniTimeline canvases
// Appears behind the timeline as a Figma-like canvas — timeline zooms into/out of its slot

import { useCallback, useMemo, useRef, useEffect, useState } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { MiniTimeline } from './MiniTimeline';
import type { Composition } from '../../stores/mediaStore';

export interface SlotRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface SlotGridProps {
  progress: number;
  onActiveSlotRect?: (rect: SlotRect | null) => void;
}

const SLOT_MIN_SIZE = 120;
const SLOT_MAX_SIZE = 180;

export function SlotGrid({ progress, onActiveSlotRect }: SlotGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeSlotRef = useRef<HTMLDivElement>(null);
  const [slotSize, setSlotSize] = useState(140);

  const compositions = useMediaStore(state => state.compositions);
  const activeCompositionId = useMediaStore(state => state.activeCompositionId);
  const openCompositionTab = useMediaStore(state => state.openCompositionTab);
  const setSlotGridProgress = useTimelineStore(state => state.setSlotGridProgress);

  // Auto-calculate slot size based on container width
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const containerWidth = entry.contentRect.width;
        const cols = Math.max(1, Math.floor(containerWidth / SLOT_MIN_SIZE));
        const size = Math.min(SLOT_MAX_SIZE, Math.floor((containerWidth - (cols - 1) * 8 - 16) / cols));
        setSlotSize(Math.max(SLOT_MIN_SIZE, size));
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Handle Ctrl+Shift+Scroll on the SlotGrid itself (to allow scrolling back to timeline)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        const store = useTimelineStore.getState();
        const delta = e.deltaY > 0 ? 0.08 : -0.08;
        let newProgress = Math.max(0, Math.min(1, store.slotGridProgress + delta));
        if (newProgress < 0.05) newProgress = 0;
        if (newProgress > 0.95) newProgress = 1;
        store.setSlotGridProgress(newProgress);
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  // Report active slot position relative to container
  useEffect(() => {
    if (!onActiveSlotRect) return;
    const activeSlot = activeSlotRef.current;
    const container = containerRef.current;
    if (!activeSlot || !container) {
      onActiveSlotRect(null);
      return;
    }
    const slotRect = activeSlot.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    onActiveSlotRect({
      x: slotRect.left - containerRect.left,
      y: slotRect.top - containerRect.top,
      width: slotRect.width,
      height: slotRect.height,
    });
  }, [activeCompositionId, slotSize, compositions.length, onActiveSlotRect]);

  // Handle slot click: switch composition and animate zoom-in
  const handleSlotClick = useCallback((comp: Composition) => {
    openCompositionTab(comp.id);
    // Animate back to full timeline (zoom in)
    animateSlotGridProgress(setSlotGridProgress, 0);
  }, [openCompositionTab, setSlotGridProgress]);

  const sortedCompositions = useMemo(() => {
    return [...compositions].sort((a, b) => a.name.localeCompare(b.name));
  }, [compositions]);

  if (sortedCompositions.length === 0) {
    return (
      <div
        ref={containerRef}
        className="slot-grid-container"
      >
        <div className="slot-grid-empty">
          No compositions
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="slot-grid-container"
      style={{ opacity: Math.min(1, progress * 3) }}
    >
      <div
        className="slot-grid"
        style={{
          gridTemplateColumns: `repeat(auto-fill, ${slotSize}px)`,
        }}
      >
        {sortedCompositions.map((comp) => {
          const isActive = comp.id === activeCompositionId;
          return (
            <div
              key={comp.id}
              ref={isActive ? activeSlotRef : undefined}
              className={`slot-grid-item ${isActive ? 'active' : ''}`}
              onClick={() => handleSlotClick(comp)}
              title={comp.name}
            >
              <MiniTimeline
                timelineData={comp.timelineData}
                compositionName={comp.name}
                compositionDuration={comp.duration}
                isActive={isActive}
                width={slotSize - 4}
                height={slotSize - 4}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Smooth animation helper: animate slotGridProgress from current value to target
function animateSlotGridProgress(
  setter: (progress: number) => void,
  targetValue: number,
) {
  const start = useTimelineStore.getState().slotGridProgress;
  const duration = 300; // ms
  const startTime = performance.now();

  function tick(now: number) {
    const elapsed = now - startTime;
    const t = Math.min(1, elapsed / duration);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - t, 3);
    const value = start + (targetValue - start) * eased;
    setter(value);

    if (t < 1) {
      requestAnimationFrame(tick);
    } else {
      setter(targetValue);
    }
  }

  requestAnimationFrame(tick);
}
