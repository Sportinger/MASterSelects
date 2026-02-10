// SlotGrid - CSS Grid container showing composition slots with MiniTimeline canvases
// Resolume-style grid view that appears when Ctrl+Shift+Scrolling the timeline

import { useCallback, useMemo, useRef, useEffect, useState } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { MiniTimeline } from './MiniTimeline';
import type { Composition } from '../../stores/mediaStore';

interface SlotGridProps {
  opacity: number;
}

const SLOT_MIN_SIZE = 120;
const SLOT_MAX_SIZE = 180;

export function SlotGrid({ opacity }: SlotGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
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
        // Calculate columns that fit, aiming for slot sizes between min and max
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
        const delta = e.deltaY > 0 ? 0.04 : -0.04;
        let newProgress = Math.max(0, Math.min(1, store.slotGridProgress + delta));
        if (newProgress < 0.05) newProgress = 0;
        if (newProgress > 0.95) newProgress = 1;
        store.setSlotGridProgress(newProgress);
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  // Handle slot click: switch composition and animate back to timeline
  const handleSlotClick = useCallback((comp: Composition) => {
    openCompositionTab(comp.id);
    // Animate back to full timeline
    animateSlotGridProgress(setSlotGridProgress, 0);
  }, [openCompositionTab, setSlotGridProgress]);

  // Filter to show only compositions (not other media items)
  const sortedCompositions = useMemo(() => {
    return [...compositions].sort((a, b) => a.name.localeCompare(b.name));
  }, [compositions]);

  if (sortedCompositions.length === 0) {
    return (
      <div
        ref={containerRef}
        className="slot-grid-container"
        style={{ opacity }}
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
      style={{ opacity }}
    >
      <div
        className="slot-grid"
        style={{
          gridTemplateColumns: `repeat(auto-fill, ${slotSize}px)`,
        }}
      >
        {sortedCompositions.map((comp) => (
          <div
            key={comp.id}
            className={`slot-grid-item ${comp.id === activeCompositionId ? 'active' : ''}`}
            onClick={() => handleSlotClick(comp)}
            title={comp.name}
          >
            <MiniTimeline
              timelineData={comp.timelineData}
              compositionName={comp.name}
              compositionDuration={comp.duration}
              isActive={comp.id === activeCompositionId}
              width={slotSize - 4}
              height={slotSize - 4}
            />
          </div>
        ))}
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
