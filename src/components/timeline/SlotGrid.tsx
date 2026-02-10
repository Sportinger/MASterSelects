// SlotGrid - CSS Grid container showing composition slots with MiniTimeline canvases
// Fades in over the timeline when Ctrl+Shift+Scroll triggers the transition

import { useCallback, useMemo, useRef, useEffect, useState } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { animateSlotGrid } from './slotGridAnimation';
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
        animateSlotGrid(e.deltaY > 0 ? 1 : 0);
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  // Handle slot click: switch composition and animate zoom-in
  const handleSlotClick = useCallback((comp: Composition) => {
    openCompositionTab(comp.id);
    animateSlotGrid(0);
  }, [openCompositionTab]);

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
        {sortedCompositions.map((comp) => {
          const isActive = comp.id === activeCompositionId;
          return (
            <div
              key={comp.id}
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
