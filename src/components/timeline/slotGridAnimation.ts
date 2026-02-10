// Shared slot grid animation — triggers a 250ms ease-out transition
// Used by both useTimelineZoom (Ctrl+Shift+Scroll) and SlotGrid (click)

import { useTimelineStore } from '../../stores/timeline';

let _animId = 0;

export function animateSlotGrid(target: 0 | 1) {
  const id = ++_animId;
  const start = useTimelineStore.getState().slotGridProgress;
  // Already at target
  if ((target === 1 && start > 0.95) || (target === 0 && start < 0.05)) return;

  const duration = 250;
  const startTime = performance.now();

  function tick(now: number) {
    if (id !== _animId) return; // Cancelled by newer animation
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    const value = start + (target - start) * eased;
    useTimelineStore.getState().setSlotGridProgress(value);
    if (t < 1) {
      requestAnimationFrame(tick);
    } else {
      useTimelineStore.getState().setSlotGridProgress(target);
    }
  }

  requestAnimationFrame(tick);
}
