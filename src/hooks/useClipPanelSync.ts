// Hook to auto-switch panels based on selected clip
// Activates Properties panel when any clip is selected

import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../stores/timeline';
import { useDockStore } from '../stores/dockStore';

export function useClipPanelSync() {
  const clips = useTimelineStore(state => state.clips);
  const selectedClipIds = useTimelineStore(state => state.selectedClipIds);
  const activatePanelType = useDockStore(state => state.activatePanelType);

  // Track previous selection to only activate on new selections
  const prevSelectedId = useRef<string | null>(null);

  useEffect(() => {
    // Get first selected clip
    const selectedId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;

    // Only react to new selections (not deselections or same selection)
    if (!selectedId || selectedId === prevSelectedId.current) {
      prevSelectedId.current = selectedId;
      return;
    }

    prevSelectedId.current = selectedId;

    // Find the selected clip
    const selectedClip = clips.find(c => c.id === selectedId);
    if (!selectedClip) return;

    // Activate Properties panel (handles both audio and video clips with appropriate tabs)
    activatePanelType('clip-properties');
  }, [selectedClipIds, clips, activatePanelType]);
}
