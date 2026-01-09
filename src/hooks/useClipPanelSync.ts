// Hook to auto-switch panels based on selected clip type
// - Audio clip selected -> Audio panel activated
// - Video clip selected -> Properties panel activated

import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../stores/timeline';
import { useDockStore } from '../stores/dockStore';

export function useClipPanelSync() {
  const clips = useTimelineStore(state => state.clips);
  const tracks = useTimelineStore(state => state.tracks);
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

    // Find the selected clip and its track
    const selectedClip = clips.find(c => c.id === selectedId);
    if (!selectedClip) return;

    const track = tracks.find(t => t.id === selectedClip.trackId);
    if (!track) return;

    // Activate the appropriate panel based on track type
    if (track.type === 'audio') {
      activatePanelType('audio');
    } else if (track.type === 'video') {
      activatePanelType('clip-properties');
    }
  }, [selectedClipIds, clips, tracks, activatePanelType]);
}
