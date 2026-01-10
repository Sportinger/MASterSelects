// Track-related actions slice

import type { TimelineTrack } from '../../types';
import type { TrackActions, SliceCreator } from './types';

export const createTrackSlice: SliceCreator<TrackActions> = (set, get) => ({
  addTrack: (type) => {
    const { tracks, expandedTracks } = get();
    const typeCount = tracks.filter(t => t.type === type).length + 1;
    const newTrack: TimelineTrack = {
      id: `${type}-${Date.now()}`,
      name: `${type === 'video' ? 'Video' : 'Audio'} ${typeCount}`,
      type,
      height: type === 'video' ? 60 : 40,
      muted: false,
      visible: true,
      solo: false,
    };

    // Video tracks: insert at TOP (before all existing video tracks)
    // Audio tracks: insert at BOTTOM (after all existing audio tracks)
    if (type === 'video') {
      // Insert at index 0 (top of timeline) and auto-expand
      const newExpanded = new Set(expandedTracks);
      newExpanded.add(newTrack.id);
      set({ tracks: [newTrack, ...tracks], expandedTracks: newExpanded });
    } else {
      // Audio: append at end (bottom of timeline)
      set({ tracks: [...tracks, newTrack] });
    }

    return newTrack.id;
  },

  removeTrack: (id) => {
    const { tracks, clips } = get();
    set({
      tracks: tracks.filter(t => t.id !== id),
      clips: clips.filter(c => c.trackId !== id),
    });
  },

  renameTrack: (id, name) => {
    const { tracks } = get();
    set({
      tracks: tracks.map(t => t.id === id ? { ...t, name } : t),
    });
  },

  setTrackMuted: (id, muted) => {
    const { tracks } = get();
    set({
      tracks: tracks.map(t => t.id === id ? { ...t, muted } : t),
    });
    // Audio changes don't affect video cache
  },

  setTrackVisible: (id, visible) => {
    const { tracks, invalidateCache } = get();
    const track = tracks.find(t => t.id === id);
    set({
      tracks: tracks.map(t => t.id === id ? { ...t, visible } : t),
    });
    // Invalidate cache if video track visibility changed
    if (track?.type === 'video') {
      invalidateCache();
    }
  },

  setTrackSolo: (id, solo) => {
    const { tracks, invalidateCache } = get();
    const track = tracks.find(t => t.id === id);
    set({
      tracks: tracks.map(t => t.id === id ? { ...t, solo } : t),
    });
    // Invalidate cache if video track solo changed
    if (track?.type === 'video') {
      invalidateCache();
    }
  },

  setTrackHeight: (id, height) => {
    const { tracks } = get();
    set({
      tracks: tracks.map(t => t.id === id ? { ...t, height: Math.max(30, Math.min(200, height)) } : t),
    });
  },

  scaleTracksOfType: (type, delta) => {
    const { tracks } = get();
    const tracksOfType = tracks.filter(t => t.type === type);

    if (tracksOfType.length === 0) return;

    // Find the max height among tracks of this type
    const maxHeight = Math.max(...tracksOfType.map(t => t.height));

    // First call: sync all to max height (if they differ)
    // Subsequent calls: scale uniformly
    const allSameHeight = tracksOfType.every(t => t.height === maxHeight);

    if (!allSameHeight && delta !== 0) {
      // Sync all to max height first
      set({
        tracks: tracks.map(t =>
          t.type === type ? { ...t, height: maxHeight } : t
        ),
      });
    } else {
      // All already synced, scale uniformly
      const newHeight = Math.max(30, Math.min(200, maxHeight + delta));
      set({
        tracks: tracks.map(t =>
          t.type === type ? { ...t, height: newHeight } : t
        ),
      });
    }
  },

  // Track parenting (layer linking) - like After Effects layer parenting
  setTrackParent: (trackId, parentTrackId) => {
    const { tracks } = get();

    // Can't parent to self
    if (parentTrackId === trackId) return;

    // Cycle detection: parent can't be a child/grandchild of this track
    if (parentTrackId) {
      const wouldCreateCycle = (checkId: string): boolean => {
        const check = tracks.find(t => t.id === checkId);
        if (!check?.parentTrackId) return false;
        if (check.parentTrackId === trackId) return true;
        return wouldCreateCycle(check.parentTrackId);
      };

      if (wouldCreateCycle(parentTrackId)) {
        console.warn('Cannot create circular track parent reference');
        return;
      }
    }

    set({
      tracks: tracks.map(t =>
        t.id === trackId ? { ...t, parentTrackId: parentTrackId || undefined } : t
      ),
    });
  },

  getTrackChildren: (trackId) => {
    const { tracks } = get();
    return tracks.filter(t => t.parentTrackId === trackId);
  },
});
