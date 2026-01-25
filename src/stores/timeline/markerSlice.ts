// Marker-related actions slice

import type { MarkerActions, SliceCreator, TimelineMarker } from './types';

// Default marker color
const DEFAULT_MARKER_COLOR = '#00d4ff';

// Generate unique marker ID
const generateMarkerId = () => `marker-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

export const createMarkerSlice: SliceCreator<MarkerActions> = (set, get) => ({
  addMarker: (time: number, label?: string, color?: string) => {
    const id = generateMarkerId();
    const { duration } = get();

    // Clamp time to valid range
    const clampedTime = Math.max(0, Math.min(time, duration));

    const marker: TimelineMarker = {
      id,
      time: clampedTime,
      label: label || '',
      color: color || DEFAULT_MARKER_COLOR,
    };

    set(state => ({
      markers: [...state.markers, marker].sort((a, b) => a.time - b.time),
    }));

    return id;
  },

  removeMarker: (markerId: string) => {
    set(state => ({
      markers: state.markers.filter(m => m.id !== markerId),
    }));
  },

  updateMarker: (markerId: string, updates: Partial<Omit<TimelineMarker, 'id'>>) => {
    const { duration } = get();

    set(state => ({
      markers: state.markers.map(m => {
        if (m.id !== markerId) return m;

        const updatedMarker = { ...m, ...updates };

        // Clamp time if it was updated
        if (updates.time !== undefined) {
          updatedMarker.time = Math.max(0, Math.min(updates.time, duration));
        }

        return updatedMarker;
      }).sort((a, b) => a.time - b.time),
    }));
  },

  moveMarker: (markerId: string, newTime: number) => {
    const { duration } = get();
    const clampedTime = Math.max(0, Math.min(newTime, duration));

    set(state => ({
      markers: state.markers.map(m =>
        m.id === markerId ? { ...m, time: clampedTime } : m
      ).sort((a, b) => a.time - b.time),
    }));
  },

  clearMarkers: () => {
    set({ markers: [] });
  },
});
