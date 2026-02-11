// Timeline store - combines all slices into a single Zustand store

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import type { TimelineStore, TimelineUtils, TimelineClip, Keyframe } from './types';
import { DEFAULT_TRACKS } from './constants';

import { createTrackSlice } from './trackSlice';
import { createClipSlice } from './clipSlice';
import { createTextClipSlice } from './textClipSlice';
import { createSolidClipSlice } from './solidClipSlice';
import { createClipEffectSlice } from './clipEffectSlice';
import { createLinkedGroupSlice } from './linkedGroupSlice';
import { createDownloadClipSlice } from './downloadClipSlice';
import { createPlaybackSlice } from './playbackSlice';
import { createRamPreviewSlice } from './ramPreviewSlice';
import { createProxyCacheSlice } from './proxyCacheSlice';
import { createSelectionSlice } from './selectionSlice';
import { createKeyframeSlice } from './keyframeSlice';
import { createMaskSlice } from './maskSlice';
import { createMarkerSlice } from './markerSlice';
import { createTransitionSlice } from './transitionSlice';
import { createClipboardSlice } from './clipboardSlice';
import { createPositioningUtils } from './positioningUtils';
import { createSerializationUtils } from './serializationUtils';
import { Logger } from '../../services/logger';

const log = Logger.create('Timeline');

// Re-export types for convenience
export type { TimelineStore, TimelineClip, Keyframe } from './types';
export { DEFAULT_TRANSFORM, DEFAULT_TRACKS, SNAP_THRESHOLD_SECONDS } from './constants';
export { seekVideo, getDefaultEffectParams } from './utils';

// Re-export selectors for optimized store subscriptions
export * from './selectors';

export const useTimelineStore = create<TimelineStore>()(
  subscribeWithSelector((set, get) => {
    // Create all slices
    const trackActions = createTrackSlice(set, get);
    const clipActions = createClipSlice(set, get);
    const textClipActions = createTextClipSlice(set, get);
    const solidClipActions = createSolidClipSlice(set, get);
    const clipEffectActions = createClipEffectSlice(set, get);
    const linkedGroupActions = createLinkedGroupSlice(set, get);
    const downloadClipActions = createDownloadClipSlice(set, get);
    const playbackActions = createPlaybackSlice(set, get);
    const ramPreviewActions = createRamPreviewSlice(set, get);
    const proxyCacheActions = createProxyCacheSlice(set, get);
    const selectionActions = createSelectionSlice(set, get);
    const keyframeActions = createKeyframeSlice(set, get);
    const maskActions = createMaskSlice(set, get);
    const markerActions = createMarkerSlice(set, get);
    const transitionActions = createTransitionSlice(set, get);
    const clipboardActions = createClipboardSlice(set, get);

    // Extracted utils (positioning + serialization)
    const positioningUtils = createPositioningUtils(set, get);
    const serializationUtils = createSerializationUtils(set, get);

    // Small utils that stay inline
    const inlineUtils: Pick<TimelineUtils, 'getClipsAtTime' | 'updateDuration' | 'findAvailableAudioTrack'> = {
      getClipsAtTime: (time) => {
        const { clips } = get();
        return clips.filter(c => time >= c.startTime && time < c.startTime + c.duration);
      },

      updateDuration: () => {
        const { clips, durationLocked } = get();
        // Don't auto-update if duration is manually locked
        if (durationLocked) return;

        if (clips.length === 0) {
          set({ duration: 60 });
          return;
        }
        const maxEnd = Math.max(...clips.map(c => c.startTime + c.duration));
        set({ duration: Math.max(60, maxEnd + 10) }); // Add 10 seconds padding
      },

      findAvailableAudioTrack: (startTime: number, duration: number) => {
        const { tracks, clips, addTrack } = get();
        const audioTracks = tracks.filter(t => t.type === 'audio');
        const endTime = startTime + duration;

        // Check each audio track for availability
        for (const track of audioTracks) {
          const trackClips = clips.filter(c => c.trackId === track.id);
          const hasOverlap = trackClips.some(clip => {
            const clipEnd = clip.startTime + clip.duration;
            // Check if time ranges overlap
            return !(endTime <= clip.startTime || startTime >= clipEnd);
          });

          if (!hasOverlap) {
            return track.id; // This track is available
          }
        }

        // No available audio track found, create a new one
        addTrack('audio');
        const { tracks: updatedTracks } = get();
        const newTrack = updatedTracks[updatedTracks.length - 1];
        log.debug('Created new audio track', { name: newTrack.name });
        return newTrack.id;
      },
    };

    // Combine all utils
    const utils: TimelineUtils = {
      ...inlineUtils,
      ...positioningUtils,
      ...serializationUtils,
    };

    // Initial state
    const initialState = {
      // Core state
      tracks: DEFAULT_TRACKS,
      clips: [] as TimelineClip[],
      playheadPosition: 0,
      duration: 60,
      zoom: 50,
      scrollX: 0,
      snappingEnabled: true,
      isPlaying: false,
      isDraggingPlayhead: false,
      selectedClipIds: new Set<string>(),
      primarySelectedClipId: null,

      // Render layers (populated by useLayerSync, used by engine)
      layers: [] as import('../../types').Layer[],
      selectedLayerId: null as string | null,

      // In/Out markers
      inPoint: null as number | null,
      outPoint: null as number | null,
      loopPlayback: false,

      // Playback speed (1 = normal, 2 = 2x, -1 = reverse, etc.)
      playbackSpeed: 1,

      // Duration lock (when true, duration won't auto-update based on clips)
      durationLocked: false,

      // RAM Preview state
      ramPreviewEnabled: false,
      ramPreviewProgress: null as number | null,
      ramPreviewRange: null as { start: number; end: number } | null,
      isRamPreviewing: false,
      cachedFrameTimes: new Set<number>(),

      // Proxy cache preloading state
      isProxyCaching: false,
      proxyCacheProgress: null as number | null,

      // Export progress state
      isExporting: false,
      exportProgress: null as number | null,
      exportCurrentTime: null as number | null,
      exportRange: null as { start: number; end: number } | null,

      // Performance toggles (enabled by default)
      thumbnailsEnabled: true,
      waveformsEnabled: true,
      showTranscriptMarkers: true,

      // Keyframe animation state
      clipKeyframes: new Map<string, Keyframe[]>(),
      keyframeRecordingEnabled: new Set<string>(),
      expandedTracks: new Set<string>(DEFAULT_TRACKS.map(t => t.id)),
      expandedTrackPropertyGroups: new Map<string, Set<string>>(),
      selectedKeyframeIds: new Set<string>(),
      expandedCurveProperties: new Map<string, Set<import('../../types').AnimatableProperty>>(),
      curveEditorHeight: 250,

      // Mask state
      maskEditMode: 'none' as const,
      activeMaskId: null as string | null,
      selectedVertexIds: new Set<string>(),
      maskDrawStart: null as { x: number; y: number } | null,
      maskDragging: false,

      // Tool mode
      toolMode: 'select' as const,

      // Timeline markers
      markers: [] as import('./types').TimelineMarker[],

      // Clip entrance animation key (increments on composition switch)
      clipEntranceAnimationKey: 0,

      // Clip animation phase for enter/exit transitions
      clipAnimationPhase: 'idle' as const,

      // Slot grid view progress (0 = full timeline, 1 = full grid view)
      slotGridProgress: 0,

      // Clipboard state for copy/paste
      clipboardData: null as import('./types').ClipboardClipData[] | null,
      clipboardKeyframes: null as import('./types').ClipboardKeyframeData[] | null,
    };

    // Layer actions (render layers for engine, moved from mixerStore)
    const layerActions = {
      setLayers: (layers: import('../../types').Layer[]) => {
        set({ layers });
      },
      updateLayer: (id: string, updates: Partial<import('../../types').Layer>) => {
        const { layers } = get();
        set({
          layers: layers.map((l) => (l?.id === id ? { ...l, ...updates } : l)),
        });
      },
      selectLayer: (id: string | null) => {
        set({ selectedLayerId: id });
      },
    };

    // Export actions (inline since they're simple)
    const exportActions = {
      setExportProgress: (progress: number | null, currentTime: number | null) => {
        set({ exportProgress: progress, exportCurrentTime: currentTime });
      },
      startExport: (start: number, end: number) => {
        set({ isExporting: true, exportProgress: 0, exportCurrentTime: start, exportRange: { start, end } });
      },
      endExport: () => {
        set({ isExporting: false, exportProgress: null, exportCurrentTime: null, exportRange: null });
      },
    };

    return {
      ...initialState,
      ...trackActions,
      ...clipActions,
      ...textClipActions,
      ...solidClipActions,
      ...clipEffectActions,
      ...linkedGroupActions,
      ...downloadClipActions,
      ...playbackActions,
      ...ramPreviewActions,
      ...proxyCacheActions,
      ...exportActions,
      ...selectionActions,
      ...keyframeActions,
      ...layerActions,
      ...maskActions,
      ...markerActions,
      ...transitionActions,
      ...clipboardActions,
      ...utils,
    };
  })
);
