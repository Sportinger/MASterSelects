// Timeline store for video editing functionality

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { TimelineClip, TimelineTrack, ClipTransform, BlendMode } from '../types';
import { useMediaStore } from './mediaStore';

// Default transform for new clips
const DEFAULT_TRANSFORM: ClipTransform = {
  opacity: 1,
  blendMode: 'normal',
  position: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1 },
  rotation: { x: 0, y: 0, z: 0 },
};

// Generate waveform data from audio file
async function generateWaveform(file: File, sampleCount: number = 200): Promise<number[]> {
  try {
    const audioContext = new AudioContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0); // Use first channel
    const samples: number[] = [];
    const blockSize = Math.floor(channelData.length / sampleCount);

    for (let i = 0; i < sampleCount; i++) {
      const start = i * blockSize;
      const end = start + blockSize;
      let sum = 0;

      for (let j = start; j < end; j++) {
        sum += Math.abs(channelData[j]);
      }

      samples.push(sum / blockSize);
    }

    // Normalize to 0-1 range
    const max = Math.max(...samples);
    if (max > 0) {
      await audioContext.close();
      return samples.map(s => s / max);
    }
    await audioContext.close();
    return samples;
  } catch (e) {
    console.warn('Failed to generate waveform:', e);
    return [];
  }
}

// Generate thumbnail filmstrip from video
async function generateThumbnails(video: HTMLVideoElement, duration: number, count: number = 10): Promise<string[]> {
  const thumbnails: string[] = [];
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return thumbnails;

  // Thumbnail dimensions (aspect ratio preserved)
  const thumbHeight = 40;
  const thumbWidth = Math.round((video.videoWidth / video.videoHeight) * thumbHeight);
  canvas.width = thumbWidth;
  canvas.height = thumbHeight;

  // Generate frames at regular intervals
  const interval = duration / count;

  for (let i = 0; i < count; i++) {
    const time = i * interval;
    try {
      await seekVideo(video, time);
      ctx.drawImage(video, 0, 0, thumbWidth, thumbHeight);
      thumbnails.push(canvas.toDataURL('image/jpeg', 0.6));
    } catch (e) {
      console.warn('Failed to generate thumbnail at', time, e);
    }
  }

  return thumbnails;
}

// Helper to seek video and wait for it to be ready
function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Seek timeout')), 3000);

    const onSeeked = () => {
      clearTimeout(timeout);
      video.removeEventListener('seeked', onSeeked);
      resolve();
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}

// Snap threshold in seconds (clips will snap when within this distance)
const SNAP_THRESHOLD_SECONDS = 0.1;

interface TimelineStore {
  // State
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  playheadPosition: number;
  duration: number;
  zoom: number;  // pixels per second
  scrollX: number;
  isPlaying: boolean;
  isDraggingPlayhead: boolean;  // Track when user is scrubbing
  selectedClipId: string | null;

  // In/Out markers for work area
  inPoint: number | null;
  outPoint: number | null;

  // Loop playback between in/out points
  loopPlayback: boolean;
  setLoopPlayback: (loop: boolean) => void;
  toggleLoopPlayback: () => void;

  // RAM Preview - pre-render frames to memory for instant scrubbing
  ramPreviewEnabled: boolean;  // Master toggle for RAM Preview feature
  ramPreviewProgress: number | null;  // null = not previewing, 0-100 = progress
  ramPreviewRange: { start: number; end: number } | null;  // cached time range
  isRamPreviewing: boolean;
  toggleRamPreviewEnabled: () => void;
  startRamPreview: () => Promise<void>;
  cancelRamPreview: () => void;
  clearRamPreview: () => void;

  // Playback frame caching (like After Effects' green line)
  cachedFrameTimes: Set<number>;  // Set of quantized times that are cached
  addCachedFrame: (time: number) => void;
  getCachedRanges: () => Array<{ start: number; end: number }>;
  invalidateCache: () => void;  // Clear cache when content changes

  // Track actions
  addTrack: (type: 'video' | 'audio') => void;
  removeTrack: (id: string) => void;
  setTrackMuted: (id: string, muted: boolean) => void;
  setTrackVisible: (id: string, visible: boolean) => void;
  setTrackSolo: (id: string, solo: boolean) => void;
  setTrackHeight: (id: string, height: number) => void;
  scaleTracksOfType: (type: 'video' | 'audio', delta: number) => void;



  // Clip actions
  addClip: (trackId: string, file: File, startTime: number, estimatedDuration?: number) => Promise<void>;
  removeClip: (id: string) => void;
  moveClip: (id: string, newStartTime: number, newTrackId?: string, skipLinked?: boolean) => void;
  trimClip: (id: string, inPoint: number, outPoint: number) => void;
  selectClip: (id: string | null) => void;
  updateClipTransform: (id: string, transform: Partial<ClipTransform>) => void;

  // Playback actions
  setPlayheadPosition: (position: number) => void;
  setDraggingPlayhead: (dragging: boolean) => void;
  play: () => void;
  pause: () => void;
  stop: () => void;

  // View actions
  setZoom: (zoom: number) => void;
  setScrollX: (scrollX: number) => void;

  // In/Out marker actions
  setInPoint: (time: number | null) => void;
  setOutPoint: (time: number | null) => void;
  clearInOut: () => void;
  setInPointAtPlayhead: () => void;
  setOutPointAtPlayhead: () => void;

  // Utils
  getClipsAtTime: (time: number) => TimelineClip[];
  updateDuration: () => void;
  findAvailableAudioTrack: (startTime: number, duration: number) => string;
  getSnappedPosition: (clipId: string, desiredStartTime: number, trackId: string) => { startTime: number; snapped: boolean };
  findNonOverlappingPosition: (clipId: string, desiredStartTime: number, trackId: string, duration: number) => number;
}

const DEFAULT_TRACKS: TimelineTrack[] = [
  { id: 'video-1', name: 'Video 1', type: 'video', height: 60, muted: false, visible: true, solo: false },
  { id: 'video-2', name: 'Video 2', type: 'video', height: 60, muted: false, visible: true, solo: false },
  { id: 'audio-1', name: 'Audio', type: 'audio', height: 40, muted: false, visible: true, solo: false },
];

export const useTimelineStore = create<TimelineStore>()(
  subscribeWithSelector((set, get) => ({
    tracks: DEFAULT_TRACKS,
    clips: [],
    playheadPosition: 0,
    duration: 60, // Default 60 seconds
    zoom: 50, // 50 pixels per second
    scrollX: 0,
    isPlaying: false,
    isDraggingPlayhead: false,
    selectedClipId: null,
    inPoint: null,
    outPoint: null,
    loopPlayback: false,
    ramPreviewEnabled: false,  // RAM Preview off by default
    ramPreviewProgress: null,
    ramPreviewRange: null,
    isRamPreviewing: false,
    cachedFrameTimes: new Set<number>(),

    // Track actions
    addTrack: (type) => {
      const { tracks } = get();
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
        // Insert at index 0 (top of timeline)
        set({ tracks: [newTrack, ...tracks] });
      } else {
        // Audio: append at end (bottom of timeline)
        set({ tracks: [...tracks, newTrack] });
      }
    },

    removeTrack: (id) => {
      const { tracks, clips } = get();
      set({
        tracks: tracks.filter(t => t.id !== id),
        clips: clips.filter(c => c.trackId !== id),
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
      const { tracks } = get();
      const track = tracks.find(t => t.id === id);
      set({
        tracks: tracks.map(t => t.id === id ? { ...t, visible } : t),
      });
      // Invalidate cache if video track visibility changed
      if (track?.type === 'video') {
        get().invalidateCache();
      }
    },

    setTrackSolo: (id, solo) => {
      const { tracks } = get();
      const track = tracks.find(t => t.id === id);
      set({
        tracks: tracks.map(t => t.id === id ? { ...t, solo } : t),
      });
      // Invalidate cache if video track solo changed
      if (track?.type === 'video') {
        get().invalidateCache();
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

    // Clip actions
    addClip: async (trackId, file, startTime, providedDuration) => {
      const isVideo = file.type.startsWith('video/');
      const isAudio = file.type.startsWith('audio/');
      const isImage = file.type.startsWith('image/');

      // Validate track type matches media type
      const { tracks } = get();
      const targetTrack = tracks.find(t => t.id === trackId);
      if (!targetTrack) {
        console.warn('[Timeline] Track not found:', trackId);
        return;
      }

      // Video/image files can only go on video tracks
      if ((isVideo || isImage) && targetTrack.type !== 'video') {
        console.warn('[Timeline] Cannot add video/image to audio track');
        return;
      }

      // Audio files can only go on audio tracks
      if (isAudio && targetTrack.type !== 'audio') {
        console.warn('[Timeline] Cannot add audio to video track');
        return;
      }

      const clipId = `clip-${Date.now()}`;
      const audioClipId = isVideo ? `clip-audio-${Date.now()}` : undefined;

      // Use provided duration or estimate (will be updated when media loads)
      const estimatedDuration = providedDuration ?? 5;

      // Helper to update clip when loaded
      const updateClip = (id: string, updates: Partial<TimelineClip>) => {
        const currentClips = get().clips;
        set({
          clips: currentClips.map(c => c.id === id ? { ...c, ...updates } : c)
        });
        get().updateDuration();
      };

      // For video: add loading placeholder for both video and audio clips immediately
      if (isVideo) {
        const { findAvailableAudioTrack, clips: currentClips, updateDuration } = get();
        const audioTrackId = findAvailableAudioTrack(startTime, estimatedDuration);

        // Create loading placeholder clips immediately
        const videoClip: TimelineClip = {
          id: clipId,
          trackId,
          name: file.name,
          file,
          startTime,
          duration: estimatedDuration,
          inPoint: 0,
          outPoint: estimatedDuration,
          source: { type: 'video', naturalDuration: estimatedDuration },
          linkedClipId: audioTrackId ? audioClipId : undefined,
          transform: { ...DEFAULT_TRANSFORM },
          isLoading: true,
        };

        const clipsToAdd: TimelineClip[] = [videoClip];

        if (audioTrackId && audioClipId) {
          const audioClip: TimelineClip = {
            id: audioClipId,
            trackId: audioTrackId,
            name: `${file.name} (Audio)`,
            file,
            startTime,
            duration: estimatedDuration,
            inPoint: 0,
            outPoint: estimatedDuration,
            source: { type: 'audio', naturalDuration: estimatedDuration },
            linkedClipId: clipId,
            transform: { ...DEFAULT_TRANSFORM },
            isLoading: true,
          };
          clipsToAdd.push(audioClip);
        }

        set({ clips: [...currentClips, ...clipsToAdd] });
        updateDuration();

        // Now load media in background
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.preload = 'auto';
        video.muted = true;
        video.crossOrigin = 'anonymous';

        // Wait for metadata
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
          video.onerror = () => resolve();
        });

        const naturalDuration = video.duration || estimatedDuration;

        // Update clip duration immediately once we know it
        updateClip(clipId, {
          duration: naturalDuration,
          outPoint: naturalDuration,
          source: { type: 'video', videoElement: video, naturalDuration },
        });
        if (audioTrackId && audioClipId) {
          updateClip(audioClipId, {
            duration: naturalDuration,
            outPoint: naturalDuration,
          });
        }

        // Wait for video to be ready for thumbnails
        await new Promise<void>((resolve) => {
          if (video.readyState >= 2) {
            resolve();
          } else {
            video.oncanplay = () => resolve();
          }
        });

        // Generate thumbnails
        let thumbnails: string[] = [];
        try {
          thumbnails = await generateThumbnails(video, naturalDuration);
          console.log(`[Timeline] Generated ${thumbnails.length} thumbnails for ${file.name}`);
        } catch (e) {
          console.warn('Failed to generate thumbnails:', e);
        }

        video.currentTime = 0;

        // Final update with thumbnails and isLoading=false
        updateClip(clipId, {
          source: { type: 'video', videoElement: video, naturalDuration },
          thumbnails,
          isLoading: false,
        });

        // Load audio
        if (audioTrackId && audioClipId) {
          const audioFromVideo = document.createElement('audio');
          audioFromVideo.src = URL.createObjectURL(file);
          audioFromVideo.preload = 'auto';

          let audioWaveform: number[] = [];
          try {
            audioWaveform = await generateWaveform(file);
          } catch (e) {
            console.warn('Failed to generate waveform:', e);
          }

          updateClip(audioClipId, {
            source: { type: 'audio', audioElement: audioFromVideo, naturalDuration },
            waveform: audioWaveform,
            isLoading: false,
          });
        }

        // Sync to media store
        const mediaStore = useMediaStore.getState();
        if (!mediaStore.getFileByName(file.name)) {
          mediaStore.importFile(file);
        }
        // Invalidate RAM preview cache - new video content added
        get().invalidateCache();

        return;
      }

      // For audio: add loading placeholder immediately
      if (isAudio) {
        const { clips: currentClips, updateDuration } = get();

        const audioClip: TimelineClip = {
          id: clipId,
          trackId,
          name: file.name,
          file,
          startTime,
          duration: estimatedDuration,
          inPoint: 0,
          outPoint: estimatedDuration,
          source: { type: 'audio', naturalDuration: estimatedDuration },
          transform: { ...DEFAULT_TRANSFORM },
          isLoading: true,
        };

        set({ clips: [...currentClips, audioClip] });
        updateDuration();

        // Load audio in background
        const audio = document.createElement('audio');
        audio.src = URL.createObjectURL(file);
        audio.preload = 'metadata';

        await new Promise<void>((resolve) => {
          audio.onloadedmetadata = () => resolve();
          audio.onerror = () => resolve();
        });

        const naturalDuration = audio.duration || estimatedDuration;

        // Generate waveform
        let waveform: number[] = [];
        try {
          waveform = await generateWaveform(file);
        } catch (e) {
          console.warn('Failed to generate waveform:', e);
        }

        updateClip(clipId, {
          duration: naturalDuration,
          outPoint: naturalDuration,
          source: { type: 'audio', audioElement: audio, naturalDuration },
          waveform,
          isLoading: false,
        });

        // Sync to media store
        const mediaStore = useMediaStore.getState();
        if (!mediaStore.getFileByName(file.name)) {
          mediaStore.importFile(file);
        }
        // Invalidate RAM preview cache - audio affects composition
        get().invalidateCache();

        return;
      }

      // For images: add loading placeholder immediately
      if (isImage) {
        const { clips: currentClips, updateDuration } = get();

        const imageClip: TimelineClip = {
          id: clipId,
          trackId,
          name: file.name,
          file,
          startTime,
          duration: estimatedDuration,
          inPoint: 0,
          outPoint: estimatedDuration,
          source: { type: 'image', naturalDuration: estimatedDuration },
          transform: { ...DEFAULT_TRANSFORM },
          isLoading: true,
        };

        set({ clips: [...currentClips, imageClip] });
        updateDuration();

        // Load image in background
        const img = new Image();
        img.src = URL.createObjectURL(file);

        await new Promise<void>((resolve) => {
          img.onload = () => resolve();
          img.onerror = () => resolve();
        });

        // Generate thumbnail
        let thumbnails: string[] = [];
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const thumbHeight = 40;
          const thumbWidth = Math.round((img.width / img.height) * thumbHeight);
          canvas.width = thumbWidth;
          canvas.height = thumbHeight;
          ctx.drawImage(img, 0, 0, thumbWidth, thumbHeight);
          thumbnails = [canvas.toDataURL('image/jpeg', 0.6)];
        }

        updateClip(clipId, {
          source: { type: 'image', imageElement: img, naturalDuration: estimatedDuration },
          thumbnails,
          isLoading: false,
        });

        // Sync to media store
        const mediaStore = useMediaStore.getState();
        if (!mediaStore.getFileByName(file.name)) {
          mediaStore.importFile(file);
        }
        // Invalidate RAM preview cache - new content added
        get().invalidateCache();
      }
    },

    removeClip: (id) => {
      const { clips, selectedClipId, updateDuration } = get();
      set({
        clips: clips.filter(c => c.id !== id),
        selectedClipId: selectedClipId === id ? null : selectedClipId,
      });
      updateDuration();
      // Invalidate RAM preview cache - content changed
      get().invalidateCache();
    },

    moveClip: (id, newStartTime, newTrackId, skipLinked = false) => {
      const { clips, tracks, updateDuration, getSnappedPosition, findNonOverlappingPosition } = get();
      const movingClip = clips.find(c => c.id === id);
      if (!movingClip) return;

      const targetTrackId = newTrackId ?? movingClip.trackId;

      // Validate track type if changing tracks
      if (newTrackId && newTrackId !== movingClip.trackId) {
        const targetTrack = tracks.find(t => t.id === newTrackId);
        const sourceType = movingClip.source?.type;

        if (targetTrack && sourceType) {
          // Video/image clips can only go on video tracks
          if ((sourceType === 'video' || sourceType === 'image') && targetTrack.type !== 'video') {
            console.warn('[Timeline] Cannot move video/image to audio track');
            return;
          }
          // Audio clips can only go on audio tracks
          if (sourceType === 'audio' && targetTrack.type !== 'audio') {
            console.warn('[Timeline] Cannot move audio to video track');
            return;
          }
        }
      }

      // Apply snapping first
      const { startTime: snappedTime } = getSnappedPosition(id, newStartTime, targetTrackId);

      // Then find non-overlapping position
      const finalStartTime = findNonOverlappingPosition(id, snappedTime, targetTrackId, movingClip.duration);

      // Calculate time delta to apply to linked clips
      const timeDelta = finalStartTime - movingClip.startTime;

      // For linked clip, also find non-overlapping position
      const linkedClip = clips.find(c => c.id === movingClip.linkedClipId || c.linkedClipId === id);
      let linkedFinalTime = linkedClip ? linkedClip.startTime + timeDelta : 0;
      if (linkedClip && !skipLinked) {
        linkedFinalTime = findNonOverlappingPosition(
          linkedClip.id,
          linkedClip.startTime + timeDelta,
          linkedClip.trackId,
          linkedClip.duration
        );
      }

      set({
        clips: clips.map(c => {
          // Move the primary clip
          if (c.id === id) {
            return {
              ...c,
              startTime: Math.max(0, finalStartTime),
              trackId: targetTrackId,
            };
          }
          // Also move linked clip (keep it in sync) - unless skipLinked is true
          if (!skipLinked && (c.id === movingClip.linkedClipId || c.linkedClipId === id)) {
            return {
              ...c,
              startTime: Math.max(0, linkedFinalTime),
              // Keep linked clip on its own track (don't change track)
            };
          }
          return c;
        }),
      });
      updateDuration();
      // Invalidate RAM preview cache - content changed
      get().invalidateCache();
    },

    trimClip: (id, inPoint, outPoint) => {
      const { clips, updateDuration } = get();
      set({
        clips: clips.map(c => {
          if (c.id !== id) return c;
          const newDuration = outPoint - inPoint;
          return {
            ...c,
            inPoint,
            outPoint,
            duration: newDuration,
          };
        }),
      });
      updateDuration();
      // Invalidate RAM preview cache - content changed
      get().invalidateCache();
    },

    selectClip: (id) => {
      set({ selectedClipId: id });
    },

    updateClipTransform: (id, transform) => {
      const { clips } = get();
      set({
        clips: clips.map(c => {
          if (c.id !== id) return c;
          return {
            ...c,
            transform: {
              ...c.transform,
              ...transform,
              position: transform.position
                ? { ...c.transform.position, ...transform.position }
                : c.transform.position,
              scale: transform.scale
                ? { ...c.transform.scale, ...transform.scale }
                : c.transform.scale,
              rotation: transform.rotation
                ? { ...c.transform.rotation, ...transform.rotation }
                : c.transform.rotation,
            },
          };
        }),
      });
      // Invalidate cache - transform affects rendered output
      get().invalidateCache();
    },

    // Playback actions
    setPlayheadPosition: (position) => {
      const { duration } = get();
      set({ playheadPosition: Math.max(0, Math.min(position, duration)) });
    },
    setDraggingPlayhead: (dragging) => {
      set({ isDraggingPlayhead: dragging });
    },

    play: () => {
      set({ isPlaying: true });
    },

    pause: () => {
      set({ isPlaying: false });
    },

    stop: () => {
      set({ isPlaying: false, playheadPosition: 0 });
    },

    // View actions
    setZoom: (zoom) => {
      set({ zoom: Math.max(10, Math.min(200, zoom)) });
    },

    setScrollX: (scrollX) => {
      set({ scrollX: Math.max(0, scrollX) });
    },

    // In/Out marker actions
    setInPoint: (time) => {
      const { outPoint, duration } = get();
      if (time === null) {
        set({ inPoint: null });
        return;
      }
      // Ensure in point doesn't exceed out point or duration
      const clampedTime = Math.max(0, Math.min(time, outPoint ?? duration));
      set({ inPoint: clampedTime });
    },

    setOutPoint: (time) => {
      const { inPoint, duration } = get();
      if (time === null) {
        set({ outPoint: null });
        return;
      }
      // Ensure out point doesn't precede in point and doesn't exceed duration
      const clampedTime = Math.max(inPoint ?? 0, Math.min(time, duration));
      set({ outPoint: clampedTime });
    },

    clearInOut: () => {
      set({ inPoint: null, outPoint: null });
    },

    setInPointAtPlayhead: () => {
      const { playheadPosition } = get();
      get().setInPoint(playheadPosition);
    },

    setOutPointAtPlayhead: () => {
      const { playheadPosition } = get();
      get().setOutPoint(playheadPosition);
    },

    setLoopPlayback: (loop) => {
      set({ loopPlayback: loop });
    },

    toggleLoopPlayback: () => {
      set({ loopPlayback: !get().loopPlayback });
    },

    // RAM Preview actions
    toggleRamPreviewEnabled: () => {
      const { ramPreviewEnabled } = get();
      if (ramPreviewEnabled) {
        // Turning OFF - cancel any running preview and clear cache
        set({ ramPreviewEnabled: false, isRamPreviewing: false, ramPreviewProgress: null });
        import('../engine/WebGPUEngine').then(({ engine }) => {
          engine.setGeneratingRamPreview(false);
          engine.clearCompositeCache();
        });
        set({ ramPreviewRange: null, cachedFrameTimes: new Set() });
      } else {
        // Turning ON - enable automatic RAM preview
        set({ ramPreviewEnabled: true });
      }
    },

    startRamPreview: async () => {
      const { inPoint, outPoint, duration, clips, tracks, isRamPreviewing, playheadPosition, addCachedFrame, ramPreviewEnabled } = get();
      // Don't start if RAM Preview is disabled or already running
      if (!ramPreviewEnabled || isRamPreviewing) return;

      // Determine range to preview (use In/Out or clips extent)
      const start = inPoint ?? 0;
      const end = outPoint ?? (clips.length > 0
        ? Math.max(...clips.map(c => c.startTime + c.duration))
        : duration);

      if (end <= start) return;

      // Import engine dynamically to avoid circular dependency
      const { engine } = await import('../engine/WebGPUEngine');

      // Tell engine to skip preview updates for efficiency
      engine.setGeneratingRamPreview(true);

      set({
        isRamPreviewing: true,
        ramPreviewProgress: 0,
        ramPreviewRange: null
      });

      const fps = 30; // Preview at 30fps
      const frameInterval = 1 / fps;

      // Helper: check if there's a video clip at a given time
      const hasVideoAt = (time: number) => {
        return clips.some(c =>
          time >= c.startTime &&
          time < c.startTime + c.duration &&
          (c.source?.type === 'video' || c.source?.type === 'image')
        );
      };

      // Generate frame times spreading outward from playhead
      // Only include times where there are video clips
      const centerTime = Math.max(start, Math.min(end, playheadPosition));
      const frameTimes: number[] = [];

      // Add center frame if it has video
      if (hasVideoAt(centerTime)) {
        frameTimes.push(centerTime);
      }

      // Alternate left and right from center, only adding frames with video
      let offset = frameInterval;
      while (offset <= (end - start)) {
        const rightTime = centerTime + offset;
        const leftTime = centerTime - offset;

        if (rightTime <= end && hasVideoAt(rightTime)) {
          frameTimes.push(rightTime);
        }
        if (leftTime >= start && hasVideoAt(leftTime)) {
          frameTimes.push(leftTime);
        }

        offset += frameInterval;
      }

      // No frames to render
      if (frameTimes.length === 0) {
        engine.setGeneratingRamPreview(false);
        set({ isRamPreviewing: false, ramPreviewProgress: null });
        return;
      }

      const totalFrames = frameTimes.length;
      let cancelled = false;

      // Store cancel function
      const checkCancelled = () => !get().isRamPreviewing;

      try {
        for (let frame = 0; frame < totalFrames; frame++) {
          if (checkCancelled()) {
            cancelled = true;
            break;
          }

          const time = frameTimes[frame];

          // Skip frames that are already cached (reuse existing work)
          const quantizedTime = Math.round(time * 30) / 30;
          if (get().cachedFrameTimes.has(quantizedTime)) {
            // Update progress even for skipped frames
            const progress = ((frame + 1) / totalFrames) * 100;
            set({ ramPreviewProgress: progress });
            continue;
          }

          // Get clips at this time
          const clipsAtTime = clips.filter(c =>
            time >= c.startTime && time < c.startTime + c.duration
          );

          // Build layers for this frame
          const videoTracks = tracks.filter(t => t.type === 'video');
          const layers: import('../types').Layer[] = [];

          // Seek all videos and build layers
          for (const clip of clipsAtTime) {
            const track = tracks.find(t => t.id === clip.trackId);
            if (!track?.visible || track.type !== 'video') continue;

            if (clip.source?.type === 'video' && clip.source.videoElement) {
              const video = clip.source.videoElement;
              const clipTime = time - clip.startTime + clip.inPoint;

              // Robust seek with verification and retry
              const seekWithVerify = async (targetTime: number, maxRetries = 3): Promise<boolean> => {
                for (let attempt = 0; attempt < maxRetries; attempt++) {
                  // Check if cancelled
                  if (checkCancelled()) return false;

                  // Seek to target time
                  await new Promise<void>((resolve) => {
                    const timeout = setTimeout(() => {
                      video.removeEventListener('seeked', onSeeked);
                      resolve();
                    }, 500);

                    const onSeeked = () => {
                      clearTimeout(timeout);
                      video.removeEventListener('seeked', onSeeked);
                      resolve();
                    };

                    video.addEventListener('seeked', onSeeked);
                    video.currentTime = targetTime;
                  });

                  // Wait for video to be fully ready (not seeking, has data)
                  await new Promise<void>((resolve) => {
                    const checkReady = () => {
                      if (!video.seeking && video.readyState >= 2) {
                        resolve();
                      } else {
                        requestAnimationFrame(checkReady);
                      }
                    };
                    checkReady();
                    // Timeout fallback
                    setTimeout(resolve, 200);
                  });

                  // Verify position is correct (within 1 frame tolerance at 30fps)
                  if (Math.abs(video.currentTime - targetTime) < 0.04) {
                    return true; // Success
                  }

                  // Position wrong (user scrubbed?), retry
                  if (checkCancelled()) return false;
                }
                return false; // Failed after retries
              };

              // Perform seek with verification
              const seekSuccess = await seekWithVerify(clipTime);
              if (!seekSuccess || checkCancelled()) {
                continue; // Skip this clip if seek failed or cancelled
              }

              // Add to layers
              layers.push({
                id: clip.id,
                name: clip.name,
                visible: true,
                opacity: clip.transform.opacity,
                blendMode: clip.transform.blendMode,
                source: { type: 'video', videoElement: video },
                effects: [],
                position: { x: clip.transform.position.x, y: clip.transform.position.y },
                scale: { x: clip.transform.scale.x, y: clip.transform.scale.y },
                rotation: clip.transform.rotation.z * (Math.PI / 180),
              });
            } else if (clip.source?.type === 'image' && clip.source.imageElement) {
              layers.push({
                id: clip.id,
                name: clip.name,
                visible: true,
                opacity: clip.transform.opacity,
                blendMode: clip.transform.blendMode,
                source: { type: 'image', imageElement: clip.source.imageElement },
                effects: [],
                position: { x: clip.transform.position.x, y: clip.transform.position.y },
                scale: { x: clip.transform.scale.x, y: clip.transform.scale.y },
                rotation: clip.transform.rotation.z * (Math.PI / 180),
              });
            }
          }

          // Sort layers by track order
          const trackOrder = new Map(videoTracks.map((t, i) => [t.id, i]));
          layers.sort((a, b) => {
            const clipA = clipsAtTime.find(c => c.id === a.id);
            const clipB = clipsAtTime.find(c => c.id === b.id);
            const orderA = clipA ? (trackOrder.get(clipA.trackId) ?? 0) : 0;
            const orderB = clipB ? (trackOrder.get(clipB.trackId) ?? 0) : 0;
            return orderA - orderB;
          });

          // Final verification: ensure all videos are still at correct position before rendering
          // This catches cases where user interaction changed position between seek and render
          let allPositionsCorrect = true;
          for (const clip of clipsAtTime) {
            if (clip.source?.type === 'video' && clip.source.videoElement) {
              const video = clip.source.videoElement;
              const expectedTime = time - clip.startTime + clip.inPoint;
              if (Math.abs(video.currentTime - expectedTime) > 0.04) {
                allPositionsCorrect = false;
                break;
              }
            }
          }

          // Skip this frame if positions are wrong (user scrubbed) or cancelled
          if (!allPositionsCorrect || checkCancelled()) {
            continue;
          }

          // Render and cache this frame
          if (layers.length > 0) {
            engine.render(layers);
          }
          await engine.cacheCompositeFrame(time);

          // Add to cached frames set (shows green indicator immediately)
          addCachedFrame(time);

          // Update progress percentage
          const progress = ((frame + 1) / totalFrames) * 100;
          set({ ramPreviewProgress: progress });

          // Yield to allow UI updates (every frame for smooth green dot updates)
          if (frame % 3 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        if (!cancelled) {
          // Set the preview range for cache hit detection
          set({
            ramPreviewRange: { start, end },
            ramPreviewProgress: null
          });
        }
      } catch (error) {
        console.error('[RAM Preview] Error:', error);
      } finally {
        engine.setGeneratingRamPreview(false);
        set({ isRamPreviewing: false, ramPreviewProgress: null });
      }
    },

    cancelRamPreview: () => {
      // IMMEDIATELY set state to cancel the loop - this must be synchronous!
      // The RAM preview loop checks !get().isRamPreviewing to know when to stop
      set({ isRamPreviewing: false, ramPreviewProgress: null });
      // Then async cleanup the engine
      import('../engine/WebGPUEngine').then(({ engine }) => {
        engine.setGeneratingRamPreview(false);
      });
    },

    clearRamPreview: async () => {
      const { engine } = await import('../engine/WebGPUEngine');
      engine.clearCompositeCache();
      set({ ramPreviewRange: null, ramPreviewProgress: null, cachedFrameTimes: new Set() });
    },

    // Playback frame caching (green line like After Effects)
    addCachedFrame: (time: number) => {
      const quantized = Math.round(time * 30) / 30; // Quantize to 30fps
      const { cachedFrameTimes } = get();
      if (!cachedFrameTimes.has(quantized)) {
        const newSet = new Set(cachedFrameTimes);
        newSet.add(quantized);
        set({ cachedFrameTimes: newSet });
      }
    },

    getCachedRanges: () => {
      const { cachedFrameTimes } = get();
      if (cachedFrameTimes.size === 0) return [];

      // Convert set to sorted array
      const times = Array.from(cachedFrameTimes).sort((a, b) => a - b);
      const ranges: Array<{ start: number; end: number }> = [];
      const frameInterval = 1 / 30;
      const gap = frameInterval * 2; // Allow gap of 2 frames

      let rangeStart = times[0];
      let rangeEnd = times[0];

      for (let i = 1; i < times.length; i++) {
        if (times[i] - rangeEnd <= gap) {
          // Continue range
          rangeEnd = times[i];
        } else {
          // End range and start new one
          ranges.push({ start: rangeStart, end: rangeEnd + frameInterval });
          rangeStart = times[i];
          rangeEnd = times[i];
        }
      }

      // Add final range
      ranges.push({ start: rangeStart, end: rangeEnd + frameInterval });

      return ranges;
    },

    // Invalidate cache when content changes (clip moved, trimmed, etc.)
    invalidateCache: () => {
      // Cancel any ongoing RAM preview
      set({ isRamPreviewing: false });
      // Clear the cache
      import('../engine/WebGPUEngine').then(({ engine }) => {
        engine.setGeneratingRamPreview(false);
        engine.clearCompositeCache();
      });
      // Clear cached frame times
      set({ cachedFrameTimes: new Set(), ramPreviewRange: null, ramPreviewProgress: null });
    },

    // Utils
    getClipsAtTime: (time) => {
      const { clips } = get();
      return clips.filter(c => time >= c.startTime && time < c.startTime + c.duration);
    },

    updateDuration: () => {
      const { clips } = get();
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
      console.log('[Timeline] Created new audio track:', newTrack.name);
      return newTrack.id;
    },

    // Get snapped position - snaps to edges of other clips on the same track
    getSnappedPosition: (clipId: string, desiredStartTime: number, trackId: string) => {
      const { clips } = get();
      const movingClip = clips.find(c => c.id === clipId);
      if (!movingClip) return { startTime: desiredStartTime, snapped: false };

      const clipDuration = movingClip.duration;
      const desiredEndTime = desiredStartTime + clipDuration;

      // Get other clips on the same track (excluding the moving clip and its linked clip)
      const otherClips = clips.filter(c =>
        c.trackId === trackId &&
        c.id !== clipId &&
        c.id !== movingClip.linkedClipId &&
        c.linkedClipId !== clipId
      );

      let snappedStart = desiredStartTime;
      let snapped = false;
      let minSnapDistance = SNAP_THRESHOLD_SECONDS;

      // Check snap points
      for (const clip of otherClips) {
        const clipEnd = clip.startTime + clip.duration;

        // Snap start of moving clip to end of other clip
        const distToEnd = Math.abs(desiredStartTime - clipEnd);
        if (distToEnd < minSnapDistance) {
          snappedStart = clipEnd;
          minSnapDistance = distToEnd;
          snapped = true;
        }

        // Snap start of moving clip to start of other clip
        const distToStart = Math.abs(desiredStartTime - clip.startTime);
        if (distToStart < minSnapDistance) {
          snappedStart = clip.startTime;
          minSnapDistance = distToStart;
          snapped = true;
        }

        // Snap end of moving clip to start of other clip
        const distEndToStart = Math.abs(desiredEndTime - clip.startTime);
        if (distEndToStart < minSnapDistance) {
          snappedStart = clip.startTime - clipDuration;
          minSnapDistance = distEndToStart;
          snapped = true;
        }

        // Snap end of moving clip to end of other clip
        const distEndToEnd = Math.abs(desiredEndTime - clipEnd);
        if (distEndToEnd < minSnapDistance) {
          snappedStart = clipEnd - clipDuration;
          minSnapDistance = distEndToEnd;
          snapped = true;
        }
      }

      // Also snap to timeline start (0)
      if (Math.abs(desiredStartTime) < SNAP_THRESHOLD_SECONDS) {
        snappedStart = 0;
        snapped = true;
      }

      return { startTime: Math.max(0, snappedStart), snapped };
    },

    // Find a valid non-overlapping position for a clip
    findNonOverlappingPosition: (clipId: string, desiredStartTime: number, trackId: string, duration: number) => {
      const { clips } = get();
      const movingClip = clips.find(c => c.id === clipId);

      // Get other clips on the same track (excluding the moving clip and its linked clip)
      const otherClips = clips.filter(c =>
        c.trackId === trackId &&
        c.id !== clipId &&
        (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
      ).sort((a, b) => a.startTime - b.startTime);

      const desiredEndTime = desiredStartTime + duration;

      // Check if desired position overlaps with any clip
      let overlappingClip: TimelineClip | null = null;
      for (const clip of otherClips) {
        const clipEnd = clip.startTime + clip.duration;
        // Check if time ranges overlap
        if (!(desiredEndTime <= clip.startTime || desiredStartTime >= clipEnd)) {
          overlappingClip = clip;
          break;
        }
      }

      // If no overlap, use desired position
      if (!overlappingClip) {
        return Math.max(0, desiredStartTime);
      }

      // There's an overlap - push clip to the nearest edge
      const overlappingEnd = overlappingClip.startTime + overlappingClip.duration;

      // Check which side is closer
      const distToStart = Math.abs(desiredStartTime - overlappingClip.startTime);
      const distToEnd = Math.abs(desiredStartTime - overlappingEnd);

      if (distToStart < distToEnd) {
        // Push to left side (end at overlapping clip's start)
        const newStart = overlappingClip.startTime - duration;

        // Check if this position overlaps with another clip
        const wouldOverlap = otherClips.some(c => {
          if (c.id === overlappingClip!.id) return false;
          const cEnd = c.startTime + c.duration;
          const newEnd = newStart + duration;
          return !(newEnd <= c.startTime || newStart >= cEnd);
        });

        if (!wouldOverlap && newStart >= 0) {
          return newStart;
        }
      }

      // Push to right side (start at overlapping clip's end)
      const newStart = overlappingEnd;

      // Check if this position overlaps with another clip
      const wouldOverlap = otherClips.some(c => {
        if (c.id === overlappingClip!.id) return false;
        const cEnd = c.startTime + c.duration;
        const newEnd = newStart + duration;
        return !(newEnd <= c.startTime || newStart >= cEnd);
      });

      if (!wouldOverlap) {
        return newStart;
      }

      // As a fallback, return the desired position (shouldn't happen often)
      return Math.max(0, desiredStartTime);
    },
  }))
);
