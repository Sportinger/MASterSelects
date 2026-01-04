// Timeline store for video editing functionality

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { TimelineClip, TimelineTrack, ClipTransform, BlendMode } from '../types';

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

interface TimelineStore {
  // State
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  playheadPosition: number;
  duration: number;
  zoom: number;  // pixels per second
  scrollX: number;
  isPlaying: boolean;
  selectedClipId: string | null;

  // Track actions
  addTrack: (type: 'video' | 'audio') => void;
  removeTrack: (id: string) => void;
  setTrackMuted: (id: string, muted: boolean) => void;
  setTrackVisible: (id: string, visible: boolean) => void;
  setTrackHeight: (id: string, height: number) => void;
  scaleTracksOfType: (type: 'video' | 'audio', delta: number) => void;



  // Clip actions
  addClip: (trackId: string, file: File, startTime: number) => Promise<void>;
  removeClip: (id: string) => void;
  moveClip: (id: string, newStartTime: number, newTrackId?: string, skipLinked?: boolean) => void;
  trimClip: (id: string, inPoint: number, outPoint: number) => void;
  selectClip: (id: string | null) => void;
  updateClipTransform: (id: string, transform: Partial<ClipTransform>) => void;

  // Playback actions
  setPlayheadPosition: (position: number) => void;
  play: () => void;
  pause: () => void;
  stop: () => void;

  // View actions
  setZoom: (zoom: number) => void;
  setScrollX: (scrollX: number) => void;

  // Utils
  getClipsAtTime: (time: number) => TimelineClip[];
  updateDuration: () => void;
  findAvailableAudioTrack: (startTime: number, duration: number) => string;
}

const DEFAULT_TRACKS: TimelineTrack[] = [
  { id: 'video-1', name: 'Video 1', type: 'video', height: 60, muted: false, visible: true },
  { id: 'video-2', name: 'Video 2', type: 'video', height: 60, muted: false, visible: true },
  { id: 'audio-1', name: 'Audio', type: 'audio', height: 40, muted: false, visible: true },
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
    selectedClipId: null,

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
    },

    setTrackVisible: (id, visible) => {
      const { tracks } = get();
      set({
        tracks: tracks.map(t => t.id === id ? { ...t, visible } : t),
      });
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
    addClip: async (trackId, file, startTime) => {
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

      // Create media element to get duration
      let naturalDuration = 5; // Default for images
      let source: TimelineClip['source'] = null;
      let thumbnails: string[] = [];

      if (isVideo) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.preload = 'auto';
        video.muted = true;
        video.crossOrigin = 'anonymous';

        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => {
            naturalDuration = video.duration;
            resolve();
          };
          video.onerror = () => resolve();
        });

        // Wait for video to be ready for thumbnail extraction
        await new Promise<void>((resolve) => {
          if (video.readyState >= 2) {
            resolve();
          } else {
            video.oncanplay = () => resolve();
          }
        });

        // Generate thumbnails
        try {
          thumbnails = await generateThumbnails(video, naturalDuration);
          console.log(`[Timeline] Generated ${thumbnails.length} thumbnails for ${file.name}`);
        } catch (e) {
          console.warn('Failed to generate thumbnails:', e);
        }

        // Reset video to start
        video.currentTime = 0;

        source = {
          type: 'video',
          videoElement: video,
          naturalDuration,
        };

        // Create linked audio clip - find available audio track or create new one
        const { findAvailableAudioTrack } = get();
        const audioTrackId = findAvailableAudioTrack(startTime, naturalDuration);
        if (audioTrackId) {
          // Create audio element from same video file
          const audioFromVideo = document.createElement('audio');
          audioFromVideo.src = URL.createObjectURL(file);
          audioFromVideo.preload = 'auto';

          // Generate waveform for audio
          let audioWaveform: number[] = [];
          try {
            audioWaveform = await generateWaveform(file);
            console.log('[Timeline] Generated waveform for', file.name);
          } catch (e) {
            console.warn('Failed to generate waveform:', e);
          }


          const audioClipId = `clip-audio-${Date.now()}`;
          const audioClip: TimelineClip = {
            id: audioClipId,
            trackId: audioTrackId,
            name: `${file.name} (Audio)`,
            file,
            startTime,
            duration: naturalDuration,
            inPoint: 0,
            outPoint: naturalDuration,
            source: {
              type: 'audio',
              audioElement: audioFromVideo,
              naturalDuration,
            },
            linkedClipId: clipId, // Link to video clip
            waveform: audioWaveform,
            transform: { ...DEFAULT_TRANSFORM },
          };

          // Add audio clip and link video to audio
          const { clips: currentClips, updateDuration } = get();
          const videoClip: TimelineClip = {
            id: clipId,
            trackId,
            name: file.name,
            file,
            startTime,
            duration: naturalDuration,
            inPoint: 0,
            outPoint: naturalDuration,
            source,
            thumbnails,
            linkedClipId: audioClipId, // Link to audio clip
            transform: { ...DEFAULT_TRANSFORM },
          };

          set({ clips: [...currentClips, videoClip, audioClip] });
          updateDuration();
          return; // Exit early, we've handled everything
        }
      } else if (isAudio) {
        const audio = document.createElement('audio');
        audio.src = URL.createObjectURL(file);
        audio.preload = 'metadata';

        await new Promise<void>((resolve) => {
          audio.onloadedmetadata = () => {
            naturalDuration = audio.duration;
            resolve();
          };
          audio.onerror = () => resolve();
        });

        source = {
          type: 'audio',
          audioElement: audio,
          naturalDuration,
        };

        // Generate waveform for standalone audio
        try {
          const waveformData = await generateWaveform(file);
          console.log('[Timeline] Generated waveform for', file.name);
          (source as any)._waveform = waveformData;
        } catch (e) {
          console.warn('Failed to generate waveform:', e);
        }
      } else if (isImage) {
        const img = new Image();
        img.src = URL.createObjectURL(file);

        await new Promise<void>((resolve) => {
          img.onload = () => resolve();
          img.onerror = () => resolve();
        });

        // Generate single thumbnail for image
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

        source = {
          type: 'image',
          imageElement: img,
          naturalDuration: 5, // Default 5 seconds for images
        };
      }

      const newClip: TimelineClip = {
        id: clipId,
        trackId,
        name: file.name,
        file,
        startTime,
        duration: naturalDuration,
        inPoint: 0,
        outPoint: naturalDuration,
        source,
        thumbnails,
        transform: { ...DEFAULT_TRANSFORM },
      };

      const { clips, updateDuration } = get();
      set({ clips: [...clips, newClip] });
      updateDuration();
    },

    removeClip: (id) => {
      const { clips, selectedClipId, updateDuration } = get();
      set({
        clips: clips.filter(c => c.id !== id),
        selectedClipId: selectedClipId === id ? null : selectedClipId,
      });
      updateDuration();
    },

    moveClip: (id, newStartTime, newTrackId, skipLinked = false) => {
      const { clips, tracks, updateDuration } = get();
      const movingClip = clips.find(c => c.id === id);
      if (!movingClip) return;

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

      // Calculate time delta to apply to linked clips
      const timeDelta = newStartTime - movingClip.startTime;

      set({
        clips: clips.map(c => {
          // Move the primary clip
          if (c.id === id) {
            return {
              ...c,
              startTime: Math.max(0, newStartTime),
              trackId: newTrackId ?? c.trackId,
            };
          }
          // Also move linked clip (keep it in sync) - unless skipLinked is true
          if (!skipLinked && (c.id === movingClip.linkedClipId || c.linkedClipId === id)) {
            return {
              ...c,
              startTime: Math.max(0, c.startTime + timeDelta),
              // Keep linked clip on its own track (don't change track)
            };
          }
          return c;
        }),
      });
      updateDuration();
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
    },

    // Playback actions
    setPlayheadPosition: (position) => {
      const { duration } = get();
      set({ playheadPosition: Math.max(0, Math.min(position, duration)) });
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
  }))
);
