// Timeline store - combines all slices into a single Zustand store

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import type { TimelineStore, TimelineUtils, TimelineClip, Keyframe, CompositionTimelineData } from './types';
import type { SerializableClip } from '../../types';
import { DEFAULT_TRACKS, SNAP_THRESHOLD_SECONDS, OVERLAP_RESISTANCE_PIXELS } from './constants';
import { useMediaStore } from '../mediaStore';
import { useMixerStore } from '../mixerStore';

import { createTrackSlice } from './trackSlice';
import { createClipSlice } from './clipSlice';
import { createPlaybackSlice } from './playbackSlice';
import { createSelectionSlice } from './selectionSlice';
import { createKeyframeSlice } from './keyframeSlice';
import { createMaskSlice } from './maskSlice';
import { projectDB } from '../../services/projectDB';
import type { ClipAnalysis, FrameAnalysisData } from '../../types';

// Re-export types for convenience
export type { TimelineStore, TimelineClip, Keyframe } from './types';
export { DEFAULT_TRANSFORM, DEFAULT_TRACKS, SNAP_THRESHOLD_SECONDS } from './constants';
export { seekVideo, generateWaveform, generateThumbnails, getDefaultEffectParams } from './utils';

export const useTimelineStore = create<TimelineStore>()(
  subscribeWithSelector((set, get) => {
    // Create all slices
    const trackActions = createTrackSlice(set, get);
    const clipActions = createClipSlice(set, get);
    const playbackActions = createPlaybackSlice(set, get);
    const selectionActions = createSelectionSlice(set, get);
    const keyframeActions = createKeyframeSlice(set, get);
    const maskActions = createMaskSlice(set, get);

    // Utils that need to be defined inline due to cross-dependencies
    const utils: TimelineUtils = {
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
        console.log('[Timeline] Created new audio track:', newTrack.name);
        return newTrack.id;
      },

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

      // Apply magnetic resistance at clip edges during drag
      // Returns position with resistance applied, and whether user has "broken through" to force overlap
      // Uses PIXEL-based resistance so it works regardless of clip duration
      getPositionWithResistance: (clipId: string, desiredStartTime: number, trackId: string, duration: number, zoom?: number) => {
        const { clips, zoom: storeZoom } = get();
        const currentZoom = zoom ?? storeZoom;
        const movingClip = clips.find(c => c.id === clipId);

        // Get other clips on the TARGET track (excluding the moving clip and its linked clip)
        const otherClips = clips.filter(c =>
          c.trackId === trackId &&
          c.id !== clipId &&
          (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
        ).sort((a, b) => a.startTime - b.startTime);

        const desiredEndTime = desiredStartTime + duration;

        // Find the clip that would be overlapped
        let overlappingClip: TimelineClip | null = null;
        for (const clip of otherClips) {
          const clipEnd = clip.startTime + clip.duration;
          if (!(desiredEndTime <= clip.startTime || desiredStartTime >= clipEnd)) {
            overlappingClip = clip;
            break;
          }
        }

        // No overlap - return desired position
        if (!overlappingClip) {
          return { startTime: Math.max(0, desiredStartTime), forcingOverlap: false };
        }

        const overlappingEnd = overlappingClip.startTime + overlappingClip.duration;

        // Calculate which non-overlapping position is closer (before or after the other clip)
        const snapBeforePosition = overlappingClip.startTime - duration; // Place our clip END at other clip START
        const snapAfterPosition = overlappingEnd; // Place our clip START at other clip END

        const distToSnapBefore = Math.abs(desiredStartTime - snapBeforePosition);
        const distToSnapAfter = Math.abs(desiredStartTime - snapAfterPosition);

        // Choose the closer snap position
        const snapToPosition = distToSnapBefore < distToSnapAfter ? snapBeforePosition : snapAfterPosition;
        const distToSnapTime = Math.min(distToSnapBefore, distToSnapAfter);

        // Convert time distance to PIXELS using current zoom level
        const distToSnapPixels = distToSnapTime * currentZoom;

        // If the user hasn't dragged far enough past the snap point (in pixels), resist (snap back)
        if (distToSnapPixels < OVERLAP_RESISTANCE_PIXELS) {
          return { startTime: Math.max(0, snapToPosition), forcingOverlap: false };
        } else {
          // User has pushed through the resistance - allow overlap
          return { startTime: Math.max(0, desiredStartTime), forcingOverlap: true };
        }
      },

      // Trim any clips that the placed clip overlaps with
      trimOverlappingClips: (clipId: string, startTime: number, trackId: string, duration: number) => {
        const { clips, invalidateCache } = get();
        const movingClip = clips.find(c => c.id === clipId);

        // Get other clips on the same track (excluding the moving clip and its linked clip)
        const otherClips = clips.filter(c =>
          c.trackId === trackId &&
          c.id !== clipId &&
          (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
        );

        const endTime = startTime + duration;
        const clipsToModify: { id: string; action: 'trim-start' | 'trim-end' | 'delete' | 'split'; trimAmount?: number; splitTime?: number }[] = [];

        for (const clip of otherClips) {
          const clipEnd = clip.startTime + clip.duration;

          // Check if this clip overlaps with the placed clip
          if (!(endTime <= clip.startTime || startTime >= clipEnd)) {
            // There's overlap - determine how to handle it

            // Case 1: Placed clip completely covers this clip -> delete it
            if (startTime <= clip.startTime && endTime >= clipEnd) {
              clipsToModify.push({ id: clip.id, action: 'delete' });
            }
            // Case 2: Placed clip covers the start of this clip -> trim start
            else if (startTime <= clip.startTime && endTime < clipEnd) {
              const trimAmount = endTime - clip.startTime;
              clipsToModify.push({ id: clip.id, action: 'trim-start', trimAmount });
            }
            // Case 3: Placed clip covers the end of this clip -> trim end
            else if (startTime > clip.startTime && endTime >= clipEnd) {
              const trimAmount = clipEnd - startTime;
              clipsToModify.push({ id: clip.id, action: 'trim-end', trimAmount });
            }
            // Case 4: Placed clip is in the middle of this clip -> split and trim
            else if (startTime > clip.startTime && endTime < clipEnd) {
              // For now, just trim the end at the placed clip's start
              // (the "hole" in the middle - user can manually handle this)
              clipsToModify.push({ id: clip.id, action: 'trim-end', trimAmount: clipEnd - startTime });
            }
          }
        }

        // Apply modifications
        if (clipsToModify.length === 0) return;

        const clipIdsToDelete = new Set(clipsToModify.filter(m => m.action === 'delete').map(m => m.id));

        set({
          clips: clips
            .filter(c => !clipIdsToDelete.has(c.id))
            .map(c => {
              const modification = clipsToModify.find(m => m.id === c.id);
              if (!modification || modification.action === 'delete') return c;

              if (modification.action === 'trim-start' && modification.trimAmount) {
                // Trim start: move startTime forward, adjust inPoint
                const newStartTime = c.startTime + modification.trimAmount;
                const newInPoint = c.inPoint + modification.trimAmount;
                const newDuration = c.duration - modification.trimAmount;
                return {
                  ...c,
                  startTime: newStartTime,
                  inPoint: newInPoint,
                  duration: newDuration,
                };
              }

              if (modification.action === 'trim-end' && modification.trimAmount) {
                // Trim end: reduce duration and outPoint
                const newDuration = c.duration - modification.trimAmount;
                const newOutPoint = c.outPoint - modification.trimAmount;
                return {
                  ...c,
                  duration: newDuration,
                  outPoint: newOutPoint,
                };
              }

              return c;
            }),
        });

        invalidateCache();
      },

      // Get serializable timeline state for saving to composition
      getSerializableState: (): CompositionTimelineData => {
        const { tracks, clips, playheadPosition, duration, durationLocked, zoom, scrollX, inPoint, outPoint, loopPlayback, clipKeyframes } = get();

        // Convert clips to serializable format (without DOM elements)
        const mediaStore = useMediaStore.getState();
        const serializableClips: SerializableClip[] = clips.map(clip => {
          // Find the mediaFile ID by matching the file name in mediaStore
          // For linked audio clips (name ends with "(Audio)"), strip the suffix to find the video file
          let lookupName = clip.name;
          if (clip.linkedClipId && clip.source?.type === 'audio' && lookupName.endsWith(' (Audio)')) {
            lookupName = lookupName.replace(' (Audio)', '');
          }
          const mediaFile = mediaStore.files.find(f => f.name === lookupName);

          // Get keyframes for this clip
          const keyframes = clipKeyframes.get(clip.id) || [];

          return {
            id: clip.id,
            trackId: clip.trackId,
            name: clip.name,
            mediaFileId: clip.isComposition ? '' : (mediaFile?.id || ''), // Comp clips don't have media files
            startTime: clip.startTime,
            duration: clip.duration,
            inPoint: clip.inPoint,
            outPoint: clip.outPoint,
            sourceType: clip.source?.type || 'video',
            naturalDuration: clip.source?.naturalDuration,
            thumbnails: clip.thumbnails,
            linkedClipId: clip.linkedClipId,
            linkedGroupId: clip.linkedGroupId,
            waveform: clip.waveform,
            transform: clip.transform,
            effects: clip.effects,
            keyframes: keyframes.length > 0 ? keyframes : undefined,
            // Nested composition support
            isComposition: clip.isComposition,
            compositionId: clip.compositionId,
            // Mask support
            masks: clip.masks && clip.masks.length > 0 ? clip.masks : undefined,
            // Transcript data
            transcript: clip.transcript && clip.transcript.length > 0 ? clip.transcript : undefined,
            transcriptStatus: clip.transcriptStatus !== 'none' ? clip.transcriptStatus : undefined,
            // Analysis data
            analysis: clip.analysis,
            analysisStatus: clip.analysisStatus !== 'none' ? clip.analysisStatus : undefined,
            // Playback
            reversed: clip.reversed || undefined,
          };
        });

        return {
          tracks,
          clips: serializableClips,
          playheadPosition,
          duration,
          durationLocked: durationLocked || undefined,  // Only save if true
          zoom,
          scrollX,
          inPoint,
          outPoint,
          loopPlayback,
        };
      },

      // Load timeline state from composition data
      loadState: async (data: CompositionTimelineData | undefined) => {
        const { pause, clearTimeline } = get();

        // Stop playback
        pause();

        // Clear current timeline
        clearTimeline();

        if (!data) {
          // No data - start with fresh default timeline
          set({
            tracks: DEFAULT_TRACKS.map(t => ({ ...t })),
            clips: [],
            playheadPosition: 0,
            duration: 60,
            durationLocked: false,
            zoom: 50,
            scrollX: 0,
            inPoint: null,
            outPoint: null,
            loopPlayback: false,
            selectedClipIds: new Set(),
          });
          return;
        }

        // Restore tracks and basic state
        set({
          tracks: data.tracks.map(t => ({ ...t })),
          clips: [], // We'll restore clips separately
          playheadPosition: data.playheadPosition,
          duration: data.duration,
          durationLocked: data.durationLocked || false,
          zoom: data.zoom,
          scrollX: data.scrollX,
          inPoint: data.inPoint,
          outPoint: data.outPoint,
          loopPlayback: data.loopPlayback,
          selectedClipIds: new Set(),
          // Clear keyframe state
          clipKeyframes: new Map<string, Keyframe[]>(),
          keyframeRecordingEnabled: new Set<string>(),
          expandedTracks: new Set<string>(data.tracks.filter(t => t.type === 'video').map(t => t.id)),
          expandedTrackPropertyGroups: new Map<string, Set<string>>(),
          selectedKeyframeIds: new Set<string>(),
        });

        // Restore keyframes from serialized clips
        const keyframeMap = new Map<string, Keyframe[]>();
        for (const serializedClip of data.clips) {
          if (serializedClip.keyframes && serializedClip.keyframes.length > 0) {
            keyframeMap.set(serializedClip.id, serializedClip.keyframes);
          }
        }
        if (keyframeMap.size > 0) {
          set({ clipKeyframes: keyframeMap });
        }

        // Restore clips - need to recreate media elements from file references
        const mediaStore = useMediaStore.getState();

        for (const serializedClip of data.clips) {
          // Handle composition clips specially
          if (serializedClip.isComposition && serializedClip.compositionId) {
            const composition = mediaStore.compositions.find(c => c.id === serializedClip.compositionId);
            if (composition) {
              // Create comp clip manually to restore specific settings
              const compClip: TimelineClip = {
                id: serializedClip.id,
                trackId: serializedClip.trackId,
                name: serializedClip.name,
                file: new File([], serializedClip.name),
                startTime: serializedClip.startTime,
                duration: serializedClip.duration,
                inPoint: serializedClip.inPoint,
                outPoint: serializedClip.outPoint,
                source: {
                  type: 'video',
                  naturalDuration: serializedClip.duration,
                },
                thumbnails: serializedClip.thumbnails,
                transform: serializedClip.transform,
                effects: serializedClip.effects || [],
                isLoading: true,
                isComposition: true,
                compositionId: serializedClip.compositionId,
                nestedClips: [],
                nestedTracks: [],
              };

              // Add clip to state
              set(state => ({
                clips: [...state.clips, compClip],
              }));

              // Load nested composition content in background
              if (composition.timelineData) {
                const nestedClips: TimelineClip[] = [];
                const nestedTracks = composition.timelineData.tracks;

                for (const nestedSerializedClip of composition.timelineData.clips) {
                  const nestedMediaFile = mediaStore.files.find(f => f.id === nestedSerializedClip.mediaFileId);
                  if (!nestedMediaFile || !nestedMediaFile.file) continue;

                  const nestedClip: TimelineClip = {
                    id: `nested-${compClip.id}-${nestedSerializedClip.id}`,
                    trackId: nestedSerializedClip.trackId,
                    name: nestedSerializedClip.name,
                    file: nestedMediaFile.file,
                    startTime: nestedSerializedClip.startTime,
                    duration: nestedSerializedClip.duration,
                    inPoint: nestedSerializedClip.inPoint,
                    outPoint: nestedSerializedClip.outPoint,
                    source: null,
                    thumbnails: nestedSerializedClip.thumbnails,
                    transform: nestedSerializedClip.transform,
                    effects: nestedSerializedClip.effects || [],
                    isLoading: true,
                  };

                  nestedClips.push(nestedClip);

                  // Load media element
                  const nestedType = nestedSerializedClip.sourceType;
                  const nestedFileUrl = URL.createObjectURL(nestedMediaFile.file);

                  if (nestedType === 'video') {
                    const video = document.createElement('video');
                    video.src = nestedFileUrl;
                    video.muted = true;
                    video.playsInline = true;
                    video.preload = 'auto';
                    video.crossOrigin = 'anonymous';

                    video.addEventListener('canplaythrough', () => {
                      nestedClip.source = {
                        type: 'video',
                        videoElement: video,
                        naturalDuration: video.duration,
                      };
                      nestedClip.isLoading = false;
                      // Trigger update
                      const currentClips = get().clips;
                      set({ clips: [...currentClips] });
                    }, { once: true });
                  } else if (nestedType === 'image') {
                    const img = new Image();
                    img.src = nestedFileUrl;
                    img.addEventListener('load', () => {
                      nestedClip.source = {
                        type: 'image',
                        imageElement: img,
                      };
                      nestedClip.isLoading = false;
                      const currentClips = get().clips;
                      set({ clips: [...currentClips] });
                    }, { once: true });
                  }
                }

                // Update comp clip with nested data
                set(state => ({
                  clips: state.clips.map(c =>
                    c.id === compClip.id
                      ? { ...c, nestedClips, nestedTracks, isLoading: false }
                      : c
                  ),
                }));
              } else {
                // No timeline data
                set(state => ({
                  clips: state.clips.map(c =>
                    c.id === compClip.id ? { ...c, isLoading: false } : c
                  ),
                }));
              }
            } else {
              console.warn('Could not find composition for clip:', serializedClip.name);
            }
            continue;
          }

          // Regular media clips
          const mediaFile = mediaStore.files.find(f => f.id === serializedClip.mediaFileId);
          if (!mediaFile || !mediaFile.file) {
            console.warn('Could not find media file for clip:', serializedClip.name);
            continue;
          }

          // Create the clip with loading state
          const clip: TimelineClip = {
            id: serializedClip.id,
            trackId: serializedClip.trackId,
            name: serializedClip.name,
            file: mediaFile.file,
            startTime: serializedClip.startTime,
            duration: serializedClip.duration,
            inPoint: serializedClip.inPoint,
            outPoint: serializedClip.outPoint,
            source: {
              type: serializedClip.sourceType,
              mediaFileId: serializedClip.mediaFileId, // Preserve mediaFileId for cache lookups
              naturalDuration: serializedClip.naturalDuration,
            },
            thumbnails: serializedClip.thumbnails,
            linkedClipId: serializedClip.linkedClipId,
            linkedGroupId: serializedClip.linkedGroupId,
            waveform: serializedClip.waveform,
            transform: serializedClip.transform,
            effects: serializedClip.effects || [],
            isLoading: true,
            masks: serializedClip.masks,  // Restore masks
            // Restore transcript data
            transcript: serializedClip.transcript,
            transcriptStatus: serializedClip.transcriptStatus || 'none',
            // Restore analysis data
            analysis: serializedClip.analysis,
            analysisStatus: serializedClip.analysisStatus || 'none',
            // Restore playback settings
            reversed: serializedClip.reversed,
          };

          // Add clip to state
          set(state => ({
            clips: [...state.clips, clip],
          }));

          // Check for cached analysis if clip doesn't have analysis but has mediaFileId
          if (!serializedClip.analysis && serializedClip.mediaFileId) {
            projectDB.getAnalysis(
              serializedClip.mediaFileId,
              serializedClip.inPoint,
              serializedClip.outPoint
            ).then(cachedAnalysis => {
              if (cachedAnalysis) {
                console.log('[Timeline] Loaded analysis from cache for:', serializedClip.name);
                const analysis: ClipAnalysis = {
                  frames: cachedAnalysis.frames as FrameAnalysisData[],
                  sampleInterval: cachedAnalysis.sampleInterval,
                };
                set(state => ({
                  clips: state.clips.map(c =>
                    c.id === clip.id
                      ? { ...c, analysis, analysisStatus: 'ready' as const }
                      : c
                  ),
                }));
              }
            }).catch(err => {
              console.warn('[Timeline] Failed to load cached analysis:', err);
            });
          }

          // Load media element async
          const type = serializedClip.sourceType;
          const fileUrl = URL.createObjectURL(mediaFile.file);

          if (type === 'video') {
            const video = document.createElement('video');
            video.src = fileUrl;
            video.muted = true;
            video.playsInline = true;
            video.preload = 'auto';

            video.addEventListener('canplaythrough', () => {
              set(state => ({
                clips: state.clips.map(c =>
                  c.id === clip.id
                    ? {
                        ...c,
                        source: {
                          type: 'video',
                          videoElement: video,
                          naturalDuration: video.duration,
                          mediaFileId: serializedClip.mediaFileId, // Needed for multicam sync
                        },
                        isLoading: false,
                      }
                    : c
                ),
              }));
            }, { once: true });
          } else if (type === 'audio') {
            // Audio clips - create audio element (works for both pure audio files and linked audio from video)
            const audio = document.createElement('audio');
            audio.src = fileUrl;
            audio.preload = 'auto';

            audio.addEventListener('canplaythrough', () => {
              set(state => ({
                clips: state.clips.map(c =>
                  c.id === clip.id
                    ? {
                        ...c,
                        source: {
                          type: 'audio',
                          audioElement: audio,
                          naturalDuration: audio.duration,
                          mediaFileId: serializedClip.mediaFileId, // Needed for multicam sync
                        },
                        isLoading: false,
                      }
                    : c
                ),
              }));
            }, { once: true });
          } else if (type === 'image') {
            const img = new Image();
            img.src = fileUrl;

            img.addEventListener('load', () => {
              set(state => ({
                clips: state.clips.map(c =>
                  c.id === clip.id
                    ? {
                        ...c,
                        source: { type: 'image', imageElement: img },
                        isLoading: false,
                      }
                    : c
                ),
              }));
            }, { once: true });
          }
        }
      },

      // Clear all timeline data
      clearTimeline: () => {
        const { clips, pause } = get();

        // Stop playback
        pause();

        // Clean up media elements
        clips.forEach(clip => {
          if (clip.source?.videoElement) {
            clip.source.videoElement.pause();
            clip.source.videoElement.src = '';
          }
          if (clip.source?.audioElement) {
            clip.source.audioElement.pause();
            clip.source.audioElement.src = '';
          }
          if (clip.source?.webCodecsPlayer) {
            clip.source.webCodecsPlayer.destroy();
          }
        });

        // Clear mixer store layers so preview shows black
        useMixerStore.setState({ layers: [] });

        const { tracks } = get();
        set({
          clips: [],
          selectedClipIds: new Set(),
          cachedFrameTimes: new Set(),
          ramPreviewProgress: null,
          ramPreviewRange: null,
          isRamPreviewing: false,
          // Clear keyframe state
          clipKeyframes: new Map<string, Keyframe[]>(),
          keyframeRecordingEnabled: new Set<string>(),
          expandedTracks: new Set<string>(tracks.filter(t => t.type === 'video').map(t => t.id)),
          expandedTrackPropertyGroups: new Map<string, Set<string>>(),
          selectedKeyframeIds: new Set<string>(),
        });
      },
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
      isPlaying: false,
      isDraggingPlayhead: false,
      selectedClipIds: new Set<string>(),

      // In/Out markers
      inPoint: null as number | null,
      outPoint: null as number | null,
      loopPlayback: false,

      // Duration lock (when true, duration won't auto-update based on clips)
      durationLocked: false,

      // RAM Preview state
      ramPreviewEnabled: false,
      ramPreviewProgress: null as number | null,
      ramPreviewRange: null as { start: number; end: number } | null,
      isRamPreviewing: false,
      cachedFrameTimes: new Set<number>(),

      // Performance toggles (enabled by default)
      thumbnailsEnabled: true,
      waveformsEnabled: true,

      // Keyframe animation state
      clipKeyframes: new Map<string, Keyframe[]>(),
      keyframeRecordingEnabled: new Set<string>(),
      expandedTracks: new Set<string>(DEFAULT_TRACKS.filter(t => t.type === 'video').map(t => t.id)),
      expandedTrackPropertyGroups: new Map<string, Set<string>>(),
      selectedKeyframeIds: new Set<string>(),

      // Mask state
      maskEditMode: 'none' as const,
      activeMaskId: null as string | null,
      selectedVertexIds: new Set<string>(),
      maskDrawStart: null as { x: number; y: number } | null,
    };

    return {
      ...initialState,
      ...trackActions,
      ...clipActions,
      ...playbackActions,
      ...selectionActions,
      ...keyframeActions,
      ...maskActions,
      ...utils,
    };
  })
);
