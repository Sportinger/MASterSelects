// Clip-related actions slice

import type { TimelineClip, Effect, EffectType } from '../../types';
import type { ClipActions, SliceCreator, Composition } from './types';
import { useMediaStore } from '../mediaStore';
import { DEFAULT_TRANSFORM } from './constants';
import { generateWaveform, generateThumbnails, getDefaultEffectParams } from './utils';

export const createClipSlice: SliceCreator<ClipActions> = (set, get) => ({
  addClip: async (trackId, file, startTime, providedDuration, mediaFileId) => {
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
        source: { type: 'video', naturalDuration: estimatedDuration, mediaFileId },
        linkedClipId: audioTrackId ? audioClipId : undefined,
        transform: { ...DEFAULT_TRANSFORM },
        effects: [],
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
          source: { type: 'audio', naturalDuration: estimatedDuration, mediaFileId },
          linkedClipId: clipId,
          transform: { ...DEFAULT_TRANSFORM },
          effects: [],
          isLoading: true,
        };
        clipsToAdd.push(audioClip);
      }

      set({ clips: [...currentClips, ...clipsToAdd] });
      updateDuration();

      // Now load media in background
      // Always create HTMLVideoElement for thumbnails and fallback
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
        source: { type: 'video', videoElement: video, naturalDuration, mediaFileId },
      });
      if (audioTrackId && audioClipId) {
        updateClip(audioClipId, {
          duration: naturalDuration,
          outPoint: naturalDuration,
        });
      }

      // Mark clip as ready immediately - thumbnails will load in background
      updateClip(clipId, {
        source: {
          type: 'video',
          videoElement: video,
          naturalDuration,
          mediaFileId,
        },
        isLoading: false,
      });

      // Generate thumbnails in background (non-blocking) - only if enabled
      if (get().thumbnailsEnabled) {
        (async () => {
          try {
            // Wait for video to be ready for thumbnails
            await new Promise<void>((resolve) => {
              if (video.readyState >= 2) {
                resolve();
              } else {
                video.oncanplay = () => resolve();
                setTimeout(resolve, 2000); // Timeout fallback
              }
            });

            // Check again in case toggle was turned off while waiting
            if (!get().thumbnailsEnabled) return;

            console.log(`[Thumbnails] Starting generation for ${file.name}...`);
            const thumbnails = await generateThumbnails(video, naturalDuration);
            console.log(`[Thumbnails] Complete: ${thumbnails.length} thumbnails for ${file.name}`);

            // Update clip with thumbnails
            const currentClips = get().clips;
            set({
              clips: currentClips.map(c => c.id === clipId ? { ...c, thumbnails } : c)
            });

            // Seek back to start
            video.currentTime = 0;
          } catch (e) {
            console.warn('Failed to generate thumbnails:', e);
          }
        })();
      }

      // Load audio - make it ready immediately, waveform loads in background
      if (audioTrackId && audioClipId) {
        const audioFromVideo = document.createElement('audio');
        audioFromVideo.src = URL.createObjectURL(file);
        audioFromVideo.preload = 'auto';

        // Mark audio clip as ready immediately
        updateClip(audioClipId, {
          source: { type: 'audio', audioElement: audioFromVideo, naturalDuration, mediaFileId },
          isLoading: false,
        });

        // Generate waveform in background (non-blocking) - only if enabled
        if (get().waveformsEnabled && audioClipId) {
          // Mark waveform generation starting
          const clipsBefore = get().clips;
          set({
            clips: clipsBefore.map(c => c.id === audioClipId ? { ...c, waveformGenerating: true, waveformProgress: 0 } : c)
          });

          (async () => {
            try {
              // Check again before expensive operation
              if (!get().waveformsEnabled) {
                const clipsNow = get().clips;
                set({
                  clips: clipsNow.map(c => c.id === audioClipId ? { ...c, waveformGenerating: false } : c)
                });
                return;
              }

              console.log(`[Waveform] Starting generation for ${file.name}...`);
              const audioWaveform = await generateWaveform(file);
              console.log(`[Waveform] Complete: ${audioWaveform.length} samples for ${file.name}`);
              const currentClips = get().clips;
              set({
                clips: currentClips.map(c => c.id === audioClipId ? { ...c, waveform: audioWaveform, waveformGenerating: false, waveformProgress: 100 } : c)
              });
            } catch (e) {
              console.warn('Failed to generate waveform:', e);
              const clipsErr = get().clips;
              set({
                clips: clipsErr.map(c => c.id === audioClipId ? { ...c, waveformGenerating: false } : c)
              });
            }
          })();
        }
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
        source: { type: 'audio', naturalDuration: estimatedDuration, mediaFileId },
        transform: { ...DEFAULT_TRANSFORM },
        effects: [],
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

      // Mark clip as ready first (waveform will load in background)
      updateClip(clipId, {
        duration: naturalDuration,
        outPoint: naturalDuration,
        source: { type: 'audio', audioElement: audio, naturalDuration, mediaFileId },
        isLoading: false,
        waveformGenerating: get().waveformsEnabled,
        waveformProgress: 0,
      });

      // Generate waveform in background - only if enabled
      if (get().waveformsEnabled) {
        (async () => {
          try {
            console.log(`[Waveform] Starting generation for ${file.name}...`);
            const waveform = await generateWaveform(file);
            console.log(`[Waveform] Complete: ${waveform.length} samples for ${file.name}`, waveform.slice(0, 5));

            // Verify clip still exists before updating
            const clipExists = get().clips.find(c => c.id === clipId);
            if (!clipExists) {
              console.warn('[Waveform] Clip no longer exists, skipping update');
              return;
            }

            updateClip(clipId, {
              waveform,
              waveformGenerating: false,
              waveformProgress: 100,
            });

            // Verify the update worked
            const updatedClip = get().clips.find(c => c.id === clipId);
            console.log(`[Waveform] After update - clip has waveform: ${updatedClip?.waveform?.length || 0} samples`);
          } catch (e) {
            console.warn('[Waveform] Failed:', e);
            updateClip(clipId, {
              waveformGenerating: false,
            });
          }
        })();
      }

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
        effects: [],
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

  // Add a composition as a clip (nested composition)
  addCompClip: async (trackId, composition: Composition, startTime) => {
    const { clips, updateDuration, findNonOverlappingPosition } = get();

    const clipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Find non-overlapping position
    const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, composition.duration);

    // Create placeholder clip immediately (will be updated with nested content)
    const compClip: TimelineClip = {
      id: clipId,
      trackId,
      name: composition.name,
      file: new File([], composition.name), // Placeholder file
      startTime: finalStartTime,
      duration: composition.duration,
      inPoint: 0,
      outPoint: composition.duration,
      source: {
        type: 'video', // Comp clips are treated as video
        naturalDuration: composition.duration,
      },
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      isLoading: true,
      isComposition: true,
      compositionId: composition.id,
      nestedClips: [],
      nestedTracks: [],
    };

    set({ clips: [...clips, compClip] });
    updateDuration();

    // Load nested composition's clips in background
    if (composition.timelineData) {
      const mediaStore = useMediaStore.getState();
      const nestedClips: TimelineClip[] = [];
      const nestedTracks = composition.timelineData.tracks;

      for (const serializedClip of composition.timelineData.clips) {
        // Find the media file
        const mediaFile = mediaStore.files.find(f => f.id === serializedClip.mediaFileId);
        if (!mediaFile || !mediaFile.file) {
          console.warn('[Nested Comp] Could not find media file for clip:', serializedClip.name);
          continue;
        }

        // Create the clip with loading state
        const nestedClip: TimelineClip = {
          id: `nested-${clipId}-${serializedClip.id}`,
          trackId: serializedClip.trackId,
          name: serializedClip.name,
          file: mediaFile.file,
          startTime: serializedClip.startTime,
          duration: serializedClip.duration,
          inPoint: serializedClip.inPoint,
          outPoint: serializedClip.outPoint,
          source: null,
          thumbnails: serializedClip.thumbnails,
          linkedClipId: serializedClip.linkedClipId,
          waveform: serializedClip.waveform,
          transform: serializedClip.transform,
          effects: serializedClip.effects || [],
          isLoading: true,
        };

        nestedClips.push(nestedClip);

        // Load media element async
        const type = serializedClip.sourceType;
        const fileUrl = URL.createObjectURL(mediaFile.file);

        if (type === 'video') {
          const video = document.createElement('video');
          video.src = fileUrl;
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
            // Trigger re-render by updating the parent clip
            const currentClips = get().clips;
            set({ clips: [...currentClips] });
          }, { once: true });
        } else if (type === 'audio') {
          const audio = document.createElement('audio');
          audio.src = fileUrl;
          audio.preload = 'auto';

          audio.addEventListener('canplaythrough', () => {
            nestedClip.source = {
              type: 'audio',
              audioElement: audio,
              naturalDuration: audio.duration,
            };
            nestedClip.isLoading = false;
          }, { once: true });
        } else if (type === 'image') {
          const img = new Image();
          img.src = fileUrl;

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

      // Update the comp clip with nested data
      const currentClips = get().clips;
      set({
        clips: currentClips.map(c =>
          c.id === clipId
            ? { ...c, nestedClips, nestedTracks, isLoading: false }
            : c
        ),
      });

      // Generate thumbnails from first video in nested comp - only if enabled
      const firstVideoClip = nestedClips.find(c => c.file.type.startsWith('video/'));
      if (firstVideoClip && get().thumbnailsEnabled) {
        // Wait a bit for video to load
        setTimeout(async () => {
          if (!get().thumbnailsEnabled) return;
          const video = firstVideoClip.source?.videoElement;
          if (video && video.readyState >= 2) {
            try {
              const thumbnails = await generateThumbnails(video, composition.duration);
              const updatedClips = get().clips;
              set({
                clips: updatedClips.map(c =>
                  c.id === clipId ? { ...c, thumbnails } : c
                ),
              });
            } catch (e) {
              console.warn('[Nested Comp] Failed to generate thumbnails:', e);
            }
          }
        }, 500);
      }
    } else {
      // No timeline data - just mark as loaded
      const currentClips = get().clips;
      set({
        clips: currentClips.map(c =>
          c.id === clipId ? { ...c, isLoading: false } : c
        ),
      });
    }

    get().invalidateCache();
  },

  removeClip: (id) => {
    const { clips, selectedClipIds, updateDuration, invalidateCache } = get();

    // Find the clip to clean up its resources
    const clipToRemove = clips.find(c => c.id === id);
    if (clipToRemove) {
      // Clean up video/audio resources
      if (clipToRemove.source?.type === 'video' && clipToRemove.source.videoElement) {
        const video = clipToRemove.source.videoElement;
        // Revoke blob URL to free memory
        if (video.src && video.src.startsWith('blob:')) {
          URL.revokeObjectURL(video.src);
        }
        // Pause and clear video
        video.pause();
        video.src = '';
        video.load();
        // Clean up engine caches
        import('../../engine/WebGPUEngine').then(({ engine }) => {
          engine.cleanupVideo(video);
        });
      }
      if (clipToRemove.source?.type === 'audio' && clipToRemove.source.audioElement) {
        const audio = clipToRemove.source.audioElement;
        // Revoke blob URL to free memory
        if (audio.src && audio.src.startsWith('blob:')) {
          URL.revokeObjectURL(audio.src);
        }
        // Pause and clear audio
        audio.pause();
        audio.src = '';
        audio.load();
      }

      // Also remove linked clip if exists
      if (clipToRemove.linkedClipId) {
        const linkedClip = clips.find(c => c.id === clipToRemove.linkedClipId);
        if (linkedClip?.source?.type === 'audio' && linkedClip.source.audioElement) {
          const audio = linkedClip.source.audioElement;
          if (audio.src && audio.src.startsWith('blob:')) {
            URL.revokeObjectURL(audio.src);
          }
          audio.pause();
          audio.src = '';
          audio.load();
        }
      }
    }

    // Remove from selection if selected
    const newSelectedIds = new Set(selectedClipIds);
    newSelectedIds.delete(id);
    if (clipToRemove?.linkedClipId) {
      newSelectedIds.delete(clipToRemove.linkedClipId);
    }

    set({
      clips: clips.filter(c => c.id !== id && c.id !== clipToRemove?.linkedClipId),
      selectedClipIds: newSelectedIds,
    });
    updateDuration();
    // Invalidate RAM preview cache - content changed
    invalidateCache();
  },

  moveClip: (id, newStartTime, newTrackId, skipLinked = false, skipGroup = false) => {
    const { clips, tracks, updateDuration, getSnappedPosition, findNonOverlappingPosition, invalidateCache } = get();
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

    // For linked clip (1:1 video-audio pair), also find non-overlapping position
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

    // For linked group (multicam), find all clips in the group
    const groupClips = !skipGroup && movingClip.linkedGroupId
      ? clips.filter(c => c.linkedGroupId === movingClip.linkedGroupId && c.id !== id)
      : [];

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
        // Also move linked clip (1:1 video-audio pair) - unless skipLinked is true
        if (!skipLinked && (c.id === movingClip.linkedClipId || c.linkedClipId === id)) {
          return {
            ...c,
            startTime: Math.max(0, linkedFinalTime),
            // Keep linked clip on its own track (don't change track)
          };
        }
        // Move group clips (multicam) - unless skipGroup is true (Alt+drag)
        if (!skipGroup && groupClips.some(gc => gc.id === c.id)) {
          const groupClipNewTime = findNonOverlappingPosition(
            c.id,
            c.startTime + timeDelta,
            c.trackId,
            c.duration
          );
          return {
            ...c,
            startTime: Math.max(0, groupClipNewTime),
          };
        }
        return c;
      }),
    });
    updateDuration();
    // Invalidate RAM preview cache - content changed
    invalidateCache();
  },

  trimClip: (id, inPoint, outPoint) => {
    const { clips, updateDuration, invalidateCache } = get();
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
    invalidateCache();
  },

  // Split a clip into two parts at the specified time
  splitClip: (clipId, splitTime) => {
    const { clips, updateDuration, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    // Validate split time is within clip bounds (not at edges)
    const clipEnd = clip.startTime + clip.duration;
    if (splitTime <= clip.startTime || splitTime >= clipEnd) {
      console.warn('[Timeline] Cannot split at edge or outside clip');
      return;
    }

    // Calculate the duration of the first part
    const firstPartDuration = splitTime - clip.startTime;
    const secondPartDuration = clip.duration - firstPartDuration;

    // Calculate the split point within the source media
    const splitInSource = clip.inPoint + firstPartDuration;

    const timestamp = Date.now();
    const randomSuffix = Math.random().toString(36).substr(2, 5);

    // Create the first clip (before split point)
    const firstClip: TimelineClip = {
      ...clip,
      id: `clip-${timestamp}-${randomSuffix}-a`,
      duration: firstPartDuration,
      outPoint: splitInSource,
      // Keep linkedClipId for now, will update after creating linked clips
      linkedClipId: undefined,
    };

    // Create the second clip (after split point)
    const secondClip: TimelineClip = {
      ...clip,
      id: `clip-${timestamp}-${randomSuffix}-b`,
      startTime: splitTime,
      duration: secondPartDuration,
      inPoint: splitInSource,
      linkedClipId: undefined,
    };

    // Build new clips array starting with non-affected clips
    const newClips: TimelineClip[] = clips.filter(c => c.id !== clipId && c.id !== clip.linkedClipId);

    // Handle linked clip (e.g., audio linked to video)
    if (clip.linkedClipId) {
      const linkedClip = clips.find(c => c.id === clip.linkedClipId);
      if (linkedClip) {
        // Create split versions of linked clip
        const linkedFirstClip: TimelineClip = {
          ...linkedClip,
          id: `clip-${timestamp}-${randomSuffix}-linked-a`,
          duration: firstPartDuration,
          outPoint: linkedClip.inPoint + firstPartDuration,
          linkedClipId: firstClip.id,
        };

        const linkedSecondClip: TimelineClip = {
          ...linkedClip,
          id: `clip-${timestamp}-${randomSuffix}-linked-b`,
          startTime: splitTime,
          duration: secondPartDuration,
          inPoint: linkedClip.inPoint + firstPartDuration,
          linkedClipId: secondClip.id,
        };

        // Update the main clips to reference their linked counterparts
        firstClip.linkedClipId = linkedFirstClip.id;
        secondClip.linkedClipId = linkedSecondClip.id;

        newClips.push(linkedFirstClip, linkedSecondClip);
      }
    }

    newClips.push(firstClip, secondClip);

    set({
      clips: newClips,
      selectedClipIds: new Set([secondClip.id]), // Select the second clip after split
    });

    updateDuration();
    invalidateCache();
    console.log(`[Timeline] Split clip "${clip.name}" at ${splitTime.toFixed(2)}s`);
  },

  // Split the clip under the playhead (or selected clip if playhead is on it)
  splitClipAtPlayhead: () => {
    const { clips, playheadPosition, selectedClipIds, splitClip } = get();
    // Get first selected clip ID for checking if it's at playhead
    const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;

    // Find clips at the current playhead position
    const clipsAtPlayhead = clips.filter(c =>
      playheadPosition > c.startTime &&
      playheadPosition < c.startTime + c.duration
    );

    if (clipsAtPlayhead.length === 0) {
      console.warn('[Timeline] No clip at playhead position');
      return;
    }

    // If selected clip is at playhead, split that one
    // Otherwise, split the topmost video clip at playhead
    let clipToSplit = clipsAtPlayhead.find(c => c.id === selectedClipId);
    if (!clipToSplit) {
      // Prefer video clips over audio clips
      clipToSplit = clipsAtPlayhead.find(c => c.source?.type === 'video' || c.source?.type === 'image');
      if (!clipToSplit) {
        clipToSplit = clipsAtPlayhead[0];
      }
    }

    splitClip(clipToSplit.id, playheadPosition);
  },

  updateClipTransform: (id, transform) => {
    const { clips, invalidateCache } = get();
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
    invalidateCache();
  },

  toggleClipReverse: (id) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c => {
        if (c.id !== id) return c;
        const newReversed = !c.reversed;
        // Reverse the thumbnails array when toggling
        const newThumbnails = c.thumbnails ? [...c.thumbnails].reverse() : c.thumbnails;
        return {
          ...c,
          reversed: newReversed,
          thumbnails: newThumbnails,
        };
      }),
    });
    // Invalidate cache - reversed playback affects rendered output
    invalidateCache();
  },

  // Clip effect actions
  addClipEffect: (clipId, effectType) => {
    const { clips, invalidateCache } = get();
    const effect: Effect = {
      id: `effect_${Date.now()}`,
      name: effectType,
      type: effectType as EffectType,
      enabled: true,
      params: getDefaultEffectParams(effectType),
    };

    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, effects: [...(c.effects || []), effect] } : c
      ),
    });
    invalidateCache();
  },

  removeClipEffect: (clipId, effectId) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, effects: c.effects.filter(e => e.id !== effectId) } : c
      ),
    });
    invalidateCache();
  },

  updateClipEffect: (clipId, effectId, params) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? {
              ...c,
              effects: c.effects.map(e =>
                e.id === effectId ? { ...e, params: { ...e.params, ...params } as Effect['params'] } : e
              ),
            }
          : c
      ),
    });
    invalidateCache();
  },

  // Create a linked group from selected clips (multicam sync)
  createLinkedGroup: (clipIds, offsets) => {
    const { clips, invalidateCache } = get();
    const groupId = `multicam-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;

    // Find the earliest start time among selected clips to use as anchor
    const selectedClips = clips.filter(c => clipIds.includes(c.id));
    if (selectedClips.length === 0) return;

    // Find clip with 0 offset (master) and use its start time as reference
    let masterStartTime = selectedClips[0].startTime;
    for (const clipId of clipIds) {
      const offset = offsets.get(clipId);
      if (offset === 0) {
        const masterClip = clips.find(c => c.id === clipId);
        if (masterClip) {
          masterStartTime = masterClip.startTime;
          break;
        }
      }
    }

    set({
      clips: clips.map(c => {
        if (!clipIds.includes(c.id)) return c;

        const offset = offsets.get(c.id) || 0;
        const offsetSeconds = offset / 1000; // Convert ms to seconds

        return {
          ...c,
          linkedGroupId: groupId,
          // Adjust start time based on audio sync offset
          startTime: Math.max(0, masterStartTime - offsetSeconds),
        };
      }),
    });

    invalidateCache();
    console.log(`[Multicam] Created linked group ${groupId} with ${clipIds.length} clips`);
  },

  // Remove a clip from its linked group (unlink multicam)
  unlinkGroup: (clipId) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip?.linkedGroupId) return;

    const groupId = clip.linkedGroupId;

    // Remove linkedGroupId from all clips in the group
    set({
      clips: clips.map(c =>
        c.linkedGroupId === groupId ? { ...c, linkedGroupId: undefined } : c
      ),
    });

    invalidateCache();
    console.log(`[Multicam] Unlinked group ${groupId}`);
  },
});
