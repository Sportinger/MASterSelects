// Clip-related actions slice - Coordinator
// Delegates to specialized modules in ./clip/ and ./helpers/
// Reduced from ~2031 LOC to ~650 LOC (68% reduction)

import type { TimelineClip, Effect, EffectType, TextClipProperties } from '../../types';
import type { ClipActions, SliceCreator, Composition } from './types';
import { DEFAULT_TRANSFORM, DEFAULT_TEXT_PROPERTIES, DEFAULT_TEXT_DURATION } from './constants';
import { generateWaveform, generateWaveformFromBuffer, getDefaultEffectParams } from './utils';
import { textRenderer } from '../../services/textRenderer';
import { googleFontsService } from '../../services/googleFontsService';
import { Logger } from '../../services/logger';

const log = Logger.create('ClipSlice');

// Import extracted modules
import { detectMediaType } from './helpers/mediaTypeHelpers';
import { createVideoClipPlaceholders, loadVideoMedia } from './clip/addVideoClip';
import { createAudioClipPlaceholder, loadAudioMedia } from './clip/addAudioClip';
import { createImageClipPlaceholder, loadImageMedia } from './clip/addImageClip';
import {
  createCompClipPlaceholder,
  loadNestedClips,
  generateCompThumbnails,
  createCompLinkedAudioClip,
} from './clip/addCompClip';
import { completeDownload as completeDownloadImpl } from './clip/completeDownload';
import {
  generateTextClipId,
  generateYouTubeClipId,
  generateEffectId,
  generateLinkedGroupId,
} from './helpers/idGenerator';
import { blobUrlManager } from './helpers/blobUrlManager';
import { updateClipById } from './helpers/clipStateHelpers';

export const createClipSlice: SliceCreator<ClipActions> = (set, get) => ({
  addClip: async (trackId, file, startTime, providedDuration, mediaFileId) => {
    const mediaType = detectMediaType(file);
    const estimatedDuration = providedDuration ?? 5;

    log.debug('Adding clip', { mediaType, file: file.name });

    // Validate track exists and matches media type
    const { tracks, clips, updateDuration, findAvailableAudioTrack, thumbnailsEnabled, waveformsEnabled, invalidateCache } = get();
    const targetTrack = tracks.find(t => t.id === trackId);
    if (!targetTrack) {
      log.warn('Track not found', { trackId });
      return;
    }

    if ((mediaType === 'video' || mediaType === 'image') && targetTrack.type !== 'video') {
      console.warn('[Timeline] Cannot add video/image to audio track');
      return;
    }
    if (mediaType === 'audio' && targetTrack.type !== 'audio') {
      console.warn('[Timeline] Cannot add audio to video track');
      return;
    }

    // Helper to update clip when loaded
    const updateClip = (id: string, updates: Partial<TimelineClip>) => {
      set({ clips: get().clips.map(c => c.id === id ? { ...c, ...updates } : c) });
      get().updateDuration();
    };
    const setClips = (updater: (clips: TimelineClip[]) => TimelineClip[]) => {
      set({ clips: updater(get().clips) });
    };

    // Handle video files
    if (mediaType === 'video') {
      const { videoClip, audioClip, audioClipId } = createVideoClipPlaceholders({
        trackId, file, startTime, estimatedDuration, mediaFileId, tracks, findAvailableAudioTrack,
      });

      set({ clips: [...clips, videoClip, ...(audioClip ? [audioClip] : [])] });
      updateDuration();

      await loadVideoMedia({
        clipId: videoClip.id,
        audioClipId,
        file,
        mediaFileId,
        thumbnailsEnabled,
        waveformsEnabled,
        updateClip,
        setClips,
      });

      invalidateCache();
      return;
    }

    // Handle audio files
    if (mediaType === 'audio') {
      const audioClip = createAudioClipPlaceholder({ trackId, file, startTime, estimatedDuration, mediaFileId });
      set({ clips: [...clips, audioClip] });
      updateDuration();

      await loadAudioMedia({
        clip: audioClip,
        file,
        mediaFileId,
        waveformsEnabled,
        updateClip,
      });

      invalidateCache();
      return;
    }

    // Handle image files
    if (mediaType === 'image') {
      const imageClip = createImageClipPlaceholder({ trackId, file, startTime, estimatedDuration });
      set({ clips: [...clips, imageClip] });
      updateDuration();

      await loadImageMedia({ clip: imageClip, updateClip });
      invalidateCache();
    }
  },

  // Add a composition as a clip (nested composition)
  addCompClip: async (trackId, composition: Composition, startTime) => {
    const { clips, updateDuration, findNonOverlappingPosition, thumbnailsEnabled, invalidateCache } = get();

    const compClip = createCompClipPlaceholder({ trackId, composition, startTime, findNonOverlappingPosition });
    set({ clips: [...clips, compClip] });
    updateDuration();

    // Load nested clips if timeline data exists
    if (composition.timelineData) {
      const nestedClips = await loadNestedClips({ compClipId: compClip.id, composition, get, set });
      const nestedTracks = composition.timelineData.tracks;

      set({
        clips: get().clips.map(c =>
          c.id === compClip.id ? { ...c, nestedClips, nestedTracks, isLoading: false } : c
        ),
      });

      // Generate thumbnails from first video
      generateCompThumbnails({
        clipId: compClip.id,
        nestedClips,
        compDuration: composition.timelineData?.duration ?? composition.duration,
        thumbnailsEnabled,
        get,
        set,
      });
    }

    // Create linked audio clip (always, even if no audio)
    await createCompLinkedAudioClip({
      compClipId: compClip.id,
      composition,
      compClipStartTime: compClip.startTime,
      compDuration: composition.timelineData?.duration ?? composition.duration,
      tracks: get().tracks,
      set,
      get,
    });

    invalidateCache();
  },

  removeClip: (id) => {
    const { clips, selectedClipIds, updateDuration, invalidateCache } = get();
    const clipToRemove = clips.find(c => c.id === id);

    if (clipToRemove) {
      // Clean up video resources
      if (clipToRemove.source?.type === 'video' && clipToRemove.source.videoElement) {
        const video = clipToRemove.source.videoElement;
        video.pause();
        video.src = '';
        video.load();
        import('../../engine/WebGPUEngine').then(({ engine }) => engine.cleanupVideo(video));
      }
      // Clean up audio resources
      if (clipToRemove.source?.type === 'audio' && clipToRemove.source.audioElement) {
        const audio = clipToRemove.source.audioElement;
        audio.pause();
        audio.src = '';
        audio.load();
      }

      // Revoke blob URLs via manager (handles all URL types for this clip)
      blobUrlManager.revokeAll(id);

      // Also cleanup linked clip
      if (clipToRemove.linkedClipId) {
        const linkedClip = clips.find(c => c.id === clipToRemove.linkedClipId);
        if (linkedClip?.source?.type === 'audio' && linkedClip.source.audioElement) {
          const audio = linkedClip.source.audioElement;
          audio.pause();
          audio.src = '';
          audio.load();
        }
        // Revoke linked clip's blob URLs
        blobUrlManager.revokeAll(clipToRemove.linkedClipId);
      }
    }

    const newSelectedIds = new Set(selectedClipIds);
    newSelectedIds.delete(id);
    if (clipToRemove?.linkedClipId) newSelectedIds.delete(clipToRemove.linkedClipId);

    set({
      clips: clips.filter(c => c.id !== id && c.id !== clipToRemove?.linkedClipId),
      selectedClipIds: newSelectedIds,
    });
    updateDuration();
    invalidateCache();
  },

  moveClip: (id, newStartTime, newTrackId, skipLinked = false, skipGroup = false) => {
    const { clips, tracks, updateDuration, getSnappedPosition, getPositionWithResistance, trimOverlappingClips, invalidateCache } = get();
    const movingClip = clips.find(c => c.id === id);
    if (!movingClip) return;

    const targetTrackId = newTrackId ?? movingClip.trackId;

    // Validate track type if changing tracks
    if (newTrackId && newTrackId !== movingClip.trackId) {
      const targetTrack = tracks.find(t => t.id === newTrackId);
      const sourceType = movingClip.source?.type;
      if (targetTrack && sourceType) {
        if ((sourceType === 'video' || sourceType === 'image') && targetTrack.type !== 'video') return;
        if (sourceType === 'audio' && targetTrack.type !== 'audio') return;
      }
    }

    const { startTime: snappedTime } = getSnappedPosition(id, newStartTime, targetTrackId);
    const { startTime: finalStartTime, forcingOverlap } = getPositionWithResistance(id, snappedTime, targetTrackId, movingClip.duration);
    const timeDelta = finalStartTime - movingClip.startTime;

    const linkedClip = clips.find(c => c.id === movingClip.linkedClipId || c.linkedClipId === id);
    let linkedFinalTime = linkedClip ? linkedClip.startTime + timeDelta : 0;
    let linkedForcingOverlap = false;
    if (linkedClip && !skipLinked) {
      const linkedResult = getPositionWithResistance(linkedClip.id, linkedClip.startTime + timeDelta, linkedClip.trackId, linkedClip.duration);
      linkedFinalTime = linkedResult.startTime;
      linkedForcingOverlap = linkedResult.forcingOverlap;
    }

    const groupClips = !skipGroup && movingClip.linkedGroupId
      ? clips.filter(c => c.linkedGroupId === movingClip.linkedGroupId && c.id !== id)
      : [];

    set({
      clips: clips.map(c => {
        if (c.id === id) return { ...c, startTime: Math.max(0, finalStartTime), trackId: targetTrackId };
        if (!skipLinked && (c.id === movingClip.linkedClipId || c.linkedClipId === id)) {
          return { ...c, startTime: Math.max(0, linkedFinalTime) };
        }
        if (!skipGroup && groupClips.some(gc => gc.id === c.id)) {
          const groupResult = getPositionWithResistance(c.id, c.startTime + timeDelta, c.trackId, c.duration);
          return { ...c, startTime: Math.max(0, groupResult.startTime) };
        }
        return c;
      }),
    });

    if (forcingOverlap) trimOverlappingClips(id, finalStartTime, targetTrackId, movingClip.duration);
    if (linkedForcingOverlap && linkedClip && !skipLinked) {
      trimOverlappingClips(linkedClip.id, linkedFinalTime, linkedClip.trackId, linkedClip.duration);
    }

    updateDuration();
    invalidateCache();
  },

  trimClip: (id, inPoint, outPoint) => {
    const { clips, updateDuration, invalidateCache } = get();
    set({
      clips: clips.map(c => {
        if (c.id !== id) return c;
        return { ...c, inPoint, outPoint, duration: outPoint - inPoint };
      }),
    });
    updateDuration();
    invalidateCache();
  },

  splitClip: (clipId, splitTime) => {
    const { clips, updateDuration, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    const clipEnd = clip.startTime + clip.duration;
    if (splitTime <= clip.startTime || splitTime >= clipEnd) {
      console.warn('[Timeline] Cannot split at edge or outside clip');
      return;
    }

    const firstPartDuration = splitTime - clip.startTime;
    const secondPartDuration = clip.duration - firstPartDuration;
    const splitInSource = clip.inPoint + firstPartDuration;

    const timestamp = Date.now();
    const randomSuffix = Math.random().toString(36).substr(2, 5);

    const firstClip: TimelineClip = {
      ...clip,
      id: `clip-${timestamp}-${randomSuffix}-a`,
      duration: firstPartDuration,
      outPoint: splitInSource,
      linkedClipId: undefined,
    };

    const secondClip: TimelineClip = {
      ...clip,
      id: `clip-${timestamp}-${randomSuffix}-b`,
      startTime: splitTime,
      duration: secondPartDuration,
      inPoint: splitInSource,
      linkedClipId: undefined,
    };

    const newClips: TimelineClip[] = clips.filter(c => c.id !== clipId && c.id !== clip.linkedClipId);

    if (clip.linkedClipId) {
      const linkedClip = clips.find(c => c.id === clip.linkedClipId);
      if (linkedClip) {
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
        firstClip.linkedClipId = linkedFirstClip.id;
        secondClip.linkedClipId = linkedSecondClip.id;
        newClips.push(linkedFirstClip, linkedSecondClip);
      }
    }

    newClips.push(firstClip, secondClip);
    set({ clips: newClips, selectedClipIds: new Set([secondClip.id]) });
    updateDuration();
    invalidateCache();
    console.log(`[Timeline] Split clip "${clip.name}" at ${splitTime.toFixed(2)}s`);
  },

  splitClipAtPlayhead: () => {
    const { clips, playheadPosition, selectedClipIds, splitClip } = get();
    const clipsAtPlayhead = clips.filter(c =>
      playheadPosition > c.startTime && playheadPosition < c.startTime + c.duration
    );

    if (clipsAtPlayhead.length === 0) {
      console.warn('[Timeline] No clip at playhead position');
      return;
    }

    let clipsToSplit = selectedClipIds.size > 0
      ? clipsAtPlayhead.filter(c => selectedClipIds.has(c.id))
      : clipsAtPlayhead;

    if (clipsToSplit.length === 0) clipsToSplit = clipsAtPlayhead;

    const linkedClipIds = new Set(clipsToSplit.map(c => c.linkedClipId).filter(Boolean));
    const clipsToSplitFiltered = clipsToSplit.filter(c => !linkedClipIds.has(c.id));

    for (const clip of clipsToSplitFiltered) {
      splitClip(clip.id, playheadPosition);
    }
  },

  updateClip: (id, updates) => {
    const { clips, updateDuration } = get();
    set({ clips: clips.map(c => c.id === id ? { ...c, ...updates } : c) });
    updateDuration();
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
            position: transform.position ? { ...c.transform.position, ...transform.position } : c.transform.position,
            scale: transform.scale ? { ...c.transform.scale, ...transform.scale } : c.transform.scale,
            rotation: transform.rotation ? { ...c.transform.rotation, ...transform.rotation } : c.transform.rotation,
          },
        };
      }),
    });
    invalidateCache();
  },

  // ========== TEXT CLIP ACTIONS ==========

  addTextClip: async (trackId, startTime, duration = DEFAULT_TEXT_DURATION) => {
    const { clips, tracks, updateDuration, invalidateCache } = get();
    const track = tracks.find(t => t.id === trackId);

    if (!track || track.type !== 'video') {
      console.warn('[Timeline] Text clips can only be added to video tracks');
      return null;
    }

    const clipId = generateTextClipId();
    await googleFontsService.loadFont(DEFAULT_TEXT_PROPERTIES.fontFamily, DEFAULT_TEXT_PROPERTIES.fontWeight);

    const canvas = textRenderer.createCanvas(1920, 1080);
    textRenderer.render(DEFAULT_TEXT_PROPERTIES, canvas);

    const textClip: TimelineClip = {
      id: clipId,
      trackId,
      name: 'Text',
      file: new File([], 'text-clip.txt', { type: 'text/plain' }),
      startTime,
      duration,
      inPoint: 0,
      outPoint: duration,
      source: { type: 'text', textCanvas: canvas, naturalDuration: duration },
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      textProperties: { ...DEFAULT_TEXT_PROPERTIES },
      isLoading: false,
    };

    set({ clips: [...clips, textClip] });
    updateDuration();
    invalidateCache();
    console.log(`[Timeline] Created text clip: ${clipId}`);
    return clipId;
  },

  updateTextProperties: (clipId, props) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c => {
        if (c.id !== clipId || !c.textProperties) return c;
        const newProps: TextClipProperties = { ...c.textProperties, ...props };

        if (props.fontFamily || props.fontWeight) {
          googleFontsService.loadFont(props.fontFamily || c.textProperties.fontFamily, props.fontWeight || c.textProperties.fontWeight);
        }

        const canvas = textRenderer.createCanvas(1920, 1080);
        textRenderer.render(newProps, canvas);

        return {
          ...c,
          textProperties: newProps,
          source: { ...c.source!, textCanvas: canvas },
          name: newProps.text.substring(0, 20) || 'Text',
        };
      }),
    });
    invalidateCache();
  },

  toggleClipReverse: (id) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c => {
        if (c.id !== id) return c;
        return {
          ...c,
          reversed: !c.reversed,
          thumbnails: c.thumbnails ? [...c.thumbnails].reverse() : c.thumbnails,
        };
      }),
    });
    invalidateCache();
  },

  // ========== EFFECT ACTIONS ==========

  addClipEffect: (clipId, effectType) => {
    const { clips, invalidateCache } = get();
    const effect: Effect = {
      id: generateEffectId(),
      name: effectType,
      type: effectType as EffectType,
      enabled: true,
      params: getDefaultEffectParams(effectType),
    };
    set({ clips: clips.map(c => c.id === clipId ? { ...c, effects: [...(c.effects || []), effect] } : c) });
    invalidateCache();
  },

  removeClipEffect: (clipId, effectId) => {
    const { clips, invalidateCache } = get();
    set({ clips: clips.map(c => c.id === clipId ? { ...c, effects: c.effects.filter(e => e.id !== effectId) } : c) });
    invalidateCache();
  },

  updateClipEffect: (clipId, effectId, params) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, effects: c.effects.map(e => e.id === effectId ? { ...e, params: { ...e.params, ...params } as Effect['params'] } : e) }
          : c
      ),
    });
    invalidateCache();
  },

  setClipEffectEnabled: (clipId, effectId, enabled) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, effects: c.effects.map(e => e.id === effectId ? { ...e, enabled } : e) }
          : c
      ),
    });
    invalidateCache();
  },

  // ========== MULTICAM / LINKED GROUP ACTIONS ==========

  createLinkedGroup: (clipIds, offsets) => {
    const { clips, invalidateCache } = get();
    const groupId = generateLinkedGroupId();
    const selectedClips = clips.filter(c => clipIds.includes(c.id));
    if (selectedClips.length === 0) return;

    let masterStartTime = selectedClips[0].startTime;
    for (const clipId of clipIds) {
      if (offsets.get(clipId) === 0) {
        const masterClip = clips.find(c => c.id === clipId);
        if (masterClip) { masterStartTime = masterClip.startTime; break; }
      }
    }

    set({
      clips: clips.map(c => {
        if (!clipIds.includes(c.id)) return c;
        const offset = offsets.get(c.id) || 0;
        return { ...c, linkedGroupId: groupId, startTime: Math.max(0, masterStartTime - offset / 1000) };
      }),
    });
    invalidateCache();
    console.log(`[Multicam] Created linked group ${groupId} with ${clipIds.length} clips`);
  },

  unlinkGroup: (clipId) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip?.linkedGroupId) return;

    set({ clips: clips.map(c => c.linkedGroupId === clip.linkedGroupId ? { ...c, linkedGroupId: undefined } : c) });
    invalidateCache();
    console.log(`[Multicam] Unlinked group ${clip.linkedGroupId}`);
  },

  // ========== WAVEFORM GENERATION ==========

  generateWaveformForClip: async (clipId: string) => {
    const { clips } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip || clip.waveformGenerating) return;

    set({ clips: updateClipById(get().clips, clipId, { waveformGenerating: true, waveformProgress: 0 }) });
    console.log(`[Waveform] Starting generation for ${clip.name}`);

    try {
      let waveform: number[];

      if (clip.isComposition && clip.compositionId) {
        const { compositionAudioMixer } = await import('../../services/compositionAudioMixer');
        const mixdownResult = await compositionAudioMixer.mixdownComposition(clip.compositionId);

        if (mixdownResult?.hasAudio) {
          waveform = mixdownResult.waveform;
          const mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
          set({
            clips: updateClipById(get().clips, clipId, {
              source: { type: 'audio' as const, audioElement: mixdownAudio, naturalDuration: mixdownResult.duration },
              mixdownBuffer: mixdownResult.buffer,
              hasMixdownAudio: true,
            }),
          });
        } else if (clip.mixdownBuffer) {
          waveform = generateWaveformFromBuffer(clip.mixdownBuffer, 50);
        } else {
          waveform = new Array(Math.max(1, Math.floor(clip.duration * 50))).fill(0);
        }
      } else if (!clip.file) {
        console.warn('[Waveform] No file found for clip:', clipId);
        set({ clips: updateClipById(get().clips, clipId, { waveformGenerating: false }) });
        return;
      } else {
        waveform = await generateWaveform(clip.file, 50, (progress, partialWaveform) => {
          set({ clips: updateClipById(get().clips, clipId, { waveformProgress: progress, waveform: partialWaveform }) });
        });
      }

      console.log(`[Waveform] Complete: ${waveform.length} samples for ${clip.name}`);
      set({ clips: updateClipById(get().clips, clipId, { waveform, waveformGenerating: false, waveformProgress: 100 }) });
    } catch (e) {
      console.error('[Waveform] Failed:', e);
      set({ clips: updateClipById(get().clips, clipId, { waveformGenerating: false }) });
    }
  },

  // ========== PARENTING (PICK WHIP) ==========

  setClipParent: (clipId: string, parentClipId: string | null) => {
    const { clips } = get();
    if (parentClipId === clipId) {
      console.warn('[Parenting] Cannot parent clip to itself');
      return;
    }

    if (parentClipId) {
      const wouldCreateCycle = (checkId: string): boolean => {
        const check = clips.find(c => c.id === checkId);
        if (!check?.parentClipId) return false;
        if (check.parentClipId === clipId) return true;
        return wouldCreateCycle(check.parentClipId);
      };
      if (wouldCreateCycle(parentClipId)) {
        console.warn('[Parenting] Cannot create circular parent reference');
        return;
      }
    }

    set({ clips: clips.map(c => c.id === clipId ? { ...c, parentClipId: parentClipId || undefined } : c) });
    console.log(`[Parenting] Set parent of ${clipId} to ${parentClipId || 'none'}`);
  },

  getClipChildren: (clipId: string) => {
    return get().clips.filter(c => c.parentClipId === clipId);
  },

  setClipPreservesPitch: (clipId: string, preservesPitch: boolean) => {
    set({ clips: updateClipById(get().clips, clipId, { preservesPitch }) });
  },

  // ========== YOUTUBE PENDING DOWNLOAD ==========

  addPendingDownloadClip: (trackId, startTime, videoId, title, thumbnail, estimatedDuration = 30) => {
    const { clips, tracks, updateDuration, findNonOverlappingPosition } = get();
    const track = tracks.find(t => t.id === trackId);
    if (!track || track.type !== 'video') {
      console.warn('[Timeline] Pending download clips can only be added to video tracks');
      return '';
    }

    const clipId = generateYouTubeClipId();
    const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, estimatedDuration);

    const pendingClip: TimelineClip = {
      id: clipId,
      trackId,
      name: title,
      file: new File([], `${title}.mp4`, { type: 'video/mp4' }),
      startTime: finalStartTime,
      duration: estimatedDuration,
      inPoint: 0,
      outPoint: estimatedDuration,
      source: null,
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      isLoading: false,
      isPendingDownload: true,
      downloadProgress: 0,
      youtubeVideoId: videoId,
      youtubeThumbnail: thumbnail,
    };

    set({ clips: [...clips, pendingClip] });
    updateDuration();
    console.log(`[Timeline] Added pending download clip: ${clipId}`);
    return clipId;
  },

  updateDownloadProgress: (clipId, progress) => {
    set({ clips: updateClipById(get().clips, clipId, { downloadProgress: progress }) });
  },

  completeDownload: async (clipId, file) => {
    await completeDownloadImpl({
      clipId,
      file,
      clips: get().clips,
      waveformsEnabled: get().waveformsEnabled,
      findAvailableAudioTrack: get().findAvailableAudioTrack,
      updateDuration: get().updateDuration,
      invalidateCache: get().invalidateCache,
      set,
      get,
    });
  },

  setDownloadError: (clipId, error) => {
    set({ clips: updateClipById(get().clips, clipId, { downloadError: error, isPendingDownload: false }) });
  },
});
