// Clipboard-related actions slice for copy/paste functionality

import type { ClipboardActions, SliceCreator, ClipboardClipData, ClipboardKeyframeData, Keyframe } from './types';
import type { TimelineClip, EasingType } from '../../types';
import { Logger } from '../../services/logger';
import { captureSnapshot } from '../historyStore';

const log = Logger.create('Clipboard');

export const createClipboardSlice: SliceCreator<ClipboardActions> = (set, get) => ({
  copyClips: () => {
    const { clips, selectedClipIds, clipKeyframes, tracks } = get();

    if (selectedClipIds.size === 0) {
      log.debug('No clips selected to copy');
      return;
    }

    // Get all selected clips
    const selectedClips = clips.filter(c => selectedClipIds.has(c.id));

    // Also include linked audio clips if a video clip is selected
    const linkedClipIds = new Set<string>();
    selectedClips.forEach(clip => {
      if (clip.linkedClipId) {
        linkedClipIds.add(clip.linkedClipId);
      }
    });

    // Add linked clips that aren't already selected
    const linkedClips = clips.filter(c => linkedClipIds.has(c.id) && !selectedClipIds.has(c.id));
    const allClipsToCopy = [...selectedClips, ...linkedClips];

    // Convert to clipboard format (serializable, without DOM elements)
    const clipboardData: ClipboardClipData[] = allClipsToCopy.map(clip => {
      // Get track type for this clip
      const track = tracks.find(t => t.id === clip.trackId);
      const trackType = track?.type || 'video';

      // Get keyframes for this clip
      const keyframes = clipKeyframes.get(clip.id) || [];

      return {
        id: clip.id,
        trackId: clip.trackId,
        trackType,
        name: clip.name,
        mediaFileId: clip.source?.mediaFileId,
        startTime: clip.startTime,
        duration: clip.duration,
        inPoint: clip.inPoint,
        outPoint: clip.outPoint,
        sourceType: clip.source?.type || 'video',
        naturalDuration: clip.source?.naturalDuration,
        transform: { ...clip.transform },
        effects: clip.effects.map(e => ({ ...e, params: { ...e.params } })),
        masks: clip.masks?.map(m => ({
          ...m,
          vertices: m.vertices.map(v => ({ ...v })),
        })),
        keyframes: keyframes.length > 0 ? keyframes.map(k => ({ ...k })) : undefined,
        linkedClipId: clip.linkedClipId,
        reversed: clip.reversed,
        speed: clip.speed,
        preservesPitch: clip.preservesPitch,
        textProperties: clip.textProperties ? { ...clip.textProperties } : undefined,
        solidColor: clip.source?.type === 'solid' ? (clip.solidColor || clip.name.replace('Solid ', '')) : undefined,
        // Visual data - reuse existing thumbnails and waveforms
        thumbnails: clip.thumbnails ? [...clip.thumbnails] : undefined,
        waveform: clip.waveform ? [...clip.waveform] : undefined,
        isComposition: clip.isComposition,
        compositionId: clip.compositionId,
      };
    });

    set({ clipboardData });
    log.info('Copied clips', { count: clipboardData.length, ids: clipboardData.map(c => c.id) });
  },

  pasteClips: () => {
    const { clipboardData, playheadPosition, tracks, clips, clipKeyframes, updateDuration, invalidateCache } = get();

    if (!clipboardData || clipboardData.length === 0) {
      log.debug('No clipboard data to paste');
      return;
    }

    // Capture snapshot for undo before making changes
    captureSnapshot('Paste clips');

    const timestamp = Date.now();
    const randomSuffix = () => Math.random().toString(36).substr(2, 5);

    // Create ID mapping for linked clips (old ID -> new ID)
    const idMapping = new Map<string, string>();
    clipboardData.forEach(clipData => {
      idMapping.set(clipData.id, `clip-${timestamp}-${randomSuffix()}`);
    });

    // Find the earliest clip in clipboard to calculate offset
    const earliestStartTime = Math.min(...clipboardData.map(c => c.startTime));
    const timeOffset = playheadPosition - earliestStartTime;

    const newClips: TimelineClip[] = [];
    const newKeyframes = new Map<string, Keyframe[]>(clipKeyframes);

    for (const clipData of clipboardData) {
      const newId = idMapping.get(clipData.id)!;
      const newStartTime = clipData.startTime + timeOffset;

      // Find a track of the same type as the original
      let targetTrackId = clipData.trackId;
      const originalTrack = tracks.find(t => t.id === clipData.trackId);

      if (!originalTrack) {
        // Original track doesn't exist, find another track of same type
        const sameTypeTrack = tracks.find(t => t.type === clipData.trackType);
        if (sameTypeTrack) {
          targetTrackId = sameTypeTrack.id;
        } else {
          log.warn('No suitable track found for clip', { clipName: clipData.name, trackType: clipData.trackType });
          continue;
        }
      }

      // Handle linked clip ID mapping
      let newLinkedClipId: string | undefined;
      if (clipData.linkedClipId) {
        newLinkedClipId = idMapping.get(clipData.linkedClipId);
      }

      // Create the new clip
      // Note: source will be null initially - the clip needs to reload media
      // For now we create a placeholder that references the mediaFileId
      const newClip: TimelineClip = {
        id: newId,
        trackId: targetTrackId,
        name: clipData.name,
        file: new File([], clipData.name), // Placeholder file
        startTime: Math.max(0, newStartTime),
        duration: clipData.duration,
        inPoint: clipData.inPoint,
        outPoint: clipData.outPoint,
        source: clipData.mediaFileId ? {
          type: clipData.sourceType,
          mediaFileId: clipData.mediaFileId,
          naturalDuration: clipData.naturalDuration,
        } : clipData.solidColor ? {
          type: 'solid' as const,
          naturalDuration: clipData.duration,
        } : null,
        transform: {
          ...clipData.transform,
          position: { ...clipData.transform.position },
          scale: { ...clipData.transform.scale },
          rotation: { ...clipData.transform.rotation },
        },
        effects: clipData.effects.map(e => ({
          ...e,
          id: `effect-${timestamp}-${randomSuffix()}`, // New effect IDs
          params: { ...e.params },
        })),
        masks: clipData.masks?.map(m => ({
          ...m,
          id: `mask-${timestamp}-${randomSuffix()}`, // New mask IDs
          vertices: m.vertices.map(v => ({
            ...v,
            id: `vertex-${timestamp}-${randomSuffix()}`, // New vertex IDs
          })),
        })),
        linkedClipId: newLinkedClipId,
        reversed: clipData.reversed,
        speed: clipData.speed,
        preservesPitch: clipData.preservesPitch,
        textProperties: clipData.textProperties ? { ...clipData.textProperties } : undefined,
        solidColor: clipData.solidColor,
        // Reuse existing thumbnails and waveforms from copied clip
        thumbnails: clipData.thumbnails ? [...clipData.thumbnails] : undefined,
        waveform: clipData.waveform ? [...clipData.waveform] : undefined,
        isComposition: clipData.isComposition,
        compositionId: clipData.compositionId,
        isLoading: true, // Will need to reload media
        needsReload: !clipData.textProperties && !clipData.solidColor, // Text/solid clips don't need reload
      };

      newClips.push(newClip);

      // Copy keyframes with new IDs and mapped clipId
      if (clipData.keyframes && clipData.keyframes.length > 0) {
        const newClipKeyframes = clipData.keyframes.map(kf => ({
          ...kf,
          id: `kf_${timestamp}_${randomSuffix()}`,
          clipId: newId,
        }));
        newKeyframes.set(newId, newClipKeyframes);
      }
    }

    if (newClips.length === 0) {
      log.warn('No clips could be pasted');
      return;
    }

    // Add new clips and keyframes to state
    set({
      clips: [...clips, ...newClips],
      clipKeyframes: newKeyframes,
      selectedClipIds: new Set(newClips.map(c => c.id)),
    });

    updateDuration();
    invalidateCache();

    log.info('Pasted clips', { count: newClips.length, ids: newClips.map(c => c.id) });

    // Reload media for pasted clips asynchronously
    import('../mediaStore').then(({ useMediaStore }) => {
      const mediaStore = useMediaStore.getState();

      for (const newClip of newClips) {
        // Skip text clips - they need special handling
        if (newClip.textProperties) {
          // Regenerate text canvas
          Promise.all([
            import('../../services/textRenderer'),
            import('../../services/googleFontsService'),
          ]).then(async ([{ textRenderer }, { googleFontsService }]) => {
            await googleFontsService.loadFont(
              newClip.textProperties!.fontFamily,
              newClip.textProperties!.fontWeight
            );
            const textCanvas = textRenderer.render(newClip.textProperties!);

            set(state => ({
              clips: state.clips.map(c =>
                c.id === newClip.id
                  ? {
                      ...c,
                      source: {
                        type: 'text' as const,
                        textCanvas,
                        naturalDuration: c.duration,
                      },
                      isLoading: false,
                      needsReload: false,
                    }
                  : c
              ),
            }));
          });
          continue;
        }

        // Handle solid clips - regenerate canvas
        if (newClip.source?.type === 'solid') {
          const originalClipData = clipboardData.find(cd => idMapping.get(cd.id) === newClip.id);
          const color = originalClipData?.solidColor || '#ffffff';
          const canvas = document.createElement('canvas');
          canvas.width = 1920;
          canvas.height = 1080;
          const ctx = canvas.getContext('2d')!;
          ctx.fillStyle = color;
          ctx.fillRect(0, 0, 1920, 1080);

          set(state => ({
            clips: state.clips.map(c =>
              c.id === newClip.id
                ? {
                    ...c,
                    source: {
                      type: 'solid' as const,
                      textCanvas: canvas,
                      naturalDuration: c.duration,
                    },
                    isLoading: false,
                    needsReload: false,
                  }
                : c
            ),
          }));
          continue;
        }

        // Skip composition clips - they reference compositions, not media files
        if (newClip.isComposition) {
          // Composition clips need their nested content loaded
          // For now just mark as not loading - the rendering will handle it
          set(state => ({
            clips: state.clips.map(c =>
              c.id === newClip.id
                ? { ...c, isLoading: false }
                : c
            ),
          }));
          continue;
        }

        const mediaFileId = newClip.source?.mediaFileId;
        if (!mediaFileId) continue;

        const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
        if (!mediaFile?.file) {
          log.warn('Media file not found for pasted clip', { clipId: newClip.id, mediaFileId });
          continue;
        }

        // Load media based on type
        const fileUrl = URL.createObjectURL(mediaFile.file);
        const sourceType = newClip.source?.type;

        if (sourceType === 'video') {
          const video = document.createElement('video');
          video.src = fileUrl;
          video.muted = true;
          video.playsInline = true;
          video.preload = 'auto';
          video.crossOrigin = 'anonymous';

          video.addEventListener('canplaythrough', async () => {
            set(state => ({
              clips: state.clips.map(c =>
                c.id === newClip.id
                  ? {
                      ...c,
                      file: mediaFile.file!,
                      source: {
                        type: 'video' as const,
                        videoElement: video,
                        naturalDuration: video.duration,
                        mediaFileId,
                      },
                      isLoading: false,
                      needsReload: false,
                    }
                  : c
              ),
            }));

            // Try to initialize WebCodecsPlayer
            const hasWebCodecs = 'VideoDecoder' in window && 'VideoFrame' in window;
            if (hasWebCodecs) {
              try {
                const { WebCodecsPlayer } = await import('../../engine/WebCodecsPlayer');
                const webCodecsPlayer = new WebCodecsPlayer({
                  loop: false,
                  useSimpleMode: true,
                  onError: (error) => {
                    log.warn('WebCodecs error', { error: error.message });
                  },
                });
                webCodecsPlayer.attachToVideoElement(video);

                set(state => ({
                  clips: state.clips.map(c =>
                    c.id === newClip.id && c.source?.type === 'video'
                      ? {
                          ...c,
                          source: { ...c.source, webCodecsPlayer },
                        }
                      : c
                  ),
                }));
              } catch (err) {
                log.warn('WebCodecsPlayer init failed for pasted clip', err);
              }
            }
          }, { once: true });
        } else if (sourceType === 'audio') {
          const audio = document.createElement('audio');
          audio.src = fileUrl;
          audio.preload = 'auto';

          audio.addEventListener('canplaythrough', () => {
            set(state => ({
              clips: state.clips.map(c =>
                c.id === newClip.id
                  ? {
                      ...c,
                      file: mediaFile.file!,
                      source: {
                        type: 'audio' as const,
                        audioElement: audio,
                        naturalDuration: audio.duration,
                        mediaFileId,
                      },
                      isLoading: false,
                      needsReload: false,
                    }
                  : c
              ),
            }));
          }, { once: true });
        } else if (sourceType === 'image') {
          const img = new Image();
          img.src = fileUrl;

          img.addEventListener('load', () => {
            set(state => ({
              clips: state.clips.map(c =>
                c.id === newClip.id
                  ? {
                      ...c,
                      file: mediaFile.file!,
                      source: {
                        type: 'image' as const,
                        imageElement: img,
                      },
                      isLoading: false,
                      needsReload: false,
                    }
                  : c
              ),
            }));
          }, { once: true });
        }
      }
    });
  },

  hasClipboardData: () => {
    const { clipboardData } = get();
    return clipboardData !== null && clipboardData.length > 0;
  },

  copyKeyframes: () => {
    const { selectedKeyframeIds, clipKeyframes } = get();

    if (selectedKeyframeIds.size === 0) {
      log.debug('No keyframes selected to copy');
      return;
    }

    // Collect all selected keyframes
    const selectedKfs: Keyframe[] = [];
    clipKeyframes.forEach((keyframes) => {
      keyframes.forEach(kf => {
        if (selectedKeyframeIds.has(kf.id)) {
          selectedKfs.push(kf);
        }
      });
    });

    if (selectedKfs.length === 0) return;

    // Find earliest time to normalize (so pasting is relative to playhead)
    const earliestTime = Math.min(...selectedKfs.map(kf => kf.time));

    const clipboardKeyframes: ClipboardKeyframeData[] = selectedKfs.map(kf => ({
      clipId: kf.clipId,
      property: kf.property,
      time: kf.time - earliestTime,
      value: kf.value,
      easing: kf.easing as EasingType,
      handleIn: kf.handleIn ? { ...kf.handleIn } : undefined,
      handleOut: kf.handleOut ? { ...kf.handleOut } : undefined,
    }));

    set({ clipboardKeyframes });
    log.info('Copied keyframes', { count: clipboardKeyframes.length });
  },

  pasteKeyframes: () => {
    const { clipboardKeyframes, playheadPosition, clips, selectedClipIds, clipKeyframes, invalidateCache, pasteClips } = get();

    if (!clipboardKeyframes || clipboardKeyframes.length === 0) {
      // Fall through to clip paste
      pasteClips();
      return;
    }

    // Determine target clip: use selected clip, or fall back to the original clip
    const targetClipId = selectedClipIds.size === 1
      ? [...selectedClipIds][0]
      : clipboardKeyframes[0].clipId;

    const targetClip = clips.find(c => c.id === targetClipId);
    if (!targetClip) {
      log.warn('No target clip found for keyframe paste');
      return;
    }

    captureSnapshot('Paste keyframes');

    const clipLocalTime = playheadPosition - targetClip.startTime;
    const newMap = new Map(clipKeyframes);
    const existingKeyframes = newMap.get(targetClipId) || [];
    const newKeyframes = [...existingKeyframes];

    const timestamp = Date.now();
    const randomSuffix = () => Math.random().toString(36).substr(2, 5);

    for (const kfData of clipboardKeyframes) {
      const newTime = Math.max(0, Math.min(targetClip.duration, clipLocalTime + kfData.time));

      const newKf: Keyframe = {
        id: `kf_${timestamp}_${randomSuffix()}`,
        clipId: targetClipId,
        time: newTime,
        property: kfData.property,
        value: kfData.value,
        easing: kfData.easing,
        handleIn: kfData.handleIn ? { ...kfData.handleIn } : undefined,
        handleOut: kfData.handleOut ? { ...kfData.handleOut } : undefined,
      };

      newKeyframes.push(newKf);
    }

    // Sort by time
    newKeyframes.sort((a, b) => a.time - b.time);
    newMap.set(targetClipId, newKeyframes);

    set({ clipKeyframes: newMap });
    invalidateCache();
    log.info('Pasted keyframes', { count: clipboardKeyframes.length, targetClipId });
  },
});
