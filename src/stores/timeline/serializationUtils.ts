// Timeline serialization utilities - save, load, clear
// Extracted from index.ts for maintainability

import type { SliceCreator, TimelineClip, TimelineUtils, Keyframe, CompositionTimelineData } from './types';
import type { SerializableClip, ClipAnalysis, FrameAnalysisData } from '../../types';
import { DEFAULT_TRACKS } from './constants';
import { useMediaStore } from '../mediaStore';
import { calculateNestedClipBoundaries, buildClipSegments } from './clip/addCompClip';
import { projectFileService } from '../../services/projectFileService';
import { Logger } from '../../services/logger';
import { engine } from '../../engine/WebGPUEngine';

const log = Logger.create('Timeline');

type SerializationUtils = Pick<TimelineUtils, 'getSerializableState' | 'loadState' | 'clearTimeline'>;

export const createSerializationUtils: SliceCreator<SerializationUtils> = (set, get) => ({
  // Get serializable timeline state for saving to composition
  getSerializableState: (): CompositionTimelineData => {
    const { tracks, clips, playheadPosition, duration, durationLocked, zoom, scrollX, inPoint, outPoint, loopPlayback, clipKeyframes, markers } = get();

    // Convert clips to serializable format (without DOM elements)
    const mediaStore = useMediaStore.getState();
    const serializableClips: SerializableClip[] = clips.map(clip => {
      // Use existing mediaFileId if available, otherwise lookup by name
      let resolvedMediaFileId = clip.source?.mediaFileId || '';

      if (!resolvedMediaFileId && !clip.isComposition) {
        // Fallback: Find the mediaFile ID by matching the file name in mediaStore
        // For linked audio clips (name ends with "(Audio)"), strip the suffix to find the video file
        let lookupName = clip.name;
        if (clip.linkedClipId && clip.source?.type === 'audio' && lookupName.endsWith(' (Audio)')) {
          lookupName = lookupName.replace(' (Audio)', '');
        }
        const mediaFile = mediaStore.files.find(f => f.name === lookupName);
        resolvedMediaFileId = mediaFile?.id || '';
      }

      // Get keyframes for this clip
      const keyframes = clipKeyframes.get(clip.id) || [];

      return {
        id: clip.id,
        trackId: clip.trackId,
        name: clip.name,
        mediaFileId: clip.isComposition ? '' : resolvedMediaFileId, // Comp clips don't have media files
        startTime: clip.startTime,
        duration: clip.duration,
        inPoint: clip.inPoint,
        outPoint: clip.outPoint,
        sourceType: clip.source?.type || 'video',
        naturalDuration: clip.source?.naturalDuration,
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
        // Text clip support
        textProperties: clip.textProperties,
        // Solid clip support
        solidColor: clip.source?.type === 'solid' ? (clip.solidColor || clip.name.replace('Solid ', '')) : undefined,
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
      markers: markers.length > 0 ? markers : undefined,  // Only save if there are markers
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
        playbackSpeed: 1,
        selectedClipIds: new Set(),
        primarySelectedClipId: null,
        markers: [],
      });
      return;
    }

    // Restore tracks and basic state
    // Increment animation key to trigger entrance animations on clips
    const { clipEntranceAnimationKey } = get();
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
      primarySelectedClipId: null,
      // Clear keyframe state
      clipKeyframes: new Map<string, Keyframe[]>(),
      keyframeRecordingEnabled: new Set<string>(),
      expandedTracks: new Set<string>(data.tracks.filter(t => t.type === 'video').map(t => t.id)),
      expandedTrackPropertyGroups: new Map<string, Set<string>>(),
      selectedKeyframeIds: new Set<string>(),
      expandedCurveProperties: new Map<string, Set<import('../../types').AnimatableProperty>>(),
      // Restore markers
      markers: data.markers || [],
      // Increment animation key for clip entrance animations
      clipEntranceAnimationKey: clipEntranceAnimationKey + 1,
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
          // Check if this is a composition AUDIO clip (linked audio for nested comp)
          if (serializedClip.sourceType === 'audio') {
            // Create composition audio clip - will regenerate mixdown
            const compAudioClip: TimelineClip = {
              id: serializedClip.id,
              trackId: serializedClip.trackId,
              name: serializedClip.name,
              file: new File([], serializedClip.name),
              startTime: serializedClip.startTime,
              duration: serializedClip.duration,
              inPoint: serializedClip.inPoint,
              outPoint: serializedClip.outPoint,
              source: {
                type: 'audio',
                audioElement: document.createElement('audio'),
                naturalDuration: serializedClip.duration,
              },
              linkedClipId: serializedClip.linkedClipId,
              waveform: serializedClip.waveform || [],
              transform: serializedClip.transform,
              effects: serializedClip.effects || [],
              isLoading: false,
              isComposition: true,
              compositionId: serializedClip.compositionId,
            };

            // Add clip to state
            set(state => ({
              clips: [...state.clips, compAudioClip],
            }));

            // Regenerate audio mixdown in background
            import('../../services/compositionAudioMixer').then(async ({ compositionAudioMixer }) => {
              try {
                log.debug('Regenerating audio mixdown', { composition: composition.name });
                const mixdownResult = await compositionAudioMixer.mixdownComposition(composition.id);

                if (mixdownResult && mixdownResult.hasAudio) {
                  const mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
                  mixdownAudio.preload = 'auto';

                  set(state => ({
                    clips: state.clips.map(c =>
                      c.id === compAudioClip.id
                        ? {
                            ...c,
                            source: {
                              type: 'audio' as const,
                              audioElement: mixdownAudio,
                              naturalDuration: mixdownResult.duration,
                            },
                            waveform: mixdownResult.waveform,
                            mixdownBuffer: mixdownResult.buffer,
                            hasMixdownAudio: true,
                          }
                        : c
                    ),
                  }));
                  log.debug('Audio mixdown restored', { composition: composition.name });
                } else {
                  // No audio - generate flat waveform
                  const flatWaveform = new Array(Math.max(1, Math.floor(serializedClip.duration * 50))).fill(0);
                  set(state => ({
                    clips: state.clips.map(c =>
                      c.id === compAudioClip.id
                        ? { ...c, waveform: flatWaveform, hasMixdownAudio: false }
                        : c
                    ),
                  }));
                }
              } catch (e) {
                log.error('Failed to regenerate audio mixdown', e);
              }
            });

            continue;
          }

          // Create comp VIDEO clip manually to restore specific settings
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
            linkedClipId: serializedClip.linkedClipId,
            transform: serializedClip.transform,
            effects: serializedClip.effects || [],
            masks: serializedClip.masks || [],  // Restore masks for composition clips
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

            log.info('Loading nested clips for comp', {
              compClipId: compClip.id,
              compositionId: composition.id,
              compositionName: composition.name,
              nestedClipCount: composition.timelineData.clips.length,
              nestedClips: composition.timelineData.clips.map((c: SerializableClip) => ({
                id: c.id,
                name: c.name,
                trackId: c.trackId,
                mediaFileId: c.mediaFileId,
                sourceType: c.sourceType,
              })),
              availableMediaFiles: mediaStore.files.map(f => ({ id: f.id, name: f.name, hasFile: !!f.file })),
            });

            for (const nestedSerializedClip of composition.timelineData.clips) {
              const nestedMediaFile = mediaStore.files.find(f => f.id === nestedSerializedClip.mediaFileId);
              const hasFile = !!(nestedMediaFile?.file);

              if (!nestedMediaFile) {
                log.warn('Skipping nested clip - media file entry not found', {
                  clipName: nestedSerializedClip.name,
                  trackId: nestedSerializedClip.trackId,
                  mediaFileId: nestedSerializedClip.mediaFileId,
                });
                continue;
              }

              // Create the nested clip - even if file is missing (will need reload)
              const nestedClip: TimelineClip = {
                id: `nested-${compClip.id}-${nestedSerializedClip.id}`,
                trackId: nestedSerializedClip.trackId,
                name: nestedSerializedClip.name,
                file: nestedMediaFile.file || new File([], nestedSerializedClip.name),
                startTime: nestedSerializedClip.startTime,
                duration: nestedSerializedClip.duration,
                inPoint: nestedSerializedClip.inPoint,
                outPoint: nestedSerializedClip.outPoint,
                source: hasFile ? null : {
                  type: nestedSerializedClip.sourceType || 'video',
                  naturalDuration: nestedSerializedClip.naturalDuration || nestedSerializedClip.duration,
                  mediaFileId: nestedSerializedClip.mediaFileId,
                },
                transform: nestedSerializedClip.transform,
                effects: nestedSerializedClip.effects || [],
                masks: nestedSerializedClip.masks || [],
                isLoading: hasFile,
                needsReload: !hasFile,
              };

              nestedClips.push(nestedClip);

              // Only load media element if file is available
              if (!hasFile) {
                log.warn('Nested clip needs reload - file not available', {
                  clipName: nestedSerializedClip.name,
                  trackId: nestedSerializedClip.trackId,
                  mediaFileId: nestedSerializedClip.mediaFileId,
                });
                continue;
              }

              // Load media element
              const nestedType = nestedSerializedClip.sourceType;
              const nestedFileRef = nestedMediaFile.file!;
              const nestedFileUrl = URL.createObjectURL(nestedFileRef);

              if (nestedType === 'video') {
                const video = document.createElement('video');
                video.src = nestedFileUrl;
                video.muted = true;
                video.playsInline = true;
                video.preload = 'auto';
                video.crossOrigin = 'anonymous';

                // Force browser to start loading
                video.load();

                // Pre-cache frame for immediate scrubbing (needs readyState >= 2, so use canplaythrough)
                video.addEventListener('canplaythrough', () => {
                  engine.preCacheVideoFrame(video);
                }, { once: true });

                video.addEventListener('loadedmetadata', async () => {
                  // Set up basic video source first
                  const videoSource: TimelineClip['source'] = {
                    type: 'video',
                    videoElement: video,
                    naturalDuration: video.duration,
                  };
                  nestedClip.source = videoSource;
                  nestedClip.isLoading = false;

                  // Initialize WebCodecsPlayer for hardware-accelerated decoding
                  const hasWebCodecs = 'VideoDecoder' in window && 'VideoFrame' in window;
                  if (hasWebCodecs) {
                    try {
                      const { WebCodecsPlayer } = await import('../../engine/WebCodecsPlayer');
                      log.debug('Initializing WebCodecsPlayer for nested comp', { file: nestedFileRef.name });

                      const webCodecsPlayer = new WebCodecsPlayer({
                        loop: false,
                        useSimpleMode: true,
                        onError: (error) => {
                          log.warn('WebCodecs error in nested comp', { error: error.message });
                        },
                      });

                      webCodecsPlayer.attachToVideoElement(video);
                      log.debug('WebCodecsPlayer ready for nested comp', { file: nestedFileRef.name });

                      // Update nested clip source with webCodecsPlayer
                      nestedClip.source = {
                        ...nestedClip.source,
                        webCodecsPlayer,
                      };
                    } catch (err) {
                      log.warn('WebCodecsPlayer init failed in nested comp', err);
                    }
                  }

                  log.info('Nested video loaded', {
                    nestedClipId: nestedClip.id,
                    nestedClipName: nestedClip.name,
                    compClipId: compClip.id,
                    hasSource: !!nestedClip.source,
                    hasVideoElement: !!nestedClip.source?.videoElement,
                    readyState: video.readyState,
                  });

                  // Properly update state with the new nested clip source
                  // This ensures React/Zustand detects the change
                  set(state => ({
                    clips: state.clips.map(c => {
                      if (c.id !== compClip.id || !c.nestedClips) return c;
                      return {
                        ...c,
                        nestedClips: c.nestedClips.map(nc =>
                          nc.id === nestedClip.id
                            ? { ...nc, source: nestedClip.source, isLoading: false }
                            : nc
                        ),
                      };
                    }),
                  }));
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

                  log.info('Nested image loaded', {
                    nestedClipId: nestedClip.id,
                    nestedClipName: nestedClip.name,
                    compClipId: compClip.id,
                  });

                  // Properly update state with the new nested clip source
                  set(state => ({
                    clips: state.clips.map(c => {
                      if (c.id !== compClip.id || !c.nestedClips) return c;
                      return {
                        ...c,
                        nestedClips: c.nestedClips.map(nc =>
                          nc.id === nestedClip.id
                            ? { ...nc, source: nestedClip.source, isLoading: false }
                            : nc
                        ),
                      };
                    }),
                  }));
                }, { once: true });
              }
            }

            // Calculate clip boundaries for visual markers and thumbnail alignment
            const compDuration = composition.timelineData?.duration ?? composition.duration;
            const boundaries = calculateNestedClipBoundaries(composition.timelineData, compDuration);

            // Load keyframes for nested clips (important for transforms and effects!)
            const nestedKeyframesMap = new Map<string, Keyframe[]>();
            for (const nestedSerializedClip of composition.timelineData.clips) {
              const nestedClipId = `nested-${compClip.id}-${nestedSerializedClip.id}`;
              if (nestedSerializedClip.keyframes && nestedSerializedClip.keyframes.length > 0) {
                // Update clip IDs in keyframes to match nested clip ID format
                const updatedKeyframes = nestedSerializedClip.keyframes.map((kf: Keyframe) => ({
                  ...kf,
                  clipId: nestedClipId,
                }));
                nestedKeyframesMap.set(nestedClipId, updatedKeyframes);
                log.debug('Loaded keyframes for nested clip in loadState', {
                  nestedClipId,
                  originalClipId: nestedSerializedClip.id,
                  keyframeCount: updatedKeyframes.length,
                });
              }
            }

            // Merge nested keyframes into store
            if (nestedKeyframesMap.size > 0) {
              const currentKeyframes = get().clipKeyframes;
              const mergedKeyframes = new Map(currentKeyframes);
              nestedKeyframesMap.forEach((keyframes, clipId) => {
                mergedKeyframes.set(clipId, keyframes);
              });
              set({ clipKeyframes: mergedKeyframes });
            }

            // Update comp clip with nested data and boundaries
            set(state => ({
              clips: state.clips.map(c =>
                c.id === compClip.id
                  ? { ...c, nestedClips, nestedTracks, nestedClipBoundaries: boundaries, isLoading: false }
                  : c
              ),
            }));

            // Always generate clip segments on project load (new segment-based thumbnail system)
            if (get().thumbnailsEnabled) {
              // Wait for nested clip sources to load, then build segments
              setTimeout(async () => {
                // Get fresh nested clips (they may have updated sources now)
                const freshCompClip = get().clips.find(c => c.id === compClip.id);
                const freshNestedClips = freshCompClip?.nestedClips || nestedClips;

                const clipSegments = await buildClipSegments(
                  composition.timelineData,
                  compDuration,
                  freshNestedClips
                );

                if (clipSegments.length > 0) {
                  set(state => ({
                    clips: state.clips.map(c =>
                      c.id === compClip.id ? { ...c, clipSegments } : c
                    ),
                  }));
                  log.info('Built clip segments on project load', {
                    clipId: compClip.id,
                    segmentCount: clipSegments.length,
                  });
                }
              }, 1000); // Wait longer on project load for all videos to load
            }
          } else {
            // No timeline data
            set(state => ({
              clips: state.clips.map(c =>
                c.id === compClip.id ? { ...c, isLoading: false } : c
              ),
            }));
          }
        } else {
          log.warn('Could not find composition for clip', { clip: serializedClip.name });
        }
        continue;
      }

      // Text clips - restore from textProperties
      if (serializedClip.sourceType === 'text' && serializedClip.textProperties) {
        const { textRenderer } = await import('../../services/textRenderer');
        const { googleFontsService } = await import('../../services/googleFontsService');

        // Load the font first
        await googleFontsService.loadFont(
          serializedClip.textProperties.fontFamily,
          serializedClip.textProperties.fontWeight
        );

        // Render text to canvas
        const textCanvas = textRenderer.render(serializedClip.textProperties);

        const textClip: TimelineClip = {
          id: serializedClip.id,
          trackId: serializedClip.trackId,
          name: serializedClip.name,
          file: new File([], 'text-clip.txt', { type: 'text/plain' }),
          startTime: serializedClip.startTime,
          duration: serializedClip.duration,
          inPoint: serializedClip.inPoint,
          outPoint: serializedClip.outPoint,
          source: {
            type: 'text',
            textCanvas,
            naturalDuration: serializedClip.duration,
          },
          transform: serializedClip.transform,
          effects: serializedClip.effects || [],
          masks: serializedClip.masks,
          textProperties: serializedClip.textProperties,
          isLoading: false,
        };

        // Add clip to state
        set(state => ({
          clips: [...state.clips, textClip],
        }));

        log.debug('Restored text clip', { clip: serializedClip.name });
        continue;
      }

      // Solid clips - restore from solidColor
      if (serializedClip.sourceType === 'solid' && serializedClip.solidColor) {
        const color = serializedClip.solidColor;
        // Use active composition dimensions, fallback to 1920x1080
        const activeComp = mediaStore.getActiveComposition?.();
        const compWidth = activeComp?.width || 1920;
        const compHeight = activeComp?.height || 1080;
        const canvas = document.createElement('canvas');
        canvas.width = compWidth;
        canvas.height = compHeight;
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, compWidth, compHeight);

        const solidClip: TimelineClip = {
          id: serializedClip.id,
          trackId: serializedClip.trackId,
          name: serializedClip.name,
          file: new File([], 'solid-clip.dat', { type: 'application/octet-stream' }),
          startTime: serializedClip.startTime,
          duration: serializedClip.duration,
          inPoint: serializedClip.inPoint,
          outPoint: serializedClip.outPoint,
          source: {
            type: 'solid',
            textCanvas: canvas,
            naturalDuration: serializedClip.duration,
          },
          transform: serializedClip.transform,
          effects: serializedClip.effects || [],
          masks: serializedClip.masks,
          solidColor: color,
          isLoading: false,
        };

        set(state => ({
          clips: [...state.clips, solidClip],
        }));

        log.debug('Restored solid clip', { clip: serializedClip.name, color });
        continue;
      }

      // Regular media clips
      const mediaFile = mediaStore.files.find(f => f.id === serializedClip.mediaFileId);
      if (!mediaFile) {
        log.warn('Media file not found for clip', { clip: serializedClip.name, mediaFileId: serializedClip.mediaFileId });
        continue;
      }

      // Create the clip - even if file is missing (needs reload after refresh)
      const needsReload = !mediaFile.file;
      if (needsReload) {
        log.debug('Clip needs reload (file permission required)', { clip: serializedClip.name });
      }

      // Create placeholder file if missing
      const file = mediaFile.file || new File([], mediaFile.name || 'pending', { type: 'video/mp4' });

      // Create the clip with loading state
      const clip: TimelineClip = {
        id: serializedClip.id,
        trackId: serializedClip.trackId,
        name: serializedClip.name || mediaFile.name || 'Untitled',
        file: file,
        startTime: serializedClip.startTime,
        duration: serializedClip.duration,
        inPoint: serializedClip.inPoint,
        outPoint: serializedClip.outPoint,
        source: {
          type: serializedClip.sourceType,
          mediaFileId: serializedClip.mediaFileId, // Preserve mediaFileId for cache lookups
          naturalDuration: serializedClip.naturalDuration,
        },
        mediaFileId: serializedClip.mediaFileId, // Restore top-level mediaFileId for audio/proxy lookup
        needsReload: needsReload, // Flag for UI to show reload indicator
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

      // Check for cached analysis in project folder if clip doesn't have analysis but has mediaFileId
      if (!serializedClip.analysis && serializedClip.mediaFileId && projectFileService.isProjectOpen()) {
        projectFileService.getAnalysis(
          serializedClip.mediaFileId,
          serializedClip.inPoint,
          serializedClip.outPoint
        ).then(cachedAnalysis => {
          if (cachedAnalysis) {
            log.debug('Loaded analysis from project folder', { clip: serializedClip.name });
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
          log.warn('Failed to load analysis from project folder', err);
        });
      }

      // Skip media loading if file needs reload (no valid File object)
      if (needsReload) {
        log.debug('Skipping media load for clip that needs reload', { clip: clip.name });
        continue;
      }

      // Load media element async
      const type = serializedClip.sourceType;
      const fileUrl = URL.createObjectURL(mediaFile.file!);

      if (type === 'video') {
        const video = document.createElement('video');
        video.src = fileUrl;
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.crossOrigin = 'anonymous';

        video.addEventListener('canplaythrough', async () => {
          // First set up the basic video source
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

          // Pre-cache frame via createImageBitmap for immediate scrubbing without play()
          // createImageBitmap is the ONLY API that decodes a frame from a never-played video after reload
          engine.preCacheVideoFrame(video);

          // Try to initialize WebCodecsPlayer for hardware-accelerated decoding
          const hasWebCodecs = 'VideoDecoder' in window && 'VideoFrame' in window;
          if (hasWebCodecs) {
            try {
              const { WebCodecsPlayer } = await import('../../engine/WebCodecsPlayer');
              log.debug('Initializing WebCodecsPlayer for restored clip', { clip: clip.name });

              const webCodecsPlayer = new WebCodecsPlayer({
                loop: false,
                useSimpleMode: true, // Use VideoFrame from HTMLVideoElement (more compatible)
                onError: (error) => {
                  log.warn('WebCodecs error', { error: error.message });
                },
              });

              // Attach to existing video element
              webCodecsPlayer.attachToVideoElement(video);
              log.debug('WebCodecsPlayer ready for restored clip', { clip: clip.name });

              // Update clip source with webCodecsPlayer
              set(state => ({
                clips: state.clips.map(c =>
                  c.id === clip.id && c.source?.type === 'video'
                    ? {
                        ...c,
                        source: {
                          ...c.source,
                          webCodecsPlayer,
                        },
                      }
                    : c
                ),
              }));
            } catch (err) {
              log.warn('WebCodecsPlayer init failed for restored clip, using HTMLVideoElement', err);
            }
          }
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

    // Clear layers so preview shows black
    set({ layers: [] });

    const { tracks } = get();
    set({
      clips: [],
      selectedClipIds: new Set(),
      primarySelectedClipId: null,
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
      expandedCurveProperties: new Map<string, Set<import('../../types').AnimatableProperty>>(),
    });
  },
});
