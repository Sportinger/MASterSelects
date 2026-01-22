// Clip-related actions slice

import type { TimelineClip, TimelineTrack, Effect, EffectType, TextClipProperties } from '../../types';
import type { ClipActions, SliceCreator, Composition } from './types';
import { useMediaStore } from '../mediaStore';
import { useSettingsStore } from '../settingsStore';
import { DEFAULT_TRANSFORM, DEFAULT_TEXT_PROPERTIES, DEFAULT_TEXT_DURATION } from './constants';
import { generateWaveform, generateThumbnails, getDefaultEffectParams } from './utils';
import { textRenderer } from '../../services/textRenderer';
import { googleFontsService } from '../../services/googleFontsService';
import { WebCodecsPlayer } from '../../engine/WebCodecsPlayer';
import { NativeDecoder } from '../../services/nativeHelper';

// Check if file is a professional codec that needs Native Helper
function isProfessionalCodecFile(file: File): boolean {
  const ext = file.name.toLowerCase().split('.').pop();
  // ProRes typically in .mov, DNxHD in .mxf or .mov
  return ext === 'mov' || ext === 'mxf';
}

// Warm up video decoder by forcing a frame decode
// This eliminates the "cold start" delay on first play
async function warmUpVideoDecoder(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    // Skip if video is already playing or has been decoded
    if (video.readyState >= 3) { // HAVE_FUTURE_DATA or HAVE_ENOUGH_DATA
      resolve();
      return;
    }

    // Use requestVideoFrameCallback if available (modern browsers)
    // This efficiently waits for the decoder to produce a frame
    if ('requestVideoFrameCallback' in video) {
      const warmUp = () => {
        video.currentTime = 0.001; // Seek to first frame (not exactly 0 to trigger decode)
        (video as any).requestVideoFrameCallback(() => {
          // Decoder has now processed at least one frame
          video.pause();
          resolve();
        });
        // Force decode by playing briefly
        video.play().catch(() => resolve());
      };

      if (video.readyState >= 1) { // HAVE_METADATA
        warmUp();
      } else {
        video.addEventListener('loadedmetadata', warmUp, { once: true });
      }
    } else {
      // Fallback: wait for canplay event which indicates decoder is ready
      const videoEl = video as HTMLVideoElement;
      if (videoEl.readyState >= 2) { // HAVE_CURRENT_DATA
        resolve();
        return;
      }
      videoEl.addEventListener('canplay', () => resolve(), { once: true });
      // Trigger buffer by seeking
      videoEl.currentTime = 0.001;
    }

    // Timeout fallback (don't block forever)
    setTimeout(resolve, 500);
  });
}

export const createClipSlice: SliceCreator<ClipActions> = (set, get) => ({
  addClip: async (trackId, file, startTime, providedDuration, mediaFileId) => {
    // Detect file type - use MIME type with fallback to extension
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
    const videoExtensions = ['mp4', 'webm', 'mov', 'avi', 'mkv', 'wmv', 'm4v', 'flv'];
    const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg'];

    const isVideo = file.type.startsWith('video/') || videoExtensions.includes(ext);
    const isAudio = file.type.startsWith('audio/') || audioExtensions.includes(ext);
    const isImage = file.type.startsWith('image/') || imageExtensions.includes(ext);

    console.log(`[Timeline] Adding file: ${file.name}, type: ${file.type}, ext: ${ext}, isAudio: ${isAudio}, isVideo: ${isVideo}`);

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
      // Check if this is a professional codec file that needs Native Helper
      const isProfessional = isProfessionalCodecFile(file);
      const { turboModeEnabled, nativeHelperConnected } = useSettingsStore.getState();
      const useNativeDecoder = isProfessional && turboModeEnabled && nativeHelperConnected;

      let nativeDecoder: NativeDecoder | null = null;
      let video: HTMLVideoElement | null = null;
      let naturalDuration = estimatedDuration;

      if (useNativeDecoder) {
        // Use Native Helper for professional codecs (ProRes, DNxHD)
        try {
          // Get file path from MediaFile or from dropped file
          const mediaFile = mediaFileId ? useMediaStore.getState().files.find(f => f.id === mediaFileId) : null;
          let filePath = mediaFile?.absolutePath || (file as any).path;

          // If no absolute path, try common locations (Linux/Mac)
          if (!filePath || !filePath.startsWith('/')) {
            const commonPaths = [
              `/home/${typeof process !== 'undefined' ? process.env?.USER : 'admin'}/Desktop/${file.name}`,
              `/home/admin/Desktop/${file.name}`,
              `/tmp/${file.name}`,
              `${file.name}`,
            ];
            // Use first path as best guess - helper will validate
            filePath = commonPaths[0];
            console.log(`[Timeline] No absolute path found, trying common locations:`, filePath);
          }

          console.log(`[Timeline] Opening ${file.name} with Native Helper, path:`, filePath);

          nativeDecoder = await NativeDecoder.open(filePath);
          naturalDuration = nativeDecoder.duration;

          console.log(`[Timeline] Native Helper ready: ${nativeDecoder.width}x${nativeDecoder.height} @ ${nativeDecoder.fps}fps, ${naturalDuration.toFixed(2)}s`);

          // Decode initial frame so preview isn't black
          await nativeDecoder.seekToFrame(0);
          console.log(`[Timeline] Initial frame decoded for ${file.name}`);

          // Update clip with NativeDecoder
          updateClip(clipId, {
            duration: naturalDuration,
            outPoint: naturalDuration,
            source: {
              type: 'video',
              naturalDuration,
              mediaFileId,
              nativeDecoder,
              filePath,
            },
            isLoading: false,
          });

          if (audioTrackId && audioClipId) {
            updateClip(audioClipId, {
              duration: naturalDuration,
              outPoint: naturalDuration,
            });
          }
        } catch (err) {
          console.warn(`[Timeline] Native Helper failed for ${file.name}, falling back to browser:`, err);
          nativeDecoder = null;
        }
      }

      // Fallback to HTMLVideoElement if not using native decoder
      if (!nativeDecoder) {
        video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.preload = 'metadata';
        video.muted = true;
        video.crossOrigin = 'anonymous';

        // Wait for metadata
        await new Promise<void>((resolve) => {
          video!.onloadedmetadata = () => resolve();
          video!.onerror = () => resolve();
        });

        naturalDuration = video.duration || estimatedDuration;

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

        // Warm up video decoder in background (non-blocking)
        warmUpVideoDecoder(video).then(() => {
          console.log(`[Timeline] Video decoder warmed up for ${file.name}`);
        });

        // Try to initialize WebCodecsPlayer for hardware-accelerated decoding
        const hasWebCodecs = 'VideoDecoder' in window && 'VideoFrame' in window;

        if (hasWebCodecs) {
          try {
            console.log(`[Timeline] Initializing WebCodecsPlayer for ${file.name}...`);

            const webCodecsPlayer = new WebCodecsPlayer({
              loop: false,
              useSimpleMode: true,
              onError: (error) => {
                console.warn('[Timeline] WebCodecs error:', error.message);
              },
            });

            webCodecsPlayer.attachToVideoElement(video);
            console.log(`[Timeline] WebCodecsPlayer ready for ${file.name}`);

            const currentClips = get().clips;
            set({
              clips: currentClips.map(c => {
                if (c.id !== clipId || !c.source) return c;
                return {
                  ...c,
                  source: {
                    type: c.source.type,
                    videoElement: c.source.videoElement,
                    naturalDuration: c.source.naturalDuration,
                    mediaFileId: c.source.mediaFileId,
                    webCodecsPlayer,
                  },
                };
              }),
            });
          } catch (err) {
            console.warn('[Timeline] WebCodecsPlayer init failed, using HTMLVideoElement:', err);
          }
        }
      }

      // Generate thumbnails in background (non-blocking) - only if enabled
      // Skip for very large files (>500MB) to avoid performance issues
      const LARGE_FILE_THRESHOLD = 500 * 1024 * 1024; // 500MB
      const isLargeFile = file.size > LARGE_FILE_THRESHOLD;

      if (isLargeFile) {
        console.log(`[Thumbnails/Waveform] Skipping for large file (${(file.size / 1024 / 1024).toFixed(0)}MB): ${file.name}`);
      }

      // Skip thumbnail generation for NativeDecoder (no video element) or large files
      if (get().thumbnailsEnabled && !isLargeFile && video) {
        (async () => {
          try {
            // Wait for video to be ready for thumbnails
            await new Promise<void>((resolve) => {
              if (video!.readyState >= 2) {
                resolve();
              } else {
                video!.oncanplay = () => resolve();
                setTimeout(resolve, 2000); // Timeout fallback
              }
            });

            // Check again in case toggle was turned off while waiting
            if (!get().thumbnailsEnabled) return;

            console.log(`[Thumbnails] Starting generation for ${file.name}...`);
            const thumbnails = await generateThumbnails(video!, naturalDuration);
            console.log(`[Thumbnails] Complete: ${thumbnails.length} thumbnails for ${file.name}`);

            // Update clip with thumbnails
            const currentClips = get().clips;
            set({
              clips: currentClips.map(c => c.id === clipId ? { ...c, thumbnails } : c)
            });

            // Seek back to start
            video!.currentTime = 0;
          } catch (e) {
            console.warn('Failed to generate thumbnails:', e);
          }
        })();
      } else if (nativeDecoder) {
        console.log(`[Thumbnails] Skipping for NativeDecoder file: ${file.name} (TODO: implement native thumbnails)`);
      }

      // Load audio - make it ready immediately, waveform loads in background
      // Skip for NativeDecoder files (browser can't decode ProRes/DNxHD audio)
      if (audioTrackId && audioClipId && !nativeDecoder) {
        const audioFromVideo = document.createElement('audio');
        audioFromVideo.src = URL.createObjectURL(file);
        audioFromVideo.preload = 'auto';

        // Mark audio clip as ready immediately
        updateClip(audioClipId, {
          source: { type: 'audio', audioElement: audioFromVideo, naturalDuration, mediaFileId },
          isLoading: false,
        });

        // Generate waveform in background (non-blocking) - only if enabled
        // Skip for large files (>500MB) as generateWaveform loads entire file into memory
        if (get().waveformsEnabled && audioClipId && !isLargeFile) {
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
      } else if (audioTrackId && audioClipId && nativeDecoder) {
        // For NativeDecoder files, mark audio as unavailable (browser can't decode ProRes/DNxHD audio)
        console.log(`[Audio] Skipping audio for NativeDecoder file: ${file.name} (TODO: implement native audio decoding)`);
        updateClip(audioClipId, {
          source: { type: 'audio', naturalDuration, mediaFileId },
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

      // Check if this is a large file before setting waveform state
      // Audio waveform can handle larger files (up to 4GB) since we just need to decode and sample
      const LARGE_AUDIO_THRESHOLD = 4 * 1024 * 1024 * 1024; // 4GB
      const isLargeAudioFile = file.size > LARGE_AUDIO_THRESHOLD;

      // Mark clip as ready first (waveform will load in background)
      updateClip(clipId, {
        duration: naturalDuration,
        outPoint: naturalDuration,
        source: { type: 'audio', audioElement: audio, naturalDuration, mediaFileId },
        isLoading: false,
        waveformGenerating: get().waveformsEnabled && !isLargeAudioFile,
        waveformProgress: 0,
      });

      // Generate waveform in background - only if enabled
      // Skip for very large files (>4GB) to avoid memory issues

      if (isLargeAudioFile) {
        console.log(`[Waveform] Skipping for very large file (${(file.size / 1024 / 1024).toFixed(0)}MB): ${file.name}`);
      }

      if (get().waveformsEnabled && !isLargeAudioFile) {
        (async () => {
          try {
            console.log(`[Waveform] Starting generation for ${file.name}...`);
            const waveform = await generateWaveform(file);
            console.log(`[Waveform] Complete: ${waveform.length} samples for ${file.name}`);

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

    // Use timeline duration if available, otherwise fall back to composition duration
    // This ensures the nested clip matches the timeline duration the user set
    const compDuration = composition.timelineData?.duration ?? composition.duration;

    // Find non-overlapping position
    const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, compDuration);

    // Create placeholder clip immediately (will be updated with nested content)
    const compClip: TimelineClip = {
      id: clipId,
      trackId,
      name: composition.name,
      file: new File([], composition.name), // Placeholder file
      startTime: finalStartTime,
      duration: compDuration,
      inPoint: 0,
      outPoint: compDuration,
      source: {
        type: 'video', // Comp clips are treated as video
        naturalDuration: compDuration,
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
          masks: serializedClip.masks || [],  // Copy masks from source clip
          isLoading: true,
        };

        nestedClips.push(nestedClip);

        // Load media element async
        const type = serializedClip.sourceType;
        const mediaFileRef = mediaFile.file!;  // Capture reference for use in callbacks
        const fileUrl = URL.createObjectURL(mediaFileRef);

        if (type === 'video') {
          const video = document.createElement('video');
          video.src = fileUrl;
          video.muted = true;
          video.playsInline = true;
          video.preload = 'auto';
          video.crossOrigin = 'anonymous';

          video.addEventListener('canplaythrough', async () => {
            // Set up basic video source first
            nestedClip.source = {
              type: 'video',
              videoElement: video,
              naturalDuration: video.duration,
            };
            nestedClip.isLoading = false;

            // Initialize WebCodecsPlayer for hardware-accelerated decoding
            const hasWebCodecs = 'VideoDecoder' in window && 'VideoFrame' in window;
            if (hasWebCodecs) {
              try {
                const { WebCodecsPlayer } = await import('../../engine/WebCodecsPlayer');
                console.log(`[Nested Comp] Initializing WebCodecsPlayer for ${mediaFileRef.name}...`);

                const webCodecsPlayer = new WebCodecsPlayer({
                  loop: false,
                  useSimpleMode: true,
                  onError: (error) => {
                    console.warn('[Nested Comp] WebCodecs error:', error.message);
                  },
                });

                webCodecsPlayer.attachToVideoElement(video);
                console.log(`[Nested Comp] WebCodecsPlayer ready for ${mediaFileRef.name}`);

                // Update nested clip source with webCodecsPlayer
                nestedClip.source = {
                  ...nestedClip.source,
                  webCodecsPlayer,
                };
              } catch (err) {
                console.warn('[Nested Comp] WebCodecsPlayer init failed, using HTMLVideoElement:', err);
              }
            }

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
              const thumbnails = await generateThumbnails(video, compDuration);
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

      // Generate audio mixdown for nested composition
      // Mark as generating first
      const clipsBefore = get().clips;
      set({
        clips: clipsBefore.map(c =>
          c.id === clipId ? { ...c, mixdownGenerating: true } : c
        ),
      });

      // Always create a linked audio clip for compositions (even if no audio)
      // Import dynamically to avoid circular dependencies
      import('../../services/compositionAudioMixer').then(async ({ compositionAudioMixer }) => {
        try {
          console.log(`[Nested Comp] Generating audio mixdown for ${composition.name}...`);
          const mixdownResult = await compositionAudioMixer.mixdownComposition(composition.id);

          // Find an audio track to place the linked audio clip
          const currentState = get();
          const audioTracks = currentState.tracks.filter(t => t.type === 'audio');
          let audioTrackId: string | null = null;

          if (audioTracks.length > 0) {
            // Use the first audio track
            audioTrackId = audioTracks[0].id;
          } else {
            // Create a new audio track
            const newTrackId = `track-${Date.now()}-audio`;
            const newTrack: TimelineTrack = {
              id: newTrackId,
              name: 'Audio 1',
              type: 'audio',
              height: 60,
              muted: false,
              visible: true,
              solo: false,
            };
            set({ tracks: [...currentState.tracks, newTrack] });
            audioTrackId = newTrackId;
            console.log(`[Nested Comp] Created new audio track for ${composition.name}`);
          }

          // Create a linked audio clip (with or without actual audio content)
          const audioClipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-audio`;
          const compClipCurrent = get().clips.find(c => c.id === clipId);

          if (compClipCurrent && audioTrackId) {
            const hasAudio = !!(mixdownResult && mixdownResult.hasAudio);

            // Create audio element - either from mixdown or silent placeholder
            let mixdownAudio: HTMLAudioElement | undefined;
            let waveform: number[] = [];
            let mixdownBuffer: AudioBuffer | undefined;

            if (hasAudio && mixdownResult) {
              mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
              mixdownAudio.preload = 'auto';
              waveform = mixdownResult.waveform;
              mixdownBuffer = mixdownResult.buffer;
            } else {
              // Create silent audio element for empty comp
              mixdownAudio = document.createElement('audio');
              // Generate flat waveform (silence)
              waveform = new Array(Math.max(1, Math.floor(compDuration * 50))).fill(0);
            }

            const audioClip: TimelineClip = {
              id: audioClipId,
              trackId: audioTrackId,
              name: `${composition.name} (Audio)`,
              file: new File([], `${composition.name}-audio.wav`),
              startTime: compClipCurrent.startTime,
              duration: compClipCurrent.duration,
              inPoint: 0,
              outPoint: hasAudio && mixdownResult ? mixdownResult.duration : compDuration,
              source: {
                type: 'audio',
                audioElement: mixdownAudio,
                naturalDuration: hasAudio && mixdownResult ? mixdownResult.duration : compDuration,
              },
              linkedClipId: clipId, // Link to the video comp clip
              waveform,
              transform: { ...DEFAULT_TRANSFORM },
              effects: [],
              isLoading: false,
              isComposition: true, // Mark as composition audio
              compositionId: composition.id,
              mixdownBuffer,
            };

            // Update the video comp clip to link to the audio clip and add both
            const clipsAfter = get().clips;
            set({
              clips: [
                ...clipsAfter.map(c =>
                  c.id === clipId
                    ? {
                        ...c,
                        linkedClipId: audioClipId,
                        mixdownGenerating: false,
                        hasMixdownAudio: hasAudio,
                      }
                    : c
                ),
                audioClip,
              ],
            });
            console.log(`[Nested Comp] Created linked audio clip for ${composition.name} (hasAudio: ${hasAudio})`);
          }
        } catch (e) {
          console.error('[Nested Comp] Failed to generate audio mixdown:', e);
          // Still create an empty linked audio clip on error
          const currentState = get();
          const audioTracks = currentState.tracks.filter(t => t.type === 'audio');
          let audioTrackId: string | null = audioTracks.length > 0 ? audioTracks[0].id : null;

          if (!audioTrackId) {
            const newTrackId = `track-${Date.now()}-audio`;
            const newTrack: TimelineTrack = {
              id: newTrackId,
              name: 'Audio 1',
              type: 'audio',
              height: 60,
              muted: false,
              visible: true,
              solo: false,
            };
            set({ tracks: [...currentState.tracks, newTrack] });
            audioTrackId = newTrackId;
          }

          const audioClipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-audio`;
          const compClipCurrent = get().clips.find(c => c.id === clipId);

          if (compClipCurrent && audioTrackId) {
            const audioClip: TimelineClip = {
              id: audioClipId,
              trackId: audioTrackId,
              name: `${composition.name} (Audio)`,
              file: new File([], `${composition.name}-audio.wav`),
              startTime: compClipCurrent.startTime,
              duration: compClipCurrent.duration,
              inPoint: 0,
              outPoint: compDuration,
              source: {
                type: 'audio',
                audioElement: document.createElement('audio'),
                naturalDuration: compDuration,
              },
              linkedClipId: clipId,
              waveform: new Array(Math.max(1, Math.floor(compDuration * 50))).fill(0),
              transform: { ...DEFAULT_TRANSFORM },
              effects: [],
              isLoading: false,
              isComposition: true,
              compositionId: composition.id,
            };

            const clipsAfter = get().clips;
            set({
              clips: [
                ...clipsAfter.map(c =>
                  c.id === clipId
                    ? { ...c, linkedClipId: audioClipId, mixdownGenerating: false, hasMixdownAudio: false }
                    : c
                ),
                audioClip,
              ],
            });
            console.log(`[Nested Comp] Created empty linked audio clip for ${composition.name} (error fallback)`);
          }
        }
      });
    } else {
      // No timeline data - still create linked audio clip for consistency
      const currentState = get();
      const audioTracks = currentState.tracks.filter(t => t.type === 'audio');
      let audioTrackId: string | null = audioTracks.length > 0 ? audioTracks[0].id : null;

      if (!audioTrackId) {
        const newTrackId = `track-${Date.now()}-audio`;
        const newTrack: TimelineTrack = {
          id: newTrackId,
          name: 'Audio 1',
          type: 'audio',
          height: 60,
          muted: false,
          visible: true,
          solo: false,
        };
        set({ tracks: [...currentState.tracks, newTrack] });
        audioTrackId = newTrackId;
      }

      const audioClipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-audio`;
      const compClipCurrent = currentState.clips.find(c => c.id === clipId);

      if (compClipCurrent && audioTrackId) {
        const audioClip: TimelineClip = {
          id: audioClipId,
          trackId: audioTrackId,
          name: `${composition.name} (Audio)`,
          file: new File([], `${composition.name}-audio.wav`),
          startTime: compClipCurrent.startTime,
          duration: compClipCurrent.duration,
          inPoint: 0,
          outPoint: compDuration,
          source: {
            type: 'audio',
            audioElement: document.createElement('audio'),
            naturalDuration: compDuration,
          },
          linkedClipId: clipId,
          waveform: new Array(Math.max(1, Math.floor(compDuration * 50))).fill(0),
          transform: { ...DEFAULT_TRANSFORM },
          effects: [],
          isLoading: false,
          isComposition: true,
          compositionId: composition.id,
        };

        set({
          clips: [
            ...currentState.clips.map(c =>
              c.id === clipId
                ? { ...c, linkedClipId: audioClipId, isLoading: false, hasMixdownAudio: false }
                : c
            ),
            audioClip,
          ],
        });
        console.log(`[Nested Comp] Created empty linked audio clip for ${composition.name} (no timeline data)`);
      } else {
        // Fallback - just mark as loaded
        const currentClips = get().clips;
        set({
          clips: currentClips.map(c =>
            c.id === clipId ? { ...c, isLoading: false } : c
          ),
        });
      }
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
    const { clips, tracks, updateDuration, getSnappedPosition, getPositionWithResistance, trimOverlappingClips, invalidateCache } = get();
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

    // Use resistance-based positioning - allows overlap if user pushes through
    const { startTime: finalStartTime, forcingOverlap } = getPositionWithResistance(
      id,
      snappedTime,
      targetTrackId,
      movingClip.duration
    );

    // Calculate time delta to apply to linked clips
    const timeDelta = finalStartTime - movingClip.startTime;

    // For linked clip (1:1 video-audio pair), also calculate position with resistance
    const linkedClip = clips.find(c => c.id === movingClip.linkedClipId || c.linkedClipId === id);
    let linkedFinalTime = linkedClip ? linkedClip.startTime + timeDelta : 0;
    let linkedForcingOverlap = false;
    if (linkedClip && !skipLinked) {
      const linkedResult = getPositionWithResistance(
        linkedClip.id,
        linkedClip.startTime + timeDelta,
        linkedClip.trackId,
        linkedClip.duration
      );
      linkedFinalTime = linkedResult.startTime;
      linkedForcingOverlap = linkedResult.forcingOverlap;
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
          const groupResult = getPositionWithResistance(
            c.id,
            c.startTime + timeDelta,
            c.trackId,
            c.duration
          );
          return {
            ...c,
            startTime: Math.max(0, groupResult.startTime),
          };
        }
        return c;
      }),
    });

    // If user forced overlap, trim the underlying clips
    if (forcingOverlap) {
      trimOverlappingClips(id, finalStartTime, targetTrackId, movingClip.duration);
    }
    if (linkedForcingOverlap && linkedClip && !skipLinked) {
      trimOverlappingClips(linkedClip.id, linkedFinalTime, linkedClip.trackId, linkedClip.duration);
    }

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

  // Split clips at the playhead position
  // If clips are selected, split only selected clips at playhead
  // If no clips are selected, split ALL clips at playhead (like standard NLE behavior)
  splitClipAtPlayhead: () => {
    const { clips, playheadPosition, selectedClipIds, splitClip } = get();

    // Find clips at the current playhead position (excluding linked clips to avoid double-split)
    const clipsAtPlayhead = clips.filter(c =>
      playheadPosition > c.startTime &&
      playheadPosition < c.startTime + c.duration
    );

    if (clipsAtPlayhead.length === 0) {
      console.warn('[Timeline] No clip at playhead position');
      return;
    }

    // Determine which clips to split
    let clipsToSplit: typeof clipsAtPlayhead;

    if (selectedClipIds.size > 0) {
      // Split only selected clips that are at playhead
      clipsToSplit = clipsAtPlayhead.filter(c => selectedClipIds.has(c.id));
      if (clipsToSplit.length === 0) {
        // No selected clips at playhead - fall back to splitting all clips at playhead
        clipsToSplit = clipsAtPlayhead;
      }
    } else {
      // No selection - split ALL clips at playhead (standard NLE behavior)
      clipsToSplit = clipsAtPlayhead;
    }

    // Filter out clips that will be split via their linked clip (to avoid double-splitting)
    const linkedClipIds = new Set(clipsToSplit.map(c => c.linkedClipId).filter(Boolean));
    const clipsToSplitFiltered = clipsToSplit.filter(c => !linkedClipIds.has(c.id));

    // Split each clip
    for (const clip of clipsToSplitFiltered) {
      splitClip(clip.id, playheadPosition);
    }
  },

  updateClip: (id, updates) => {
    const { clips, updateDuration } = get();
    set({
      clips: clips.map(c => c.id === id ? { ...c, ...updates } : c)
    });
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

  // ========== TEXT CLIP ACTIONS ==========

  addTextClip: async (trackId, startTime, duration = DEFAULT_TEXT_DURATION) => {
    const { clips, tracks, updateDuration, invalidateCache } = get();
    const track = tracks.find(t => t.id === trackId);

    // Text clips can only go on video tracks
    if (!track || track.type !== 'video') {
      console.warn('[Timeline] Text clips can only be added to video tracks');
      return null;
    }

    const clipId = `clip-text-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Pre-load the default font
    await googleFontsService.loadFont(DEFAULT_TEXT_PROPERTIES.fontFamily, DEFAULT_TEXT_PROPERTIES.fontWeight);

    // Create canvas for text rendering
    const canvas = textRenderer.createCanvas(1920, 1080);

    // Initial render with default text
    textRenderer.render(DEFAULT_TEXT_PROPERTIES, canvas);

    const textClip: TimelineClip = {
      id: clipId,
      trackId,
      name: 'Text',
      file: new File([], 'text-clip.txt', { type: 'text/plain' }), // Placeholder file
      startTime,
      duration,
      inPoint: 0,
      outPoint: duration,
      source: {
        type: 'text',
        textCanvas: canvas,
        naturalDuration: duration,
      },
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

        // Load font if changed
        if (props.fontFamily || props.fontWeight) {
          googleFontsService.loadFont(
            props.fontFamily || c.textProperties.fontFamily,
            props.fontWeight || c.textProperties.fontWeight
          );
        }

        // Create new canvas for re-render (ensures GPU texture cache is invalidated)
        // The TextureManager caches by canvas reference, so a new canvas = new texture upload
        const canvas = textRenderer.createCanvas(1920, 1080);
        textRenderer.render(newProps, canvas);

        return {
          ...c,
          textProperties: newProps,
          source: {
            ...c.source!,
            textCanvas: canvas,
          },
          // Update name to reflect text content (first 20 chars)
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

  setClipEffectEnabled: (clipId, effectId, enabled) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? {
              ...c,
              effects: c.effects.map(e =>
                e.id === effectId ? { ...e, enabled } : e
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

  generateWaveformForClip: async (clipId: string) => {
    const { clips } = get();
    const clip = clips.find(c => c.id === clipId);

    if (!clip) {
      console.warn('[Waveform] Clip not found:', clipId);
      return;
    }

    // Check if already generating
    if (clip.waveformGenerating) {
      console.log('[Waveform] Already generating for clip:', clipId);
      return;
    }

    // Mark as generating
    set({
      clips: get().clips.map(c =>
        c.id === clipId ? { ...c, waveformGenerating: true, waveformProgress: 0 } : c
      ),
    });

    console.log(`[Waveform] Starting generation for ${clip.name}`);

    try {
      let waveform: number[];

      // Handle composition audio clips differently - use mixdown buffer
      if (clip.isComposition && clip.compositionId) {
        // For composition clips, regenerate mixdown and extract waveform
        const { compositionAudioMixer } = await import('../../services/compositionAudioMixer');
        const { generateWaveformFromBuffer } = await import('./utils');

        console.log(`[Waveform] Generating mixdown for composition ${clip.name}...`);
        const mixdownResult = await compositionAudioMixer.mixdownComposition(clip.compositionId);

        if (mixdownResult && mixdownResult.hasAudio) {
          waveform = mixdownResult.waveform;
          // Also update the mixdown buffer and audio element
          const mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
          set({
            clips: get().clips.map(c =>
              c.id === clipId
                ? {
                    ...c,
                    source: {
                      type: 'audio' as const,
                      audioElement: mixdownAudio,
                      naturalDuration: mixdownResult.duration,
                    },
                    mixdownBuffer: mixdownResult.buffer,
                    hasMixdownAudio: true,
                  }
                : c
            ),
          });
        } else if (clip.mixdownBuffer) {
          // Use existing mixdown buffer
          waveform = generateWaveformFromBuffer(clip.mixdownBuffer, 50);
        } else {
          // No audio - generate flat waveform
          waveform = new Array(Math.max(1, Math.floor(clip.duration * 50))).fill(0);
        }
      } else if (!clip.file) {
        console.warn('[Waveform] No file found for clip:', clipId);
        set({
          clips: get().clips.map(c =>
            c.id === clipId ? { ...c, waveformGenerating: false } : c
          ),
        });
        return;
      } else {
        // Regular audio/video clip - decode from file
        waveform = await generateWaveform(
          clip.file,
          50, // samples per second
          (progress, partialWaveform) => {
            // Update progress and partial waveform in real-time
            set({
              clips: get().clips.map(c =>
                c.id === clipId
                  ? { ...c, waveformProgress: progress, waveform: partialWaveform }
                  : c
              ),
            });
          }
        );
      }

      console.log(`[Waveform] Complete: ${waveform.length} samples for ${clip.name}`);

      // Final update with complete waveform
      set({
        clips: get().clips.map(c =>
          c.id === clipId
            ? { ...c, waveform, waveformGenerating: false, waveformProgress: 100 }
            : c
        ),
      });
    } catch (e) {
      console.error('[Waveform] Failed to generate:', e);
      set({
        clips: get().clips.map(c =>
          c.id === clipId ? { ...c, waveformGenerating: false } : c
        ),
      });
    }
  },

  // Parenting (pick whip) - set a clip's parent for transform inheritance
  setClipParent: (clipId: string, parentClipId: string | null) => {
    const { clips } = get();

    // Can't parent to self
    if (parentClipId === clipId) {
      console.warn('[Parenting] Cannot parent clip to itself');
      return;
    }

    // Cycle detection: ensure parent chain doesn't lead back to this clip
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

    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, parentClipId: parentClipId || undefined } : c
      ),
    });

    console.log(`[Parenting] Set parent of ${clipId} to ${parentClipId || 'none'}`);
  },

  // Get all clips that have the given clip as their parent
  getClipChildren: (clipId: string) => {
    const { clips } = get();
    return clips.filter(c => c.parentClipId === clipId);
  },

  // Set clip preservesPitch property (for audio pitch correction when speed changes)
  setClipPreservesPitch: (clipId: string, preservesPitch: boolean) => {
    const { clips } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, preservesPitch } : c
      ),
    });
  },

  // Add a pending download clip (YouTube videos being downloaded)
  addPendingDownloadClip: (trackId, startTime, videoId, title, thumbnail, estimatedDuration = 30) => {
    const { clips, tracks, updateDuration, findNonOverlappingPosition } = get();

    const track = tracks.find(t => t.id === trackId);
    if (!track || track.type !== 'video') {
      console.warn('[Timeline] Pending download clips can only be added to video tracks');
      return '';
    }

    const clipId = `clip-yt-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Find non-overlapping position
    const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, estimatedDuration);

    const pendingClip: TimelineClip = {
      id: clipId,
      trackId,
      name: title,
      file: new File([], `${title}.mp4`, { type: 'video/mp4' }), // Placeholder file
      startTime: finalStartTime,
      duration: estimatedDuration,
      inPoint: 0,
      outPoint: estimatedDuration,
      source: null, // No source until download complete
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

    console.log(`[Timeline] Added pending download clip: ${clipId} for video ${videoId}`);
    return clipId;
  },

  // Update download progress for a pending clip
  updateDownloadProgress: (clipId, progress) => {
    const { clips } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, downloadProgress: progress } : c
      ),
    });
  },

  // Complete the download - replace pending clip with actual video
  completeDownload: async (clipId, file) => {
    const { clips, updateDuration, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);

    if (!clip || !clip.isPendingDownload) {
      console.warn('[Timeline] Clip not found or not a pending download:', clipId);
      return;
    }

    console.log(`[Timeline] Completing download for clip: ${clipId}`);

    // Create video element for the downloaded file
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    const url = URL.createObjectURL(file);
    video.src = url;

    await new Promise<void>((resolve, reject) => {
      video.addEventListener('loadedmetadata', () => resolve(), { once: true });
      video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
      video.load();
    });

    const naturalDuration = video.duration || 30;

    // Generate thumbnails for the video
    const thumbCount = Math.max(1, Math.min(20, Math.ceil(naturalDuration / 3)));
    const thumbnails: string[] = [];
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 90;
    const ctx = canvas.getContext('2d')!;

    for (let i = 0; i < thumbCount; i++) {
      const time = (i / thumbCount) * naturalDuration;
      video.currentTime = time;
      await new Promise<void>(resolve => {
        video.onseeked = () => {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          thumbnails.push(canvas.toDataURL('image/jpeg', 0.6));
          resolve();
        };
      });
    }

    // Reset to start and wait for video to be ready
    video.currentTime = 0;
    await new Promise<void>(resolve => {
      if (video.readyState >= 2) {
        resolve();
      } else {
        video.addEventListener('canplay', () => resolve(), { once: true });
      }
    });

    // Update the clip with actual video data
    set({
      clips: clips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          file,
          duration: naturalDuration,
          outPoint: naturalDuration,
          source: {
            type: 'video' as const,
            videoElement: video,
            naturalDuration,
          },
          thumbnails,
          isPendingDownload: false,
          downloadProgress: undefined,
          youtubeVideoId: undefined,
          youtubeThumbnail: undefined,
        };
      }),
    });

    updateDuration();
    invalidateCache();

    // Import to media store and get the mediaFileId
    const mediaStore = useMediaStore.getState();
    const mediaFile = await mediaStore.importFile(file);

    // Update clip with mediaFileId so audio can be loaded
    set({
      clips: get().clips.map(c =>
        c.id === clipId ? { ...c, mediaFileId: mediaFile.id } : c
      ),
    });

    console.log(`[Timeline] Download complete for clip: ${clipId}, duration: ${naturalDuration}s, mediaFileId: ${mediaFile.id}`);
  },

  // Set download error for a clip
  setDownloadError: (clipId, error) => {
    const { clips } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, downloadError: error, isPendingDownload: false }
          : c
      ),
    });
  },
});
